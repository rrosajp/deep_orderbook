import asyncio
import collections
from pathlib import Path
from typing import AsyncGenerator, Iterator, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import polars as pl

from coinbase.websocket import WSClient
from deep_orderbook import marketdata as md
from deep_orderbook.config import FeedConfig
from deep_orderbook.feeds.base_feed import BaseFeed, EndFeed
from deep_orderbook.replayer import EndReplay
from deep_orderbook.utils import logger


class Settings(BaseSettings):
    api_key: str = ''
    api_secret: str = ''
    model_config = SettingsConfigDict(
        env_file='credentials/coinbase.txt', env_file_encoding='utf-8'
    )


class SubscriptionsEvent(BaseModel):
    class Subscription(BaseModel):
        level2: list[str] | None = None
        market_trades: list[str] | None = None

    subscriptions: Subscription


class CoinbasePriceLevel(md.OrderLevel):
    side: Literal['bid', 'offer']
    # event_time: datetime = Field(alias='event_time')


class L2Event(BaseModel):
    type: Literal['snapshot', 'update']
    product_id: str = Field(alias='product_id')
    updates: list[CoinbasePriceLevel]

    def book_update(self) -> md.BookUpdate:
        update = md.BookUpdate(
            bids=[x for x in self.updates if x.side == 'bid'],
            asks=[x for x in self.updates if x.side == 'offer'],
        )
        return update

    @property
    def pair(self) -> str:
        return self.product_id


class TradeEvent(BaseModel):
    type: Literal['snapshot', 'update']
    trades: list[md.Trade]

    @property
    def product_id(self) -> str:
        # check it is the same symbol in all trades
        assert (
            len(set([trade.product_id for trade in self.trades])) == 1
        ), f"Not the same symbol in  trades: {self.model_dump_json()}"
        return self.trades[0].product_id


class CoinbaseMessage(md.Message):
    channel: Literal['l2_data', 'market_trades', 'subscriptions', 'heartbeats']
    timestamp: datetime
    sequence_num: int = Field(alias='sequence_num')
    events: list[L2Event] | list[TradeEvent] | list[SubscriptionsEvent]

    @property
    def symbol(self) -> str:
        # check it is the same symbol in all events
        assert (
            len(set([event.product_id for event in self.events])) == 1
        ), f"Not the same symbol in events: {self.model_dump_json()}"
        return self.events[0].product_id

    def is_book_update(self) -> bool:
        return self.channel == 'l2_data'

    def is_trade_update(self) -> bool:
        return self.channel == 'market_trades'

    def is_subscription(self) -> bool:
        return self.channel == 'subscriptions'


class CoinbaseFeed(BaseFeed):
    PRINT_EVENTS = False
    PRINT_MESSAGE = False

    def __init__(
        self,
        config: FeedConfig = FeedConfig(),
        markets: list[str] | None = None,
        feed_msg_queue: bool = False,
        replayer: Iterator[CoinbaseMessage] | None = None,
    ) -> None:
        settings = Settings()
        self.config = config
        self.feed_msg_queue = feed_msg_queue
        self.markets = markets or config.markets
        self.msg_history: list[str] = []
        self.depth_managers: dict[str, md.DepthCachePlus] = {
            s: md.DepthCachePlus() for s in self.markets
        }
        self.trade_tapes: dict[str, list[md.Trade]] = collections.defaultdict(list)
        self.run_timer = False
        self.feed_time: datetime | None = None
        self.next_cut_time = 0.0
        self.queue: asyncio.Queue[CoinbaseMessage] = asyncio.Queue()
        self.queue_one_sec: asyncio.Queue[md.MulitSymbolOneSecondEnds] = asyncio.Queue(
            16
        )
        self.is_live = False

        if not replayer:
            self.is_live = True
            self._client = WSClient(
                api_key=settings.api_key,
                api_secret=settings.api_secret,
                on_message=self._on_message,
                on_open=self.on_open,
                on_close=self.on_close,
            )
        else:
            self._client = replayer
            self._client.on_message = self._on_polars  # type: ignore[attr-defined]

    async def __aenter__(self) -> 'CoinbaseFeed':
        await self._client.open_async()
        # Subscribe to the necessary channels, adjust according to your requirements
        await self._client.subscribe_async(
            product_ids=self.markets,
            channels=[
                'level2',
                'market_trades',
                # 'ticker',
                # 'hearbeats',
            ],
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.run_timer = False
        await self._client.unsubscribe_all_async()
        await self._client.close_async()
        await self.close_queue()

    async def close_queue(self) -> None:
        self.queue.put_nowait(EndFeed())  # type: ignore[arg-type]
        # await self.queue.()

    @classmethod
    async def polarize(
        cls,
        *,
        jsonl_path: Path | None = None,
        json_str: str | None = None,
        df: pl.DataFrame | None = None,
        explode: list[str] | None = None,
    ) -> pl.DataFrame:
        if json_str:
            df = pl.DataFrame([CoinbaseMessage.model_validate_json(json_str)])
        elif jsonl_path:
            df = pl.read_ndjson(jsonl_path)
        else:
            assert df is not None

        df = df.with_columns(
            [
                pl.col('channel').cast(
                    pl.Enum(['l2_data', 'market_trades', 'subscriptions'])
                ),
                pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ"),
                pl.col('sequence_num').cast(pl.Int64),
            ]
        )
        explode = explode or []
        if explode:
            df = df.explode('events')
            df = df.unnest('events')
            if 'updates' in explode:
                df = df.explode('updates')
                df = df.unnest('updates')
            if 'trades' in explode:
                df = df.explode('trades')
                df = df.unnest('trades')
            df = df.with_columns(
                [
                    pl.col('type').cast(pl.Enum(['snapshot', 'update'])),
                    pl.col('product_id').cast(pl.Categorical),
                    pl.col('side').cast(pl.Enum(['bid', 'offer', 'BUY', 'SELL'])),
                ]
            )
        return df

    @classmethod
    async def depolarize(
        cls,
        df: pl.DataFrame,
        regroup: list[str] | None = None,
    ) -> pl.DataFrame:
        if regroup:
            if 'updates' in regroup:
                # Reconstruct the 'updates' nested structure
                df = df.group_by(
                    ['channel', 'timestamp', 'sequence_num', 'product_id', 'type'],
                    maintain_order=True,
                ).agg([pl.struct(['price', 'size', 'side']).alias('updates')])
                # Reconstruct the 'events' nested structure
                df = df.group_by(
                    ['channel', 'timestamp', 'sequence_num'],
                    maintain_order=True,
                ).agg([pl.struct(['type', 'product_id', 'updates']).alias('events')])
            if 'trades' in regroup:
                df = df.group_by(
                    ['channel', 'timestamp', 'sequence_num', 'type'],
                    maintain_order=True,
                ).agg(
                    [pl.struct(['product_id', 'price', 'size', 'side']).alias('trades')]
                )
                df = df.group_by(
                    ['channel', 'timestamp', 'sequence_num'],
                    maintain_order=True,
                ).agg([pl.struct(['type', 'trades']).alias('events')])
        return df

    def cut_trade_tape(self) -> dict[str, list[md.Trade]]:
        logger.debug("cutting trade tapes")
        for symbol, tape in self.trade_tapes.items():
            self.depth_managers[symbol].trade_bunch.trades = [t for t in tape]
        self.trade_tapes = collections.defaultdict(list)
        return {
            symbol: self.depth_managers[symbol].trade_bunch.trades
            for symbol in self.markets
        }

    async def __aiter__(self) -> AsyncGenerator[CoinbaseMessage, None]:
        while True:
            msg = await self.queue.get()
            # logger.debug(f"yielding message from queue: {msg}")
            if isinstance(msg, EndFeed):
                return
            yield msg

    async def _on_polars(
        self, t_win: datetime, df_per_time: pl.DataFrame | EndReplay
    ) -> None:
        if isinstance(df_per_time, EndReplay):
            logger.warning("Received EndReplayFeed message")
            await self.queue_one_sec.put(None)  # type: ignore[arg-type]
            return
        # print(t_win, df_per_time)
        df_books = await self.depolarize(
            df_per_time.filter(pl.col('channel') == 'l2_data'), regroup=['updates']
        )
        df_trade = await self.depolarize(
            df_per_time.filter(pl.col('channel') == 'market_trades'), regroup=['trades']
        )

        self.feed_time = t_win

        for msg in (CoinbaseMessage(**row) for row in df_books.iter_rows(named=True)):
            self.process_message(msg)
        for msg in (CoinbaseMessage(**row) for row in df_trade.iter_rows(named=True)):
            self.process_message(msg)

        onesec = self.on_one_second_end()
        await self.queue_one_sec.put(onesec)
        await asyncio.sleep(0.00001)
        if qs := self.queue_one_sec.qsize():
            logger.info(f"queue_one_sec size: {qs}")

    def _on_message(self, msg: str | dict[str, Any]) -> None:
        if isinstance(msg, str):
            try:
                message = CoinbaseMessage.model_validate_json(msg)
            except ValidationError as e:
                logger.error(f"Failed to validate message: {msg}")
                logger.error(e)
                return
        else:
            message = msg

        if self.feed_msg_queue:
            self.queue.put_nowait(message)

        # if we passed to the next (or several) seconds, we send to the queue
        number_of_time_steps_past_last_message = (
            (
                int(message.timestamp.timestamp() * self.config.freq)
                - int(self.feed_time.timestamp() * self.config.freq)
            )
            if self.feed_time
            else None
        )
        while number_of_time_steps_past_last_message:
            number_of_time_steps_past_last_message -= 1
            onesec = self.on_one_second_end()
            if not self.queue_one_sec.full():
                self.queue_one_sec.put_nowait(onesec)

        self.feed_time = message.timestamp
        self.process_message(message)

    def process_message(self, message: CoinbaseMessage | EndFeed) -> None:
        if isinstance(message, EndFeed):
            logger.warning("Received EndFeed message")
            return

        assert len(message.events) <= 1
        match message.channel:
            case 'heartbeats':
                if self.PRINT_EVENTS:
                    logger.debug(f"timer message: {message}")
            case 'subscriptions':
                for event in message.events:
                    assert isinstance(event, SubscriptionsEvent) and event.subscriptions
                    logger.info("Successfully subscribed to the following channels:")
                    for subscription in event.subscriptions:
                        logger.info(f"{subscription=}")
                return
            case 'l2_data':
                for event in message.events:
                    assert isinstance(event, L2Event)
                    if self.PRINT_MESSAGE:
                        logger.debug(
                            f"{message.channel}       {event.type} {event.product_id} {message.timestamp}: {len(event.updates)}"
                        )
                    if self.PRINT_EVENTS:
                        logger.debug(event)
                    match event.type:
                        case 'snapshot':
                            self.depth_managers[event.product_id].reset(
                                event.book_update()
                            )
                        case 'update':
                            self.depth_managers[event.product_id].update(
                                event.book_update()
                            )
            case 'market_trades':
                assert len(message.events) == 1
                assert isinstance(message.events[0], TradeEvent) and (
                    len(set([trade.product_id for trade in message.events[0].trades]))
                    == 1
                )
                for event in message.events:
                    assert isinstance(event, TradeEvent)
                    if self.PRINT_MESSAGE:
                        logger.debug(
                            f"{message.channel} {event.type} {message.timestamp}: {len(event.trades)}"
                        )
                    if self.PRINT_EVENTS:
                        logger.debug(event)
                    match event.type:
                        case 'snapshot':
                            pass
                        case 'update':
                            for trade in event.trades:
                                self.trade_tapes[trade.product_id].append(trade)
            case _:
                logger.error(f"Unhandled channel: {message.channel}")

    def on_open(self) -> None:
        logger.info("Connection opened!")

    def on_close(self) -> None:
        logger.info("Connection closed!")

    def on_one_second_end(self) -> md.MulitSymbolOneSecondEnds:
        """this function is used to run the replay of the market data in parallel for all the symbols."""
        self.cut_trade_tape()
        oneSec = md.MulitSymbolOneSecondEnds.make_one_second(
            ts=self.feed_time,
            depth_managers=self.depth_managers,
        )
        return oneSec

    async def one_second_iterator(
        self,
    ) -> AsyncGenerator[md.MulitSymbolOneSecondEnds, None]:
        num_samples = self.config.max_samples
        while num_samples != 0:
            num_samples -= 1
            one_sec = await self.queue_one_sec.get()
            if not one_sec:
                return
            yield one_sec


async def main() -> None:
    import pyinstrument

    async with CoinbaseFeed(
        markets=['ETH-USD', 'BTC-USD'], feed_msg_queue=True
    ) as feed:
        # feed.PRINT_MESSAGE = True
        await asyncio.sleep(10)

    with pyinstrument.Profiler() as profiler:
        async for msg in feed:
            feed.process_message(msg)

    # profiler.print(show_all=True)
    profiler.open_in_browser(timeline=False)


if __name__ == '__main__':
    asyncio.run(main())
