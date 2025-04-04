from pathlib import Path
import polars as pl
import asyncio
import json
import pyinstrument
from tqdm.auto import tqdm

from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed


async def process_line(line: str, schema: pl.Schema, categories: dict[str, list[str]]) -> pl.DataFrame:
    data = json.loads(line)
    event = data["events"][0]
    updates = event["updates"]

    # Create a DataFrame from the updates with predefined schema
    df = pl.DataFrame(updates, schema=schema)

    # Add common columns to the DataFrame with predefined categories
    df = df.with_columns(
        [
            pl.lit(data["channel"]).alias("channel"),
            pl.lit(data["timestamp"]).alias("timestamp"),
            pl.lit(data["sequence_num"]).alias("sequence_num"),
            pl.lit(event["type"]).alias("type"),
            pl.lit(event["product_id"]).alias("product_id"),
        ]
    )

    # Convert string columns to categorical types
    for col, cats in categories.items():
        df = df.with_columns(pl.col(col).cast(pl.Categorical).cat.set_ordering(cats))

    return df


async def main(folder: Path = Path('data/L2/BTC-USD/'), msg_type: str = 'update') -> None:
    input_file = folder / f'2024-08-04T23-00-00_{msg_type}.jsonl'
    input_file = folder / f'2024-08-05T04-26-38_{msg_type}.jsonl'
    output_file = input_file.with_suffix('.parquet')

    EXPLODE = None
    EXPLODE = (
        ['updates']
        if 'update' in msg_type
        else ['trades'] if 'trades' in msg_type else None
    )

    print(f"Reading {input_file}")
    df_orig = pl.read_ndjson(input_file)
    print("df_orig", df_orig)
    df = await CoinbaseFeed.polarize(df=df_orig, explode=EXPLODE)
    print("polarized", df)

    df.write_parquet(output_file)
    print(f"Data has been written to {output_file}")

    with pyinstrument.Profiler() as profiler:
        df_read = pl.read_parquet(output_file)
        df_read = await CoinbaseFeed.depolarize(df_read, regroup=EXPLODE)
        print(df_read)
    # profiler.open_in_browser(timeline=True)

    assert df_read.equals(
        df_orig.with_columns(
            [
                pl.col('channel').cast(
                    pl.Enum(['l2_data', 'market_trades', 'subscriptions'])
                ),
                pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ"),
                pl.col('sequence_num').cast(pl.Int64),
            ]
        )
    )


async def merge(msg_type: str) -> pl.DataFrame:
    with pl.StringCache():
        df: pl.DataFrame | None = None
        for folder in Path('data/L2').iterdir():
            if folder.is_dir():
                for tstr in [
                    # '2024-08-04T23-00-00',
                    '2024-08-05T00-00-00',
                    '2024-08-05T01-00-00',
                    '2024-08-05T02-00-00',
                    '2024-08-05T03-00-00',
                    '2024-08-05T04-00-00',
                    '2024-08-05T04-26-38',
                ]:
                    input_file = folder / f'{tstr}_{msg_type}.jsonl'
                    print(f"Reading {input_file}")
                    EXPLODE = (
                        ['updates']
                        if 'update' in msg_type
                        else ['trades'] if 'trades' in msg_type else None
                    )
                    df_part = await CoinbaseFeed.polarize(
                        jsonl_path=input_file, explode=EXPLODE
                    )
                    print(df_part)
                    df = (
                        df.merge_sorted(df_part, key='timestamp')
                        if df is not None
                        else df_part
                    )
        return df
    
async def resort_by_ts() -> None:
    for filename in Path('/media/photoDS216/crypto').iterdir():
        if filename.is_file() and filename.suffix == '.parquet':
            print(f"Reading {filename}")
            df = pl.read_parquet(filename)
            df = df.sort(by='timestamp')
            print("rewriting", filename)
            df.write_parquet(filename)

if __name__ == '__main__':
    import asyncio

    # asyncio.run(resort_by_ts())
    # exit()

    # asyncio.run(main())

    # for msg_type in ['trades', 'update']:
    #     for folder in Path('data/L2').iterdir():
    #         if folder.is_dir():
    #             asyncio.run(main(folder=folder, msg_type=msg_type))

    # dfall = pl.read_parquet('data/2024-08-05T20-52-37_all.parquet')
    # print(dfall.schema)

    with pl.StringCache():
        df_trades = asyncio.run(merge(msg_type='trades'))
        df_books = asyncio.run(merge(msg_type='update'))

        df_all = df_books.merge_sorted(df_trades, key='timestamp')

    print(df_all)
    df_all.write_parquet('data/2024-08-05T00-00-00.parquet')
