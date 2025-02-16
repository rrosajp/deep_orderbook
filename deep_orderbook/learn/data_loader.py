# data_loader.py

import threading
import asyncio
import queue
from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger
import numpy as np
class DataLoaderWorker:
    """Data loading worker that reads data from files and puts it into a queue."""

    def __init__(
        self, *, data_queue: queue.Queue, replay_config: ReplayConfig, shaper_config: ShaperConfig
    ) -> None:
        self.data_queue = data_queue
        self.replay_config = replay_config
        self.shaper_config = shaper_config
        self._running = True
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the data loading worker in a separate thread."""
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread gracefully."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def run(self) -> None:
        """Worker function to load data and put it into the queue."""
        from deep_orderbook.shaper import iter_shapes_t2l

        while self._running:
            try:
                rand_replay_config = self.replay_config.but_random_file()
                logger.debug(f"Loading data from {rand_replay_config.file_list()}")

                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def load_data() -> None:
                    # probability_of_skipping = 0.9
                    async for books_array, time_levels, pxar in iter_shapes_t2l(
                        replay_config=rand_replay_config,
                        shaper_config=self.shaper_config,
                    ):
                        # if random.random() < probability_of_skipping:
                        #     # logger.warning(f"[DataLoader] Skipping data {probability_of_skipping=}")
                        #     continue
                        
                        try:
                            self.data_queue.put(
                                (books_array, time_levels, pxar),
                                # block=False,
                            )
                        except queue.Full:
                            # logger.warning(f"[DataLoader] Queue is full, skipping data {probability_of_skipping=}")
                            # probability_of_skipping *= 1.1
                            pass
                        # Optional sleep interval
                        await asyncio.sleep(0.001)
                    logger.debug("load_data completed")

                loop.run_until_complete(load_data())
                logger.debug(
                    f"Data loading completed for {rand_replay_config.file_list()}"
                )
            except Exception as e:
                logger.error(f"Exception in data loading worker: {e}")
                print(
                    f"Exception in data loading worker: {e}\n with :{rand_replay_config.file_list()}"
                )
            finally:
                loop.close()


async def main() -> None:
    # Example usage of DataLoaderWorker
    from deep_orderbook.config import ReplayConfig, ShaperConfig

    replay_config = ReplayConfig(date_regexp="2024-0")
    shaper_config = ShaperConfig(only_full_arrays=False)
    print(f"{replay_config.file_list()=}")

    # Create a queue for data
    data_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray]] = queue.Queue(maxsize=1000)

    # Initialize and start the data loader worker
    data_loader_worker = DataLoaderWorker(
        data_queue=data_queue, replay_config=replay_config, shaper_config=shaper_config
    )
    num_workers = 4
    for _ in range(num_workers):
        data_loader_worker = DataLoaderWorker(
            data_queue=data_queue,
            replay_config=replay_config,
            shaper_config=shaper_config,
        )
        data_loader_worker.start()

    # Consume data from the queue
    try:
        while True:
            data = data_queue.get(timeout=30)
            books_array, time_levels, pxar = data
            if qs := data_queue.qsize():
                logger.info(
                    f"Queue size: {qs}, books: {books_array.shape}, t2l: {time_levels.shape} pxar: {pxar.shape}"
                )
            await asyncio.sleep(0.1)
    except queue.Empty:
        print("Data queue is empty.")


if __name__ == '__main__':
    asyncio.run(main())
