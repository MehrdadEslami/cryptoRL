from abc import ABCMeta, abstractmethod
from gym.spaces import Box, Space
from kafka import KafkaConsumer
import numpy as np
import time

consumer = KafkaConsumer(
    'imagesBucket',
    bootstrap_servers=['localhost:29092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

class Observer(metaclass=ABCMeta):
    """A component to generate an observation at each step of an episode."""

    registered_name = "observer"

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the `TradingEnv`. (`Space`, read-only)"""
        raise NotImplementedError()

    @abstractmethod
    def observe(self, env: 'TradingEnv') -> np.array:
        """Gets the observation at the current step of an episode"""
        raise NotImplementedError()

    def next(self):
        """Retrieve Next Image"""
        pass

    def reset(self, random_start=0):
        """Resets the observer."""
        pass


class ObserverM(Observer):

    def __init__(self, influx_config, symbol, buffer_size: int = 25, **kwargs) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.state = None
        # self.image_queue = Queue(maxsize=10)  # Adjust maxsize as needed
        self.step = 0
        self._observation_space = Box(
            low=0,
            high=1,
            shape=(self.buffer_size, self.buffer_size, 4),
            dtype=np.uint8
        )

        # Start the thread to fetch and process trades
        # self.fetch_thread = threading.Thread(target=self.fetch_and_process_trades)
        # self.fetch_thread.daemon = True
        # self.fetch_thread.start()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def observe(self, env) -> np.array:
        self.state = self.next()
        return self.state

    def next(self):
        # image = self.image_queue.get()  # This will block if the queue is empty
        image = consumer.get_message(block=False, timeout=self.IterTimeout, get_partition_info=True )
        return image

    def reset(self) -> None:
        self.trades = None
        self.start_query_time = 0
        self.query_trades(self.start_query_time, self.end_query_time)


    # def fetch_and_process_trades(self):
    #     while True:
    #         self.query_trades(self.start_query_time, "now()")
    #         num_trades = len(self.trades)
    #         for i in range(0, num_trades, self.buffer_size ** 2):
    #             end = i + self.buffer_size ** 2
    #             trades_slice = self.trades.iloc[i:end]
    #             if len(trades_slice) < self.buffer_size ** 2:
    #                 break
    #             print('trades slice :', len(trades_slice))
    #             image = self.trades_to_normal_image(trades_slice.reset_index(drop=True))
    #             self.image_queue.put(image)
    #         time.sleep(10)  # Adjust sleep time as needed to control fetch rate
