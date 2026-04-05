from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a price DataFrame, returns it with a 'signal' column added.
        signal:  1 = buy, -1 = sell, 0 = hold
        """
        pass