import pandas as pd
from torch.utils.data import Dataset
import torch


class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: callable, precision: torch.dtype = torch.float32):
        self.dataframe = dataframe
        self.transform = transform
        self.precision = precision

    def __getitem__(self, index) -> tuple:
        row = self.dataframe.iloc[index]
        feature = row.iloc[0]
        feature = self.transform(feature)
        label = row.iloc[1]
        feature = torch.tensor(feature, dtype=self.precision)
        label = torch.tensor(label, dtype=self.precision)
        return feature, label

    def __len__(self) -> int:
        return len(self.dataframe)
