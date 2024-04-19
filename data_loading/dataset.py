import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __getitem__(self, index) -> tuple:
        row = self.dataframe.iloc[index]
        feature = row.iloc[0]
        label = row.iloc[1]
        return feature, label

    def __len__(self) -> int:
        return len(self.dataframe)
