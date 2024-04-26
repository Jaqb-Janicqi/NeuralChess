import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __getitem__(self, index) -> tuple:
        row = self.dataframe.iloc[index]
        return row.iloc[0], row.iloc[1]

    def __len__(self) -> int:
        return len(self.dataframe)
