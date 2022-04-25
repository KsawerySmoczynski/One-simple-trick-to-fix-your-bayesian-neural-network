from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    def __init__(
        self,
        # train:bool,
        horizon: int = 14,
        sales_train_evaluation: str = "datasets/m5/sales_train_evaluation.csv",
        # sales_train_validation:str = "datasets/m5/sales_train_validation.csv",
        sell_prices: str = "datasets/m5/sell_prices.csv",
        calendar: str = "datasets/m5/calendar.csv",
    ) -> None:
        super().__init__()
        # self.train = train
        self.horizon = horizon
        try:
            self.sales_train_evaluation = pd.read_csv(sales_train_evaluation).iloc[:10, :100]
            # self.sales_train_validation = pd.read_csv(sales_train_validation)
            self.sell_prices = pd.read_csv(sell_prices)
            self.calendar = pd.read_csv(calendar)
        except FileNotFoundError as e:
            import sys

            sys.exit(e)

        self.sales_train_evaluation["id"] = self.sales_train_evaluation["id"].apply(
            lambda x: x.replace("_evaluation", "")
        )
        ste_data_columns = list(filter(lambda x: "d_" in x, self.sales_train_evaluation.columns))
        self.sales_train_evaluation = self.sales_train_evaluation.set_index("id")[ste_data_columns]

        # self.sales_train_validation["id"] = self.sales_train_validation["id"].apply(lambda x: x.replace("_validation", ""))
        # ste_data_columns = list(filter(lambda x: "d_" in x, self.sales_train_validation.columns))
        # self.sales_train_validation = self.sales_train_validation.set_index("id")[[ste_data_columns]]

        self.sell_prices["id"] = self.sell_prices["item_id"] + "_" + self.sell_prices["store_id"]

        # CALENDAR_TODO

        self.n_items = self.sales_train_evaluation.shape[0]
        self.n_days = self.sales_train_evaluation.shape[1] - 2 * horizon

    def __len__(self):
        return self.n_items * self.n_days

    def __getitem__(self, idx):
        day = idx % self.n_days
        item = (idx - idx % self.n_days) // self.n_days
        X = self.sales_train_evaluation.values[item, day : day + self.horizon]
        y = self.sales_train_evaluation.values[item, day + self.horizon : day + 2 * self.horizon]
        return X, y

    def __next__(self):
        idx = 0
        while idx < self.len:
            yield self[idx]
            idx += 1


if __name__ == "__main__":
    dataset = M5Dataset()
    dl = iter(dataset)
    [print(item) for item in dl]
