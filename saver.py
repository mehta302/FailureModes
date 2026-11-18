import pandas as pd
import numpy as np


def rsave_data(val: np.array, model: str):
    df = pd.read_excel(
        "C:\\Users\\Yash Mehta\\Downloads\\Splitting data_PI bars (1).xlsx"
    )
    df[model] = val.tolist()
    df.to_excel(
        "C:\\Users\\Yash Mehta\\Downloads\\Splitting data_PI bars (1).xlsx"
    )
