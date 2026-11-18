import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_excel("C:\\Users\\Yash Mehta\\Downloads\\Splitting data_PI bars (1).xlsx")
    x = np.arange(0, 91)
    df["ML Linear Regression"] = x.tolist()

    df.to_excel("C:\\Users\\Yash Mehta\\Downloads\\Splitting data_PI bars (1).xlsx")
