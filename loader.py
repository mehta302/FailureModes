import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats


def load_stat():
    np.random.seed(42)
    df = pd.read_excel(
        "./Splitting data_PI bars Original Data.xlsx"
    )
    return np.array(df["Peak load-FE (kN) or Exp(kN)"].dropna()), np.array(df["Spring Model"].dropna())


def rload_data1(sheet_name="Bond strength-final-25 August "):
    np.random.seed(42)
    df = pd.read_excel(
        "./Splitting_data_PI_bars.xlsx",
        # sheet_name=sheet_name
    )
    # df = df[df["Cmin"] < df["Cmax"]]
    df = df[:91]

    X = df[[
        "Dia (mm)",
        "hef (mm)",
        "f'c (MPa)",
        "Taub-Mortar (MPa)",
        "Cmin",
        "Cmax"
    ]]

    # print(X)

    Y = df[[
        "Peak load-FE (kN) or Exp(kN)",
        "FEM"
    ]]

    print(Y)

    # X = X[:13891]
    # Y = Y[:13891]
    # X = X[:93]
    # Y = Y[:93]

    return X, Y


def rload_data(sheet_name="Bond strength-final-25 August"):
    np.random.seed(42)
    df = pd.read_excel(
        "./Splitting_data_PI_bars.xlsx",
        # sheet_name=sheet_name
    )
    df = df[df["Cmin"] <= df["Cmax"]]

    X = df[[
        # "Dia (mm)",
        "hef (mm)",
        "f'c (MPa)",
        # "Taub-Mortar (MPa)",
        "Cmin",
        "Cmax"
    ]]

    Y = df[[
        "Peak load-FE (kN) or Exp(kN)",
        "Taubm,PI-FE (MPa)"
    ]]

    df = df[[
        "hef (mm)",
        "f'c (MPa)",
        "Cmin",
        "Cmax",
        "Peak load-FE (kN) or Exp(kN)"
    ]]

    # df.columns = [
    #     "hef (mm)",
    #     "f'c (MPa)",
    #     "Cmin (mm)",
    #     "Cmax (mm)",
    #     "Peak load (kN)"
    # ]
    #
    # plt.matshow(df.corr())
    # plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, rotation=45)
    # plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns)
    # cb = plt.colorbar()
    # cb.ax.tick_params()
    # plt.title('Correlation Matrix')
    # plt.show()

    return X, Y


def load_split(sheet_name="Database") -> tuple[np.array, np.array, np.array, np.array]:
    np.random.seed(42)
    df = pd.read_excel(
        "./Data_for_chemofast.xlsx",
        sheet_name=sheet_name
    )

    X = df[[
        "d",
        "hef",
        "tau",
        "fc"
    ]]

    Y1 = df["alpha"]
    Y2 = df["alpha1"]

    X = X[1:]
    Y1 = Y1[1:].astype(float)
    Y2 = Y2[1:].astype(float)

    X["d"] = X["d"].astype("category")
    X["hef"] = X["hef"].astype("category")
    X["tau"] = X["tau"].astype(float)
    X["fc"] = X["fc"].astype("category")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y1, test_size=0.2, random_state=42
    )

    Y2_train = train_test_split(
        X, Y2, test_size=0.2, random_state=42
    )[2]

    nrows = Y_train.shape[0]
    replace_idx = np.random.choice(nrows, size=nrows // 2, replace=False)

    Y_train.iloc[replace_idx] = Y2_train.iloc[replace_idx]

    return X_train, X_test, Y_train, Y_test


def load_data(sheet_name="Database") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        "./Data_for_chemofast.xlsx",
        # sheet_name="Evaluation_New"
        sheet_name=sheet_name
    )

    # Diameter, Embedment, Avg Tau, Concrete Str
    X = df[[
        "d",
        "hef",
        "tau",
        "fc"
    ]]
    # N_u Unconfined
    # Y = df["Unconfined load"]
    Y = df["alpha"]
    # fdf = pd.concat([X[2:269], Y[2:269]], axis=1)
    # print(fdf.corr())

    X = X[1:]
    Y = Y[1:]

    return X, Y


def load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(
        "./Data_for_chemofast.xlsx",
        sheet_name="Evaluation_New"
        # sheet_name="Database"
    )

    # Diameter, Embedment, Avg Tau, Concrete Str
    X = df[[
        "Unnamed: 1",
        "Unnamed: 3",
        "Unnamed: 9",
        "Unnamed: 14"
    ]]
    X.columns = [
        # "Anchor diameter",
        # "Embedment depth",
        "d",
        "hef",
        "tau",
        "fc"
    ]
    # N_u Unconfined
    Y = df["Unconfined load"]
    # Y = df["alpha"]

    # fdf = pd.concat([X[2:269], Y[2:269]], axis=1)
    # print(fdf.corr())

    X = X[2:269]
    Y = Y[2:269]

    return X, Y


def plot4d(X: pd.DataFrame) -> None:
    nX = X.to_numpy()
    x = nX[:, 3].astype(float)
    y = nX[:, 2].astype(float)
    z = nX[:, 1].astype(float)
    c = nX[:, 0].astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()


def plot5d(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    fdf = pd.concat([X, Y], axis=1)

    groups = fdf.groupby("Anchor diameter")
    M = {8: '*', 12: 'x', 16: 'o'}
    C = {
        48: (0, 0, 0),
        64: (0.11, 0, 0),
        80: (0.22, 0, 0),
        96: (0.33, 0, 0),
        72: (0.44, 0, 0),
        120: (0.55, 0, 0),
        160: (0.66, 0, 0),
        144: (0.77, 0, 0),
        128: (0.88, 0, 0),
        110: (0.99, 0, 0),
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for name, group in groups:
        for x in group.values:
            ax.plot(x[3], x[2], x[4], c=C[x[1]], linestyle='', marker=M[x[0]])
    plt.show()


def plotEach(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=2)
    nX = X[:92].to_numpy()

    print(f"Embedment Depth: (mm) {np.max(nX[:, 0]) - np.min(nX[:, 0])} {np.mean(nX[:, 0])} {np.std(nX[:, 0])} {np.median(nX[:, 0])} {stats.mode(nX[:, 0])} {stats.skew(nX[:, 0])} {stats.kurtosis(nX[:, 0])}")
    print(f"Concrete Strength: (MPa) {np.max(nX[:, 1]) - np.min(nX[:, 1])} {np.mean(nX[:, 1])} {np.std(nX[:, 1])} {np.median(nX[:, 1])} {stats.mode(nX[:, 1])} {stats.skew(nX[:, 1])} {stats.kurtosis(nX[:, 1])}")
    print(f"Cmin: (mm) {np.max(nX[:, 2]) - np.min(nX[:, 2])} {np.mean(nX[:, 2])} {np.std(nX[:, 2])} {np.median(nX[:, 2])} {stats.mode(nX[:, 2])} {stats.skew(nX[:, 2])} {stats.kurtosis(nX[:, 2])}")
    print(f"Cmax: (mm) {np.max(nX[:, 3]) - np.min(nX[:, 3])} {np.mean(nX[:, 3])} {np.std(nX[:, 3])} {np.median(nX[:, 3])} {stats.mode(nX[:, 3])} {stats.skew(nX[:, 3])} {stats.kurtosis(nX[:, 3])}")
    # nX = np.log(X.to_numpy())
    x = nX[:, 3].astype(float)
    y = nX[:, 2].astype(float)
    z = nX[:, 1].astype(float)
    c = nX[:, 0].astype(float)
    i = Y[:92].to_numpy()[:, 0].astype(float)
    print(f"Peak Load: (kN) {np.max(i) - np.min(i)} {np.mean(i)} {np.std(i)} {np.median(i)} {stats.mode(i)} {stats.skew(i)} {stats.kurtosis(i)}")
    # i = np.log(Y.to_numpy()[:, 0].astype(float))

    ax[0][0].scatter(x, i)
    ax[0][1].scatter(y, i)
    ax[1][0].scatter(z, i)
    ax[1][1].scatter(c, i)
    # "Dia (mm)",
    # "hef (mm)",
    # "f'c (MPa)",
    # "Taub-Mortar (MPa)",
    # "Cmin",
    # "Cmax"
    ax[0][0].title.set_text("Cmax")
    ax[0][1].title.set_text("Cmin")
    ax[1][0].title.set_text("f'c")
    ax[1][1].title.set_text("hef")
    plt.show()


if __name__ == "__main__":
    X, Y = rload_data()

    plotEach(X, Y)
