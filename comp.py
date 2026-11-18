from loader import load_stat
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def bland_altman(mean_vals, d):
    plt.figure()
    plt.scatter(mean_vals, d)
    plt.axhline(np.mean(d), linestyle='--', color='gray')
    plt.axhline(np.mean(d) + 1.96*np.std(d), linestyle=':', color='red')
    plt.axhline(np.mean(d) - 1.96*np.std(d), linestyle=':', color='red')
    plt.xlabel("Mean Load")
    plt.ylabel("Prediction Error (Synthetic - Empirical)")
    plt.title("Bland–Altman Plot")
    plt.show()


if __name__ == "__main__":
    y_emp, y_syn = load_stat()
    d = y_syn - y_emp

    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    print(wilcoxon_p)

    delta = 0.05 * np.mean(y_emp)
    t1 = stats.ttest_1samp(d, -delta, alternative="greater")
    t2 = stats.ttest_1samp(d,  delta, alternative="less")
    print(f"Lower bound p-value: {t1.pvalue}")
    print(f"Upper bound p-value: {t2.pvalue}")

    mean_vals = (y_syn + y_emp) / 2
    bland_altman(mean_vals, d)
