from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def one_sample_t_test(returns: pd.Series, mu0: float = 0.0) -> dict:
    """
    H0: mean return = mu0
    H1: mean return != mu0
    """
    clean_returns = returns.dropna().values

    if len(clean_returns) == 0:
        return {
            "n": 0,
            "mean_return": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
        }

    t_stat, p_value = stats.ttest_1samp(clean_returns, popmean=mu0)

    return {
        "n": len(clean_returns),
        "mean_return": float(np.mean(clean_returns)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def bootstrap_mean_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> dict:
    """
    Bootstrap confidence interval for mean return
    """
    clean_returns = returns.dropna().values

    if len(clean_returns) == 0:
        return {
            "n": 0,
            "mean_return": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    rng = np.random.default_rng(random_state)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(clean_returns, size=len(clean_returns), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "n": len(clean_returns),
        "mean_return": float(np.mean(clean_returns)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }