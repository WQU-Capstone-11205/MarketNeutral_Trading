import numpy as np
import pandas as pd

def policy_stability_metrics(
        actions, 
        eps=1e-3, 
        small_rev_thresh=0.02
    ):
    """
    Compute policy stability metrics for actions

    Returns:
      details: list of dicts with per-stability metrics:
        {'reversal_rate_per_step', 'acf1'}
    """
    a = np.asarray(actions)
    N = len(a)

    # sign flips for dead-zone
    sign = np.sign(a)
    sign[np.abs(a) < eps] = 0
    # treat 0->+ or 0->- as changes only if next nonzero differs
    sign_changes = np.where(np.roll(sign, -1) * sign < 0)[0]  # indices where sign flips
    n_sign_flips = len(sign_changes)
    reversal_rate = n_sign_flips / N

    # autocorrelation lag-1
    x = a[:-1]
    y = a[1:]

    mask = ~np.isnan(x) & ~np.isnan(y)
    acf1 = np.corrcoef(x[mask], y[mask])[0, 1]

    return {
        'reversal_rate_per_step': reversal_rate,
        'acf1': acf1
    }

def compute_adaptation_lags(
    dates,
    perf,               # portfolio returns
    change_probs,       # BOCPD change probability (same length)
    cp_flags            # change point flags (same length)
):
    """
    Compute adaptation lags for each detected change point.

    Returns:
      details: list of dicts with per-CP diagnostics:
        {'cp_idx', 'next_cp_idx', 'baseline_var', 'stabilization_lag'}
    """
    min_sep=1           # minimum separation for clustering
    pre_window=1        # lookback window
    delta_rel=0.20      # tolerance for recovery (20%)
    hold_len=2          # number of consecutive steps for stabilization
    var_frac=0.6        # post-break variance <= var_frac * pre-break variance to be stable

    perf = np.asarray(perf)
    change_probs = np.asarray(change_probs)
    N = len(perf)
    if len(change_probs) != N:
        raise ValueError("perf and change_probs must have same length")
    flags = np.asarray(cp_flags).astype(int)
    if len(flags) != N:
        raise ValueError("cp_flags must have same length as perf")

    # derive CP indices and clusters
    raw_indices = np.where(flags == 1)[0]
    if raw_indices.size == 0:
        return np.array([]), []
    clustered = []
    last = -10**9
    for idx in raw_indices:
        if idx - last >= min_sep:
            clustered.append(int(idx))
            last = int(idx)
    cp_indices = np.array(clustered, dtype=int)

    results = []

    for i, cp_idx in enumerate(cp_indices):
        if i + 1 < len(cp_indices):
            next_cp = cp_indices[i + 1]
        else:
            next_cp = N

        # baseline window 
        start_pre = max(0, cp_idx - pre_window)
        baseline_window = perf[start_pre:cp_idx]
        if baseline_window.size == 0:
            baseline_mean = np.nan
            baseline_var = np.nan
        else:
            baseline_mean = np.nanmedian(baseline_window)
            baseline_var = np.nanvar(baseline_window)

        # target bounds for recovery
        left_bound = baseline_mean * (1 - delta_rel) if baseline_mean is not np.nan else -np.inf
        right_bound = baseline_mean * (1 + delta_rel) if baseline_mean is not np.nan else np.inf

        recovered_idx = None
        reason = None

        low_bound = min(left_bound, right_bound)
        high_bound = max(left_bound, right_bound)

        # Precompute rolling metrics
        post_slice = perf[cp_idx+1:next_cp]

        L = len(post_slice)
        # pad for less than hold_len
        for offset in range(0, L - hold_len + 1):
            j = cp_idx + 1 + offset
            window_vals = perf[j : j + hold_len]

            # rolling mean within bounds
            cond_mean = np.all((window_vals >= low_bound) & (window_vals <= high_bound))

            if baseline_var is np.nan or baseline_var == 0:
                cond_var = True
            else:
                post_var = np.nanvar(window_vals)
                cond_var = (post_var <= var_frac * baseline_var)

            if cond_mean:
                recovered_idx = j + (hold_len - 1)
                reason = {
                    'cond_mean': bool(cond_mean),
                    'cond_var': bool(cond_var),
                }
                break

        # no recovery for this CP
        if recovered_idx is None:
            results.append({
                'cp_idx': int(cp_idx),
                'next_cp_idx': int(next_cp),
                'baseline_var': float(baseline_var) if not np.isnan(baseline_var) else None,
                'stabilization_lag': np.nan,
            })
        else:
            lag = recovered_idx - cp_idx
            results.append({
                'cp_idx': int(cp_idx),
                'next_cp_idx': int(next_cp),
                'baseline_var': float(baseline_var) if not np.isnan(baseline_var) else None,
                'stabilization_lag': int(lag),
            })

    return results

def generate_stabilization_lag_report(results):
    """
    results = list of dicts from compute_adaptation_lags(), e.g.:
    {
        "cp_idx": int,
        "next_cp_idx": int,
        "baseline_var": float,
        "stabilization_lag": int
    }
    """

    df = pd.DataFrame(results)

    # Compute normalized stabilization ratio
    df["stabil_ratio"] = df["stabilization_lag"] / (df["next_cp_idx"] - df["cp_idx"]).replace(0, np.nan)

    # Summary
    summary = {
        "mean_stabilization_lag": df["stabilization_lag"].mean(),
        "p95_stabilization_lag": df["stabilization_lag"].quantile(0.95),
        "mean_stabil_ratio": df["stabil_ratio"].mean()
    }

    return df, summary
