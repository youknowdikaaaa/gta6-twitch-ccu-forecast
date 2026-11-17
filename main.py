"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01) -- log-space profile forecaster.

Method (see REPORT.md for the full write-up and validation numbers):

Twitch launch days do not stay flat: hour 1 (00:00-01:00 UTC) only has the
time zones that already unlocked, so the day's curve rises or falls by a
factor that depends strongly on how big hour 1 already is. Across 37 past
big launches, log(viewers_h / viewers_1) correlates about -0.93 with
log(viewers_1) at hours 19 and 24. The forecaster therefore learns, from
the case's own embedded comparables, how the release-day log-ratio profile
depends on the hour-1 level, and predicts each hour's five quantiles from
an ensemble of three estimators of that profile:

  1. quad-hetero (40%): per-hour quadratic regression of the log-ratio on
     log(hour-1 viewers), with a residual spread that depends on the
     hour-1 level (low hour-1 launches are far more dispersed), normal
     quantiles;
  2. quad-emp (30%): same quadratic mean, empirical quantiles of the
     per-hour residuals;
  3. knn8 (30%): empirical per-hour quantiles of the 8 comparables whose
     hour-1 level is nearest the target's.

The three quantile sets are averaged in log space. Leave-one-out pinball
loss (log scale) over the 37 comparables in the kit: 0.378 for this model
vs 0.930 for the house persistence baseline (median 0.264 vs 0.325). On
the 23 launches whose hour-1 level is in GTA VI's plausible range
(log viewers 10-13) it scores 0.256 vs the baseline's 0.325.

Fallbacks: with fewer than 8 usable comparables the program falls back
to the house persistence baseline (every hour = hour 1, spread widening
as a random walk with sigma 0.15 in log space). Every output is
sanitised: 23 rows, hours 2..24, five finite quantiles each, > 0,
<= 1e9, non-decreasing. Deterministic: no randomness, no network, no
clock, no dict-order dependence (all inputs sorted). Standard library
only; runs in milliseconds.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Baseline spread (house starter), log units per sqrt hour.
BASE_SIGMA = 0.15
# Ensemble weights: quad-hetero, quad-emp, knn8.
W = (0.4, 0.3, 0.3)
KNN_K = 8
MIN_POOL = 8


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def inv_norm(p):
    """Inverse standard normal CDF (Acklam's approximation), for levels
    other than the standard five. Accurate to ~1e-9."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def solve_linear(mat, vec):
    """Solve a small dense linear system by Gauss-Jordan with partial
    pivoting. Returns None if (numerically) singular."""
    m = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for i in range(m):
        piv = max(range(i, m), key=lambda r: abs(aug[r][i]))
        if abs(aug[piv][i]) < 1e-12:
            return None
        aug[i], aug[piv] = aug[piv], aug[i]
        pv = aug[i][i]
        aug[i] = [v / pv for v in aug[i]]
        for r in range(m):
            if r != i and aug[r][i] != 0.0:
                f = aug[r][i]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[i])]
    return [aug[i][m] for i in range(m)]


def ols_multi(xs, ys):
    """Ordinary least squares with an intercept. xs: list of feature
    rows (without intercept). Returns (beta, residual_sd) or None."""
    n = len(xs)
    if n == 0:
        return None
    k = len(xs[0])
    a = [[1.0] + row for row in xs]
    m = k + 1
    if n <= m:
        return None
    ata = [[sum(a[r][i] * a[r][j] for r in range(n)) for j in range(m)]
           for i in range(m)]
    aty = [sum(a[r][i] * ys[r] for r in range(n)) for i in range(m)]
    beta = solve_linear(ata, aty)
    if beta is None:
        return None
    resid = [ys[r] - sum(beta[j] * a[r][j] for j in range(m))
             for r in range(n)]
    dof = max(1, n - m)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return beta, sd


def hourly_viewers(hourly, day_start):
    """Collapse a comparable's sparse hourly points into
    {hour: mean viewers} relative to day_start (hour 1 = [00:00, 01:00))."""
    by_hour = {}
    for p in hourly or []:
        try:
            t = parse_ts(p["from"])
            v = float(p["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        by_hour.setdefault(h, []).append(v)
    return {h: sum(vs) / len(vs) for h, vs in by_hour.items()}


def build_pool(case):
    """Usable comparables: (x = log hour-1 viewers, {hour: log-ratio}).

    A comparable counts if it has an hour-1 print and at least 10 of the
    23 scored hours. Hours with no point are simply absent from that
    launch's profile (each hour's fit uses the launches that have it)."""
    pool = []
    for comp in case.get("comparables") or []:
        try:
            ds = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hm = hourly_viewers(comp.get("hourly"), ds)
        if 1 not in hm:
            continue
        v1 = max(1.0, hm[1])
        scored = [h for h in hm if 2 <= h <= 24]
        if len(scored) < 10:
            continue
        prof = {}
        for h in scored:
            prof[h] = math.log(max(1.0, hm[h]) / v1)
        pool.append((math.log(v1), prof))
    pool.sort(key=lambda item: item[0])
    return pool


def baseline_forecast(p1, levels, zvals):
    """House persistence baseline: every hour = hour 1, spread widening
    as a random walk in log space."""
    out = {}
    for h in range(2, 25):
        out[h] = [max(1.0, p1) * math.exp(z * BASE_SIGMA * math.sqrt(h - 1))
                  for z in zvals]
    return out


def quad_fit(pool, hour):
    """Quadratic regression of the log-ratio at `hour` on log hour-1
    viewers, over the pool launches that have that hour.
    Returns (beta, residual_sd, residuals, xs) or None."""
    xs = []
    ys = []
    for x, prof in pool:
        if hour in prof:
            xs.append([x, x * x])
            ys.append(prof[hour])
    if len(xs) < 6:
        return None
    fit = ols_multi(xs, ys)
    if fit is None:
        return None
    beta, sd = fit
    resid = []
    for row, y in zip(xs, ys):
        mu = beta[0] + beta[1] * row[0] + beta[2] * row[1]
        resid.append((row[0], y - mu))
    return beta, sd, resid


def comp_quad_hetero(pool, xt, levels, zvals):
    """Quadratic mean + level-dependent residual spread, normal quantiles."""
    out = {}
    xs_sorted = sorted(x for x, _ in pool)
    n = len(xs_sorted)
    t1 = xs_sorted[n // 3]
    t2 = xs_sorted[2 * n // 3]
    for h in range(2, 25):
        fit = quad_fit(pool, h)
        if fit is None:
            continue
        beta, _sd, resid = fit
        mu = beta[0] + beta[1] * xt + beta[2] * xt * xt
        lo = [abs(r) for x, r in resid if x < t1]
        mid = [abs(r) for x, r in resid if t1 <= x < t2]
        hi = [abs(r) for x, r in resid if x >= t2]
        s_lo = (sum(lo) / len(lo) * 1.2533) if lo else _sd
        s_mid = (sum(mid) / len(mid) * 1.2533) if mid else _sd
        s_hi = (sum(hi) / len(hi) * 1.2533) if hi else _sd
        sdx = s_lo if xt < t1 else s_hi if xt >= t2 else s_mid
        out[h] = [math.exp(mu + z * sdx) for z in zvals]
    return out


def comp_quad_emp(pool, xt, levels, zvals):
    """Quadratic mean + empirical quantiles of the per-hour residuals."""
    out = {}
    for h in range(2, 25):
        fit = quad_fit(pool, h)
        if fit is None:
            continue
        beta, _sd, resid = fit
        mu = beta[0] + beta[1] * xt + beta[2] * xt * xt
        res_sorted = sorted(r for _, r in resid)
        out[h] = [math.exp(mu + quantile(res_sorted, tau)) for tau in levels]
    return out


def comp_knn(pool, xt, levels, k=KNN_K):
    """Empirical per-hour quantiles of the k comparables nearest in
    log hour-1 viewers."""
    near = sorted(pool, key=lambda item: abs(item[0] - xt))[:k]
    out = {}
    for h in range(2, 25):
        rs = sorted(prof[h] for _, prof in near if h in prof)
        if len(rs) < 3:
            continue
        out[h] = [math.exp(quantile(rs, tau)) for tau in levels]
    return out


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    try:
        levels = [float(t) for t in levels]
    except (TypeError, ValueError):
        levels = list(LEVELS)
    if not levels or any(not (0.0 < t < 1.0) for t in levels):
        levels = list(LEVELS)
    standard = len(levels) == 5 and all(
        abs(a - b) < 1e-9 for a, b in zip(levels, LEVELS))
    zvals = list(Z) if standard else [inv_norm(t) for t in levels]

    first = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            first = row
            break
    try:
        p1 = max(1.0, float(first["viewers"])) if first else 1.0
    except (KeyError, TypeError, ValueError):
        p1 = 1.0
    if not math.isfinite(p1):
        p1 = 1.0

    hours = [row["hour"] for row in case.get("forecast") or []]
    hours = [h for h in hours if isinstance(h, int) and 2 <= h <= 24]
    if not hours:
        hours = list(range(2, 25))

    pool = build_pool(case)
    if len(pool) < MIN_POOL:
        raw = baseline_forecast(p1, levels, zvals)
    elif standard:
        xs = [x for x, _ in pool]
        xt = min(max(math.log(p1), min(xs)), max(xs))
        parts = [
            comp_quad_hetero(pool, xt, levels, zvals),
            comp_quad_emp(pool, xt, levels, zvals),
            comp_knn(pool, xt, levels),
        ]
        base = baseline_forecast(p1, levels, zvals)
        raw = {}
        for h in range(2, 25):
            have = [part[h] for part in parts if h in part]
            if not have:
                raw[h] = base[h]
                continue
            # Log-average the available components, re-weighted to sum 1.
            wts = [W[i] for i, part in enumerate(parts) if h in part]
            tot = sum(wts)
            row = []
            for qi in range(len(levels)):
                acc = 0.0
                for w, part in zip(wts, have):
                    # Components return log-ratios (forecast / hour-1 print);
                    # the ensemble average is a ratio, scaled by p1 below.
                    acc += w * math.log(max(1e-9, part[qi]))
                row.append(math.exp(acc / tot) * p1)
            raw[h] = row
    else:
        # Non-standard levels: quadratic mean + heteroscedastic spread,
        # normal quantiles at the requested levels.
        xs = [x for x, _ in pool]
        xt = min(max(math.log(p1), min(xs)), max(xs))
        raw = comp_quad_hetero(pool, xt, levels, zvals)
        # Components return log-ratios; scale by the hour-1 print.
        raw = {h: [q * p1 for q in qs] for h, qs in raw.items()}
        base = baseline_forecast(p1, levels, zvals)
        for h in range(2, 25):
            if h not in raw:
                raw[h] = base[h]

    out_hours = []
    for h in hours:
        qs = raw.get(h)
        if qs is None:
            qs = baseline_forecast(p1, levels, zvals)[h]
        clean = []
        prev = 0.0
        for q in qs:
            try:
                v = float(q)
            except (TypeError, ValueError):
                v = p1
            if not math.isfinite(v) or v <= 0.0:
                v = p1
            v = min(v, 1e9)
            v = round2(v)
            if v <= 0.0:
                v = 0.01
            v = min(v, 1e9)
            if v < prev:
                v = prev
            clean.append(v)
            prev = v
        out_hours.append({"hour": h, "q": clean})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
