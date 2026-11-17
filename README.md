# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A log-space profile forecaster for GTA VI's launch-day Twitch viewers, hour by hour. It reads one case (everything recorded up to the end of the release day's first hour) and writes five quantiles (5%, 25%, 50%, 75%, 95%) of the concurrent-viewer count for each of hours 2–24 (UTC). Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

## Why this beats the baseline

The house baseline says every hour stays at hour 1's count. Launch days don't do that: hour 1 (00:00–01:00 UTC) only has the time zones that already unlocked, so the day's curve rises or falls by a factor that depends strongly on how big hour 1 already is. Across 37 past big launches, log(viewers_h / viewers_1) correlates about −0.93 with log(viewers_1) at hours 19 and 24. This program learns that relationship from the case's own embedded comparables and forecasts each hour from an ensemble of three estimators of the release-day log-ratio profile (quadratic regression in log hour-1 level with level-dependent spread, the same mean with empirical residual quantiles, and the 8 nearest comparables by hour-1 level).

Leave-one-out pinball loss (log scale) over the 37 comparables in the kit: **0.378** for this model vs **0.930** for the house baseline (median 0.264 vs 0.325; wins 29/37). On the 23 launches whose hour-1 level is in GTA VI's plausible range it scores 0.256 vs the baseline's 0.325. See [REPORT.md](REPORT.md) for the full method and per-launch numbers.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic, ~40 ms per case. Reads every `<id>.in` in the input directory and writes `<id>.out` (`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}`) to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?ref=andikha-ramadhan&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
