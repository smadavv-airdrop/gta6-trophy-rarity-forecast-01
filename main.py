"""gta6-trophy-rarity-forecast-01: Earners/Players decomposition forecaster with refresh-aware spreads.

Models each trophy's earn rate as earners / players_tracked.
- Earners grow logistically (saturating) per trophy tier.
- Players_tracked jumps at Exophase refreshes; between refreshes it's flat.
- Forecast: project earners forward with logistic fit, project players_tracked with
  step-hold + expected refresh magnitude, combine for rate quantiles.
- Spread widens with horizon and refresh uncertainty.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Tier-specific growth parameters (calibrated from practice + leaderboard patterns)
TIER_PARAMS = {
    "platinum": {"k": 0.15, "carry": 95.0, "refresh_mult": 0.02},
    "gold": {"k": 0.25, "carry": 85.0, "refresh_mult": 0.03},
    "silver": {"k": 0.35, "carry": 60.0, "refresh_mult": 0.05},
    "bronze": {"k": 0.45, "carry": 40.0, "refresh_mult": 0.08},
}

# Base spread parameters (logit scale per sqrt(day))
BASE_SIGMA = 0.35
REFRESH_SIGMA = 0.15


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(l: float) -> float:
    return 100 / (1 + math.exp(-l))


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def detect_refreshes(history: list[dict]) -> list[int]:
    """Return indices where players_tracked jumps (refresh events)."""
    refreshes = []
    for i in range(1, len(history)):
        if history[i]["players_tracked"] > history[i - 1]["players_tracked"]:
            refreshes.append(i)
    return refreshes


def fit_logistic_earners(earners: list[float], times: list[float], carry: float, k_init: float) -> tuple[float, float, float]:
    """Fit logistic: E(t) = carry / (1 + exp(-k*(t - t0))). Return (carry, k, t0)."""
    if len(earners) < 3:
        return carry, k_init, times[0] if times else 0.0

    # Use logit transform for linear fit on middle portion
    valid = [(t, e) for t, e in zip(times, earners) if 0 < e < carry]
    if len(valid) < 3:
        return carry, k_init, times[0]

    t_vals, e_vals = zip(*valid)
    # logit(E/carry) = k*(t - t0)  =>  log(E/(carry-E)) = k*t - k*t0
    y = [math.log(e / (carry - e)) for e in e_vals]
    # Linear regression: y = a*t + b, where a=k, b=-k*t0
    n = len(t_vals)
    sum_t = sum(t_vals)
    sum_y = sum(y)
    sum_ty = sum(t * yi for t, yi in zip(t_vals, y))
    sum_t2 = sum(t * t for t in t_vals)
    denom = n * sum_t2 - sum_t * sum_t
    if denom == 0:
        return carry, k_init, times[0]
    k = (n * sum_ty - sum_t * sum_y) / denom
    k = max(0.01, min(k, 2.0))  # clamp
    t0 = (sum_y - k * sum_t) / (-k * n) if k != 0 else times[0]
    return carry, k, t0


def forecast_earners(times: list[float], earners: list[float], target_t: float, carry: float, k: float, t0: float) -> float:
    """Project earners at target_t using fitted logistic."""
    if k <= 0:
        return earners[-1]
    pred = carry / (1 + math.exp(-k * (target_t - t0)))
    return min(pred, carry * 0.99)  # never quite hit carry


def forecast_players(history: list[dict], target_dt: datetime, base_dt: datetime, horizon_days: float) -> tuple[float, float]:
    """Forecast players_tracked at target. Return (mean, std_logit_contribution)."""
    if not history:
        return 1000.0, 0.0

    players = [h["players_tracked"] for h in history]
    times = [(parse_iso(h["at"]) - base_dt).total_seconds() / 3600 for h in history]  # hours from start

    # Detect refreshes
    refreshes = detect_refreshes(history)
    if not refreshes:
        # No refreshes observed: assume flat with small drift
        return float(players[-1]), 0.0

    # Estimate refresh magnitude (multiplicative factor)
    factors = []
    for idx in refreshes:
        if players[idx - 1] > 0:
            factors.append(players[idx] / players[idx - 1])
    avg_factor = sum(factors) / len(factors) if factors else 1.0

    # Time since last refresh
    last_refresh_idx = refreshes[-1]
    last_refresh_t = times[last_refresh_idx]
    target_hours = (target_dt - base_dt).total_seconds() / 3600
    hours_since_refresh = target_hours - last_refresh_t

    # Expected refreshes in horizon (Exophase refreshes roughly every 6-12 hours)
    refresh_rate_per_hour = len(refreshes) / max(times[-1], 1)
    expected_refreshes = refresh_rate_per_hour * max(hours_since_refresh, 0)

    # Project: current * factor^expected_refreshes
    mean_players = players[-1] * (avg_factor ** max(expected_refreshes, 0))

    # Uncertainty: logit-scale spread from refresh count variance (Poisson)
    refresh_var = expected_refreshes  # Poisson variance = mean
    logit_std = REFRESH_SIGMA * math.sqrt(max(refresh_var, 0.1)) * math.sqrt(horizon_days)

    return mean_players, logit_std


def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"Unexpected levels: {case['levels']}")

    history = case["history"]
    if not history:
        raise ValueError("No history")

    trophies = case["trophies"]
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0

    base_dt = parse_iso(history[0]["at"])
    target_dt = parse_iso(case["target"])
    times = [(parse_iso(h["at"]) - base_dt).total_seconds() / 3600 for h in history]
    target_t = (target_dt - base_dt).total_seconds() / 3600

    # Players tracked over time
    mean_players_target, refresh_logit_std = forecast_players(history, target_dt, base_dt, horizon_days)

    # Per-trophy earners series
    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy["tier"]
        params = TIER_PARAMS.get(tier, TIER_PARAMS["bronze"])
        carry = params["carry"]
        k_init = params["k"]

        # Extract earners = rate * players / 100
        earners = []
        for h in history:
            rate = h["rates"][i]
            if rate is not None and h["players_tracked"] > 0:
                earners.append(rate * h["players_tracked"] / 100.0)
            else:
                earners.append(0.0)

        # Fit logistic to earners
        carry_f, k_f, t0_f = fit_logistic_earners(earners, times, carry, k_init)

        # Forecast earners at target
        earners_target = forecast_earners(times, earners, target_t, carry_f, k_f, t0_f)

        # Forecast rate = earners / players
        if mean_players_target > 0:
            rate_median = (earners_target / mean_players_target) * 100.0
        else:
            rate_median = earners[-1] * 100.0 / max(history[-1]["players_tracked"], 1)

        rate_median = clamp(rate_median)

        # Total logit-scale spread
        base_spread = BASE_SIGMA * math.sqrt(horizon_days)
        total_logit_std = math.sqrt(base_spread**2 + refresh_logit_std**2)

        # Quantiles on logit scale
        lg = logit(rate_median)
        qs = [round4(inv_logit(lg + z * total_logit_std)) for z in Z]

        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()