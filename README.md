[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/smadavv-airdrop/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/smadavv-airdrop/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01

Forecast how many GTA VI players earn each trophy in week one.

## Approach

Earners/Players decomposition with refresh-aware spreads.

- **Earners** grow logistically per trophy tier (platinum/gold/silver/bronze) with tier-specific carry capacities and growth rates.
- **Players tracked** jumps at Exophase refreshes; between refreshes it's flat. Model refresh magnitude from history and project Poisson-distributed refresh count forward.
- **Rate quantiles** combine earners/players on logit scale: base spread √(horizon_days) + refresh spread √(expected_refreshes).

## Run

```bash
python3 main.py
```

Input: `MAKEFUN_INPUT_DIR` with `*.in` files. Output: `MAKEFUN_OUTPUT_DIR` with `*.out` files.