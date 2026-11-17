# IMC Prosperity 4: team IMCooked

Our algorithms for the first two rounds of [IMC Prosperity 4](https://prosperity.imc.com/), IMC's algorithmic trading competition, April 2026. Team of four.

**Result:** 1st worldwide in Round 1 manual trading, and top 15% of 18,803 teams overall.

Everything here is simulated. Prosperity is a simulated exchange with an in-game currency, and every profit figure below is a backtest in that currency.

The short case study, including the later rounds, is at [alexlahdiri.com/#imc](https://alexlahdiri.com/#imc).

## What is in this repository

Rounds 1 and 2 traded two products with a position limit of 80 each. The option and Monte Carlo work from the later rounds is not uploaded here.

| Product | What it does | How we traded it |
|---|---|---|
| `ASH_COATED_OSMIUM` | Mean-reverts around a fixed value of 10,000 | Market making around 10,000, with quotes skewed by inventory |
| `INTARIAN_PEPPER_ROOT` | Drifts up by 1 for every 1,000 timestamps | Buy and hold the drift, with a regime check as a safety net |

| File | What it is |
|---|---|
| `trader.py` | Round 1. Osmium market maker quoting 10,000 ± 5; Pepper Root priced off its fitted trend line |
| `trader4.py` | Round 1 iteration with tighter Osmium quotes and a maximum-long Pepper Root position; the comments mark which ideas came from a teammate |
| `trademlpalgo.py`, `advalgo.py`, `trader+MLP` | Iterations that add a teammate's neural network price model (18 order-book features in; `MLP_WB` holds a set of its weights) |
| `trader_r2_final.py` | **Round 2 final.** The version to read first: its parameters are explained in comments beside them |
| `MohilTradingAlgo/` | A teammate's Round 1 submission with the platform's result log |
| `prosperity4bt/` | The backtester we used, by kevin-fu1 (MIT licence, see [`LICENSE`](LICENSE)), which builds on [jmerle's Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester). Its own guide is [`BACKTESTER.md`](BACKTESTER.md) |

I wrote the trader files. Teammates contributed the neural network and the ideas credited in the code comments.

## The Round 2 algorithm

**Osmium.** Passive market making at a fair value of 10,000. Quotes sit 5 either side, inside the roughly 16-wide spread the exchange's own bots leave. 80% of the size goes on that quote and 20% two ticks behind it. Quotes shift against the current position, more than proportionally as it grows, and the position is flattened before the session ends.

**Pepper Root.** The price rises by 1 per 1,000 timestamps on all three days of sample data, with residuals of about 2.5. The algorithm buys to the position limit and holds. Exponential averages of the residual and of momentum classify the market as bullish, hold or bearish; both have to turn before it unwinds, at no more than 10 units per tick.

**Market access fee.** Round 2 sold extra order flow by sealed bid: the top half of bids won 25% more flow and paid what they bid. We bid 8,000 against an estimated value of 20,000 to 25,000.

## Two decisions you can see in the code

**Paying one tick more to get filled.** An earlier version bought Pepper Root up to 8 above fair value. The exchange's asks moved from 6.5 to 7.5 above mid across the three sample days, so that rule filled 97.5%, then 81.9%, then 71.6% of the time. Raising the limit to 9 filled 99.3%, 97.3% and 94.9%. One extra tick per purchase is small against a price that rises about 1,000 a day.

**Not trusting the neural network.** We suspected the price model had been trained with data leakage. It stayed in the final algorithm on a short lead: it can move the fair value by at most 3, only in a bullish regime, and only when its confidence is above 0.4.

## Run it

```bash
pip install -r requirements.txt

# macOS or Linux
PYTHONPATH=prosperity4bt python -m prosperity4bt trader_r2_final.py 1 --no-vis

# Windows PowerShell
$env:PYTHONPATH="prosperity4bt"; python -m prosperity4bt trader_r2_final.py 1 --no-vis
```

On the three days of Round 1 data in this repository, that prints a profit of 259,762 for `trader_r2_final.py` and 51,337 for `trader.py`.

Read those as a comparison between versions, not as results. The backtester fills orders against a recorded order book with no other teams competing for the same trades, so it overstates what a live round pays.
