# Hull-White Rate Engine

This is a small Python project I built to understand how a Treasury curve gets turned into something a short-rate model can actually use.

The script pulls U.S. Treasury CMT yields, bootstraps an approximate zero curve, builds discount factors and forward rates, and then uses a one-factor Hull-White model to simulate short rates.

## Why I built it

I originally wanted to understand Hull-White beyond just the equation. The harder part turned out to be the curve underneath the model. Treasury CMT yields are par yields, so I had to convert them into discount factors and zero rates before using them in the model. I also ran into instability when differentiating the forward curve, which is why the script uses a light smoothing spline before calculating the drift.

## What it does

- loads the latest Treasury CMT curve
- bootstraps an approximate zero curve
- calculates discount factors and instantaneous forward rates
- builds the Hull-White drift term
- simulates short-rate paths
- compares base, +50 bp, and -50 bp curve scenarios
- plots the curve and simulated paths

## Model

I use the one-factor Hull-White process

```text
dr(t) = [theta(t) - a*r(t)]dt + sigma*dW(t)
```

where `a` controls mean reversion and `sigma` controls short-rate volatility.

The default values are assumptions. I am not calibrating them to caps or swaptions.

## Run it

```bash
pip install -r requirements.txt
python hull_white_engine.py
```

## Tests

```bash
python -m unittest test_hull_white_engine.py
```

The tests are basic sanity checks for the bootstrap, a flat curve, the Hull-White drift, simulation moments, and a parallel curve shock.

## Files

```text
hull_white_engine.py        curve construction, model, simulation, and plots
test_hull_white_engine.py   small regression test suite
requirements.txt
```

## Limits

This is a learning/research project, not a production pricing system. The Treasury par-to-zero conversion is approximate, the model has only one factor, and the Hull-White parameters are user supplied rather than market calibrated.
