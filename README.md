# Hull-White Rate Engine

I built this project to understand how a Treasury yield curve actually gets turned into something a short-rate model can use. I started with the Hull-White equation, but most of the work ended up being about the curve underneath it and how sensitive the model becomes once you start taking derivatives.

The full model is still here. I did not want this to just be a short Monte Carlo script.

## What it does

The program pulls the latest U.S. Treasury CMT curve and treats those observations as par yields rather than zero rates. It then bootstraps an approximate zero curve, builds discount factors and instantaneous forward rates, fits the Hull-White term structure, and simulates rates.

The simulation uses the exact Ornstein-Uhlenbeck state transition for the stochastic part of the model. From there it can price zero-coupon bonds and turn the simulated short-rate paths into constant-maturity zero-yield paths.

The default run also checks the model against its analytic moments and reruns the curve under parallel +50 bp and -50 bp shocks.

## Sample outputs

### Yield-curve scenario comparison

![Yield-curve scenario comparison](images/scenario_comparison.svg)

### Monte Carlo validation

![Monte Carlo validation](images/monte_carlo_validation.svg)

### Drift regularization

![Hull-White drift regularization](images/theta_regularization.svg)

## Main math

The one-factor Hull-White process is

```text
dr(t) = [theta(t) - a*r(t)]dt + sigma*dW(t)
```

The curve side of the project is just as important as that equation. Treasury CMT yields are par yields, so the program first solves for discount factors and zero rates. The instantaneous forward rate comes from the log discount curve, and that forward curve is used to fit the time-dependent Hull-White drift.

I kept both the market curve and a lightly regularized model curve because differentiating a bumpy curve made the forward-rate derivative and theta unstable. The regularized curve is kept close to the bootstrapped discount factors instead of replacing them with something unrelated.

For simulation, the model separates the short rate into a deterministic curve-fitting part and a zero-mean OU state. The OU state is simulated with its exact transition rather than an Euler step.

## Other things in the project

- live Treasury CMT data
- CSV curve input for par or zero curves
- semiannual par-yield bootstrapping
- discount factors, zero yields, and instantaneous forwards
- curve regularization for stable derivatives
- zero-coupon bond pricing
- 2Y, 5Y, and 10Y constant-maturity zero-yield paths by default
- Monte Carlo vs analytic moment checks
- base and parallel-shock scenarios
- curve, simulation, yield-path, and scenario plots
- diagnostics for repricing and curve stability

## Run it

```bash
pip install -r requirements.txt
python hull_white_engine.py
```

The default assumptions include `a = 0.15` and `sigma = 1%`. Those are inputs, not parameters I calibrated from caps or swaptions.

## Tests

```bash
python -m unittest test_hull_white_engine.py
```

The tests cover the bootstrap, curve repricing, Hull-White fitting, exact-transition simulation moments, negative-rate handling, inverted curves, and scenario shocks.

## Files

```text
hull_white_engine.py        file to run
core.py                     curve inputs, settings, bootstrap, and shared helpers
engine.py                   Hull-White math, pricing, simulation, and diagnostics
reporting.py                scenarios, summaries, and plots
test_hull_white_engine.py   regression tests
images/                     sample graphs
```

## Limits

This is a learning and research project, not a production rates system. The Treasury par-to-zero conversion is approximate because Treasury does not publish the exact internal zero curve behind the CMT series. The model only has one stochastic factor, and `a` and `sigma` are user-supplied assumptions. The simulated distributions are model scenarios, not forecasts of where Treasury yields will actually go.
