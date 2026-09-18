# Model card

**Model:** StandardScaler + Ridge regression.  
**Task:** One-hour-ahead demand forecasting for a fictional small electricity grid.  
**Status:** Educational demonstration; created with OpenAI Codex assistance.

## Inputs and target

Demand (MW) and temperature (°C) are generated locally from smooth seasonal
patterns, weekday effects, heating demand and autocorrelated random noise.
Neither the generator nor its units are calibrated to an actual grid.
UTC timestamps avoid local daylight-saving ambiguity in this example.

Features include lagged load at 1, 2, 24, 48 and 168 hours, a shifted 24-hour
mean, temperature observed one hour earlier, heating proxy, cyclical hour terms,
weekend indicator and an elapsed-time trend. Target is demand at the next hour.

## Training and evaluation

Use chronological 60%/20%/20% splits after creating past-only features.
Select alpha by validation MAE; refit on training plus validation.
Compare the fixed fitted model with last-hour, daily and weekly naive forecasts
on identical final test timestamps. Exact boundaries and versions are saved in
`reports/metrics.json`. No hyperparameter is chosen using the final test score.

## Appropriate interpretation

The benchmark exercises a forecasting workflow and its temporal assumptions.
It cannot establish effectiveness on Denmark's electricity system, power prices,
maritime data, customer demand or any other real operational process. No
classification accuracy, prediction interval, trading return or commercial
deployment is claimed. The model's relationship to a known synthetic generator
makes these results substantially easier to obtain than real-world evidence.

## Main risks and limitations

- Only one generator, seed and holdout period are reported.
- Observation latency is idealised; real metering data may arrive too late.
- Unseen shocks, outages, demand response, holidays and changing behaviour
  are not represented realistically.
- This is one-step forecasting with revealed history; it is not a day-ahead test.
- No uncertainty calibration, drift monitoring or production service exists.
- A production extension needs independent real-data evaluation and monitoring.
