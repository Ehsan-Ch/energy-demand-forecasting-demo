# Model evaluation — synthetic demonstration

These are computed regression errors from the included experiment, not mock accuracy
percentages or evidence of performance on real electricity markets. Seed: 42.
The data cover 365 fictional days. All timestamps are UTC.

## Evaluation design

Predict demand one hour ahead. At the start of hour t, observations through t-1 are
assumed available. Earlier test observations are therefore used as lags for later
test predictions. This is sequential one-step evaluation, not a day-ahead or a
multi-step forecast. The fitted model stays fixed throughout the test period.

| Split | First target | Last target | Rows |
| --- | --- | --- | ---: |
| Train | 2024-01-08T00:00:00+00:00 | 2024-08-09T18:00:00+00:00 | 5155 |
| Validation | 2024-08-09T19:00:00+00:00 | 2024-10-20T08:00:00+00:00 | 1718 |
| Test | 2024-10-20T09:00:00+00:00 | 2024-12-30T23:00:00+00:00 | 1719 |

The first 168 observations supply lag history and are excluded from target rows.
Ridge alpha is chosen using validation MAE from (0.1, 1.0, 10.0, 100.0). Selected alpha:
**100.0**. Each candidate scaler is fitted only to the training
split. The final scaler and model are refitted on train plus validation, then
evaluated on the final test period. No random split, full-series scaling or
backfilling is used. The test period is not used to choose alpha.

## Measured test results

| Model | MAE (MW) | RMSE (MW) |
| --- | ---: | ---: |
| Ridge | 2.512 | 3.171 |
| Last hour | 3.367 | 4.220 |
| Daily naive | 7.933 | 9.868 |
| Weekly naive | 6.054 | 7.674 |

MAE is the mean absolute error. RMSE is the square root of mean squared error
and gives larger errors more weight. Both use the same target rows. A regression
forecast has no universal accuracy percentage, so none is reported.

![Measured errors](figures/errors.svg)

![Final test week](figures/forecast.svg)

## Limits

- The generator is intentionally simple and known to the author. Results measure
  performance on that generator, not generalisation to an independent real system.
- One seed and one chronological holdout are insufficient for a production claim.
- Demand and temperature are assumed to arrive without operational latency.
- No price forecast, trading return, prediction interval or deployment is claimed.
- Calendar effects omit local holidays and daylight-saving transitions.
- Real data would require availability-aware weather inputs, missing-data policies,
  multiple rolling-origin evaluations, drift checks and appropriate data licensing.

## Reproduce and inspect

Run `python forecast.py`, then `python -m unittest discover -s tests -v`.
`metrics.json` stores exact scores, split boundaries, environment versions and a
synthetic-data hash. `test_predictions.csv` contains every holdout prediction.
`synthetic_data_sample.csv` contains the first 48 invented observations.
