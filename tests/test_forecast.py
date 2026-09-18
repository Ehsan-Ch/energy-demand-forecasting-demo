"""Tests for temporal leakage, split integrity and reproducible input data."""
import unittest
import numpy as np
import pandas as pd
from forecast import generate_data, make_features, split_bounds


class TemporalIntegrityTests(unittest.TestCase):
    def test_future_and_current_targets_do_not_change_past_features(self):
        data = generate_data(days=60)
        original, _ = make_features(data)
        target_time = data.index[500]
        changed = data.copy()
        changed.loc[target_time:, ["demand_mw", "temperature_c"]] += 10000
        altered, _ = make_features(changed)
        pd.testing.assert_frame_equal(original.loc[:target_time], altered.loc[:target_time])

    def test_one_hour_history_has_expected_timing(self):
        data = generate_data(days=60)
        x, y = make_features(data)
        np.testing.assert_allclose(x.load_lag_1, data.demand_mw.shift(1).loc[x.index])
        pd.testing.assert_series_equal(y, data.demand_mw.loc[x.index])
        self.assertEqual(x.index[0], data.index[168])

    def test_split_is_chronological_and_disjoint(self):
        x, _ = make_features(generate_data(days=60))
        a, b = split_bounds(len(x))
        self.assertLess(x.index[a - 1], x.index[a])
        self.assertLess(x.index[b - 1], x.index[b])
        self.assertTrue(0 < a < b < len(x))

    def test_generator_is_reproducible(self):
        pd.testing.assert_frame_equal(generate_data(60, 42), generate_data(60, 42))
        self.assertFalse(generate_data(60, 42).equals(generate_data(60, 43)))

    def test_missing_hour_is_rejected(self):
        data = generate_data(60).drop(generate_data(60).index[100])
        with self.assertRaisesRegex(ValueError, "regular hourly"):
            make_features(data)

    def test_missing_observation_is_rejected(self):
        data = generate_data(60)
        data.iloc[100, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing observations"):
            make_features(data)


if __name__ == "__main__":
    unittest.main()
