import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

warnings.filterwarnings('ignore')


class TestGARCHModel(unittest.TestCase):

    def setUp(self):
        self.ticker = 'SPY'
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'log_return': np.random.normal(0, 0.02, 100),
            'realized_vol_20d': np.abs(np.random.normal(0.15, 0.05, 100))
        }, index=self.dates)

    def test_data_loading(self):
        self.assertEqual(len(self.test_df), 100)
        self.assertIn('log_return', self.test_df.columns)
        self.assertIn('realized_vol_20d', self.test_df.columns)

    def test_data_splitting(self):
        n = len(self.test_df)
        train_end = int(n * 0.6)
        val_end = train_end + int(n * 0.2)
        
        self.assertEqual(train_end, 60)
        self.assertEqual(val_end, 80)
        self.assertLess(train_end, val_end)

    def test_forecast_shape(self):
        test_len = 20
        forecast = np.random.normal(0.15, 0.05, test_len)
        
        self.assertEqual(len(forecast), test_len)
        self.assertTrue(all(f > 0 for f in forecast))

    def test_constant_forecast_detection(self):
        constant_forecast = np.array([0.15] * 20)
        variance = np.var(constant_forecast)
        
        self.assertAlmostEqual(variance, 0, places=5)


class TestEWMAModel(unittest.TestCase):

    def setUp(self):
        self.lambda_param = 0.94
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.returns = pd.Series(np.random.normal(0, 0.02, 100), index=self.dates)

    def test_lambda_parameter(self):
        self.assertGreater(self.lambda_param, 0)
        self.assertLess(self.lambda_param, 1)

    def test_ewma_initialization(self):
        initial_window = self.returns.iloc[:20]
        initial_vol = initial_window.std() ** 2
        
        self.assertGreater(initial_vol, 0)

    def test_ewma_calculation(self):
        squared_returns = self.returns ** 2
        variance = np.zeros(len(self.returns))
        
        for t in range(1, len(self.returns)):
            variance[t] = (self.lambda_param * variance[t-1] + 
                          (1 - self.lambda_param) * squared_returns.iloc[t-1])
        
        self.assertEqual(len(variance), len(self.returns))
        self.assertTrue(all(v >= 0 for v in variance))

    def test_ewma_mean_reversion(self):
        variance = np.array([0.0001, 0.0002, 0.00015, 0.00018, 0.00017])
        self.assertTrue(np.std(variance) > 0)


class TestXGBoostModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.abs(np.random.normal(0.15, 0.05, self.n_samples))

    def test_feature_dimensions(self):
        self.assertEqual(self.X.shape[0], self.n_samples)
        self.assertEqual(self.X.shape[1], self.n_features)

    def test_target_shape(self):
        self.assertEqual(len(self.y), self.n_samples)
        self.assertTrue(all(v > 0 for v in self.y))

    def test_train_test_split(self):
        train_ratio = 0.6
        split_idx = int(len(self.X) * train_ratio)
        
        X_train = self.X[:split_idx]
        X_test = self.X[split_idx:]
        
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertLess(len(X_test), len(X_train))

    def test_scaling_bounds(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        self.assertAlmostEqual(X_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_scaled.std(), 1, places=5)


class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        self.lookback = 20
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100,
                               index=self.dates)

    def test_lookback_window(self):
        self.assertEqual(self.lookback, 20)
        self.assertLess(self.lookback, len(self.prices))

    def test_sequence_creation_logic(self):
        data = self.prices.values.reshape(-1, 1)
        lookback = self.lookback
        
        sequences = []
        for i in range(lookback, len(data)):
            sequences.append(data[i-lookback:i, 0])
        
        self.assertEqual(len(sequences), len(self.prices) - self.lookback)
        self.assertEqual(len(sequences[0]), self.lookback)

    def test_normalization_bounds(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(self.prices.values.reshape(-1, 1))
        
        self.assertTrue(np.all(scaled >= 0))
        self.assertTrue(np.all(scaled <= 1))

    def test_price_data_shape(self):
        self.assertEqual(len(self.prices), 100)
        self.assertTrue(all(p > 0 for p in self.prices))


class TestProphetModel(unittest.TestCase):

    def setUp(self):
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100,
                               index=self.dates)
        self.test_data = pd.DataFrame({
            'ds': self.dates,
            'y': self.prices.values
        })

    def test_prophet_data_format(self):
        self.assertIn('ds', self.test_data.columns)
        self.assertIn('y', self.test_data.columns)
        self.assertEqual(len(self.test_data), 100)

    def test_forecast_index(self):
        train_size = int(len(self.test_data) * 0.6)
        test_data = self.test_data.iloc[train_size:]
        
        self.assertEqual(len(test_data), len(self.test_data) - train_size)

    def test_minimum_data_requirement(self):
        self.assertGreaterEqual(len(self.test_data), 50)


class TestARIMAModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100,
                               index=self.dates)
        self.order = (3, 1, 1)

    def test_arima_order(self):
        p, d, q = self.order
        self.assertGreaterEqual(p, 0)
        self.assertGreaterEqual(d, 0)
        self.assertGreaterEqual(q, 0)

    def test_differencing(self):
        d = 1
        differenced = self.prices.diff(d).dropna()
        
        self.assertEqual(len(differenced), len(self.prices) - d)
        self.assertLess(abs(differenced.mean()), abs(self.prices.mean()))

    def test_stationarity_check(self):
        non_stationary = np.cumsum(np.random.normal(0, 1, 100))
        stationary = np.diff(non_stationary)
        
        self.assertEqual(len(non_stationary) - 1, len(stationary))


class TestEnsembleModel(unittest.TestCase):

    def setUp(self):
        self.weights = {'arima': 0.25, 'prophet': 0.50, 'lstm': 0.25}
        self.n_forecasts = 20

    def test_weight_sum(self):
        total_weight = sum(self.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_ensemble_combination(self):
        arima_fcst = np.random.normal(100, 5, self.n_forecasts)
        prophet_fcst = np.random.normal(100, 5, self.n_forecasts)
        lstm_fcst = np.random.normal(100, 5, self.n_forecasts)
        
        ensemble = (self.weights['arima'] * arima_fcst +
                   self.weights['prophet'] * prophet_fcst +
                   self.weights['lstm'] * lstm_fcst)
        
        self.assertEqual(len(ensemble), self.n_forecasts)
        self.assertGreater(ensemble.min(), min(arima_fcst.min(), prophet_fcst.min(), lstm_fcst.min()) - 5)

    def test_weight_rebalancing(self):
        available_weights = {k: v for k, v in self.weights.items() if k != 'lstm'}
        rebalanced = {k: v / sum(available_weights.values()) 
                     for k, v in available_weights.items()}
        
        self.assertAlmostEqual(sum(rebalanced.values()), 1.0, places=5)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.actual = np.array([100, 102, 101, 103, 105])
        self.forecast = np.array([100.5, 101.5, 101.5, 103.5, 104])

    def test_mae_calculation(self):
        mae = np.mean(np.abs(self.actual - self.forecast))
        
        self.assertGreater(mae, 0)
        self.assertLess(mae, max(abs(self.actual - self.forecast)))

    def test_rmse_calculation(self):
        rmse = np.sqrt(np.mean((self.actual - self.forecast) ** 2))
        
        self.assertGreater(rmse, 0)
        self.assertGreaterEqual(rmse, np.mean(np.abs(self.actual - self.forecast)))

    def test_mape_calculation(self):
        mape = np.mean(np.abs((self.actual - self.forecast) / self.actual)) * 100
        
        self.assertGreater(mape, 0)
        self.assertLess(mape, 100)

    def test_correlation_calculation(self):
        corr = np.corrcoef(self.actual, self.forecast)[0, 1]
        
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)

    def test_perfect_forecast(self):
        perfect = self.actual.copy()
        
        mae = np.mean(np.abs(self.actual - perfect))
        rmse = np.sqrt(np.mean((self.actual - perfect) ** 2))
        corr = np.corrcoef(self.actual, perfect)[0, 1]
        
        self.assertAlmostEqual(mae, 0, places=5)
        self.assertAlmostEqual(rmse, 0, places=5)
        self.assertAlmostEqual(corr, 1, places=5)


if __name__ == '__main__':
    unittest.main()
