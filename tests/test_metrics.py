"""Hand-computed metric fixtures."""

import numpy as np

from forecast.metrics import mase, rmsse, smape


def test_rmsse_perfect_forecast():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    assert rmsse(y_true, y_pred, y_train) == 0.0


def test_rmsse_known_value():
    y_train = np.array([10.0, 12.0, 14.0, 16.0])
    y_true = np.array([20.0])
    y_pred = np.array([18.0])
    # scale = mean((2)^2) = 4; mse = 4; rmsse = sqrt(4/4) = 1
    assert abs(rmsse(y_true, y_pred, y_train) - 1.0) < 1e-9


def test_mase_known_value():
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    y_true = np.array([9.0])
    y_pred = np.array([7.0])
    # scale = mean(|2|,|2|,|2|) = 2; mase = 2/2 = 1
    assert abs(mase(y_true, y_pred, y_train, seasonality=1) - 1.0) < 1e-9


def test_smape_symmetric():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    # 2*|10|/(100+110) * 100 = 9.5238...
    expected = 200.0 / 21.0
    assert abs(smape(y_true, y_pred) - expected) < 1e-9


def test_smape_zero_actual_and_pred():
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([0.0, 5.0])
    assert smape(y_true, y_pred) == 0.0
