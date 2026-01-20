# Advanced Topics in Time Series Forecasting

## Table of Contents
- [Overview](#overview)
- [Probabilistic Forecasting](#probabilistic-forecasting)
- [Hierarchical Forecasting](#hierarchical-forecasting)
- [Transfer Learning](#transfer-learning)
- [Online Learning](#online-learning)
- [Conformal Prediction](#conformal-prediction)
- [Causal Forecasting](#causal-forecasting)
- [AutoML for Time Series](#automl-for-time-series)

---

## Overview

Advanced techniques for complex forecasting scenarios.

---

## Probabilistic Forecasting

### Quantile Regression

```python
import torch
import torch.nn as nn

class QuantileRegressionNN(nn.Module):
    """
    Neural network for quantile regression
    """

    def __init__(self, input_dim, hidden_dim, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate head for each quantile
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in quantiles
        ])

    def forward(self, x):
        shared = self.shared(x)
        outputs = [head(shared) for head in self.heads]
        return torch.cat(outputs, dim=-1)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i]
            losses.append(
                torch.max(q * errors, (q - 1) * errors)
            )
        return torch.mean(torch.stack(losses))
```

### Deep Ensembles

```python
class DeepEnsembleForecaster:
    """
    Ensemble of neural networks for uncertainty estimation
    """

    def __init__(self, model_class, n_models=5, **model_kwargs):
        self.models = [
            model_class(**model_kwargs)
            for _ in range(n_models)
        ]

    def fit(self, X, y):
        for i, model in enumerate(self.models):

            # Bootstrap sampling
            idx = np.random.choice(len(X), len(X), replace=True)
            model.fit(X[idx], y[idx])

    def predict(self, X, return_std=True):
        predictions = np.array([m.predict(X) for m in self.models])

        mean = predictions.mean(axis=0)

        if return_std:
            std = predictions.std(axis=0)
            return mean, std
        return mean

    def prediction_intervals(self, X, alpha=0.05):
        predictions = np.array([m.predict(X) for m in self.models])

        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)

        return lower, upper
```

---

## Hierarchical Forecasting

### Reconciliation Methods

```python
import numpy as np
from scipy.linalg import inv

class HierarchicalReconciliation:
    """
    Hierarchical time series reconciliation
    """

    def __init__(self, S_matrix):
        """
        Args:
            S_matrix: Summing matrix defining hierarchy
                      Shape: (n_bottom, n_all)
        """
        self.S = S_matrix
        self.n_bottom = S_matrix.shape[0]
        self.n_all = S_matrix.shape[1]

    def bottom_up(self, base_forecasts):
        """
        Bottom-up: Only use bottom level forecasts
        """
        bottom_forecasts = base_forecasts[-self.n_bottom:]
        return self.S @ bottom_forecasts

    def top_down(self, base_forecasts, proportions):
        """
        Top-down: Disaggregate top level using proportions
        """
        top_forecast = base_forecasts[0]
        bottom_forecasts = top_forecast * proportions
        return self.S @ bottom_forecasts

    def ols_reconciliation(self, base_forecasts):
        """
        OLS reconciliation (Ordinary Least Squares)
        """
        G = inv(self.S.T @ self.S) @ self.S.T
        return self.S @ G @ base_forecasts

    def wls_reconciliation(self, base_forecasts, weights):
        """
        Weighted Least Squares reconciliation
        """
        W_inv = np.diag(1 / weights)
        G = inv(self.S.T @ W_inv @ self.S) @ self.S.T @ W_inv
        return self.S @ G @ base_forecasts

    def mint_reconciliation(self, base_forecasts, residual_covariance):
        """
        MinT (Minimum Trace) reconciliation
        Uses residual covariance for optimal reconciliation
        """
        W_inv = inv(residual_covariance)
        G = inv(self.S.T @ W_inv @ self.S) @ self.S.T @ W_inv
        return self.S @ G @ base_forecasts

# Example: Retail hierarchy
"""
        Total
       /    \
    RegionA  RegionB
    /    \      |
 Store1 Store2 Store3

S matrix:
         Total RegA RegB S1  S2  S3
Store1     1    1    0   1   0   0
Store2     1    1    0   0   1   0
Store3     1    0    1   0   0   1
"""
```

---

## Transfer Learning

### Pre-trained Models for Time Series

```python
class TransferLearningForecaster:
    """
    Transfer learning for time series
    """

    def __init__(self, pretrained_model, freeze_layers=True):
        self.model = pretrained_model

        if freeze_layers:

            # Freeze feature extractor layers
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

        # Replace prediction head
        self.model.prediction_head = nn.Linear(
            self.model.hidden_dim,
            1  # New target dimension
        )

    def fine_tune(self, X, y, epochs=10, lr=1e-4):
        """
        Fine-tune on target domain
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            predictions = self.model(X)
            loss = F.mse_loss(predictions, y)

            loss.backward()
            optimizer.step()

class DomainAdaptation:
    """
    Domain adaptation for time series across different sources
    """

    def __init__(self, source_model, adaptation_method='coral'):
        self.source_model = source_model
        self.method = adaptation_method

    def coral_loss(self, source_features, target_features):
        """
        CORAL: Correlation Alignment
        Aligns second-order statistics
        """
        d = source_features.shape[1]

        # Covariance matrices
        source_cov = torch.cov(source_features.T)
        target_cov = torch.cov(target_features.T)

        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d ** 2)

        return loss

    def adapt(self, source_data, target_data, epochs=50):
        """
        Adapt source model to target domain
        """
        for epoch in range(epochs):

            # Extract features
            source_features = self.source_model.extract_features(source_data)
            target_features = self.source_model.extract_features(target_data)

            # Minimize domain discrepancy
            adaptation_loss = self.coral_loss(source_features, target_features)

            # Backward and update
            adaptation_loss.backward()

            # ... optimizer step
```

---

## Online Learning

### Incremental Model Updates

```python
class OnlineForecaster:
    """
    Online learning for streaming time series
    """

    def __init__(self, base_model, update_frequency=24):
        self.model = base_model
        self.update_frequency = update_frequency
        self.buffer = []
        self.update_count = 0

    def predict(self, x):
        """Make prediction with current model"""
        return self.model.predict(x)

    def update(self, x, y_true):
        """
        Update model with new observation
        """
        self.buffer.append((x, y_true))
        self.update_count += 1

        if self.update_count >= self.update_frequency:
            self._retrain()
            self.buffer = []
            self.update_count = 0

    def _retrain(self):
        """Incremental retrain on buffer"""
        X = np.array([b[0] for b in self.buffer])
        y = np.array([b[1] for b in self.buffer])

        self.model.partial_fit(X, y)

class AdaptiveEnsemble:
    """
    Ensemble with adaptive weights based on recent performance
    """

    def __init__(self, models, window_size=50):
        self.models = models
        self.window_size = window_size
        self.weights = np.ones(len(models)) / len(models)
        self.error_history = {i: [] for i in range(len(models))}

    def predict(self, x):
        predictions = np.array([m.predict(x) for m in self.models])
        return np.average(predictions, weights=self.weights)

    def update(self, x, y_true):
        """Update weights based on recent errors"""
        for i, model in enumerate(self.models):
            pred = model.predict(x)
            error = np.abs(y_true - pred)
            self.error_history[i].append(error)

            # Keep only recent history
            if len(self.error_history[i]) > self.window_size:
                self.error_history[i].pop(0)

        # Update weights inversely proportional to error
        mean_errors = np.array([
            np.mean(self.error_history[i]) for i in range(len(self.models))
        ])

        # Softmax weighting
        inv_errors = 1 / (mean_errors + 1e-8)
        self.weights = inv_errors / inv_errors.sum()
```

---

## Conformal Prediction

### Distribution-Free Prediction Intervals

```python
class ConformalForecaster:
    """
    Conformal prediction for guaranteed coverage
    """

    def __init__(self, base_model, alpha=0.1):
        self.model = base_model
        self.alpha = alpha  # Miscoverage rate
        self.calibration_scores = None

    def calibrate(self, X_cal, y_cal):
        """
        Calibrate on held-out calibration set
        """
        predictions = self.model.predict(X_cal)

        # Nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - predictions)

        # Find quantile threshold
        n = len(self.calibration_scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(self.calibration_scores, q)

    def predict(self, X):
        """
        Predict with conformal intervals
        """
        point_pred = self.model.predict(X)

        lower = point_pred - self.threshold
        upper = point_pred + self.threshold

        return point_pred, lower, upper

class AdaptiveConformalForecaster:
    """
    Adaptive conformal prediction for non-stationary series
    """

    def __init__(self, base_model, alpha=0.1, gamma=0.01):
        self.model = base_model
        self.alpha = alpha
        self.gamma = gamma  # Adaptation rate
        self.threshold = None

    def update(self, y_true, prediction_set):
        """
        Update threshold based on coverage
        """
        lower, upper = prediction_set
        covered = (y_true >= lower) and (y_true <= upper)

        # Adjust threshold to maintain coverage
        if covered:
            self.threshold *= (1 - self.gamma * (1 - self.alpha))
        else:
            self.threshold *= (1 + self.gamma * self.alpha)
```

---

## Causal Forecasting

### Causal Discovery for Time Series

```python
class CausalForecaster:
    """
    Incorporate causal structure in forecasting
    """

    def __init__(self):
        self.causal_graph = None
        self.model = None

    def discover_causal_structure(self, data, method='pcmci'):
        """
        Discover causal relationships between variables
        """
        if method == 'pcmci':
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests import ParCorr

            pcmci = PCMCI(
                dataframe=data,
                cond_ind_test=ParCorr()
            )

            results = pcmci.run_pcmci(
                tau_max=10,  # Max lag to consider
                pc_alpha=0.05
            )

            self.causal_graph = results['graph']

        return self.causal_graph

    def build_features_from_causes(self, data, target_var):
        """
        Build features using only causal parents
        """
        if self.causal_graph is None:
            raise ValueError("Must discover causal structure first")

        # Find causal parents of target
        parents = self._get_parents(target_var)

        features = pd.DataFrame()
        for var, lag in parents:
            features[f'{var}_lag_{lag}'] = data[var].shift(lag)

        return features

    def _get_parents(self, target_var):
        """Extract causal parents from graph"""

        # Parse causal graph to find parents
        parents = []

        # ... implementation depends on graph format
        return parents

class CounterfactualForecaster:
    """
    What-if analysis for forecasting
    """

    def __init__(self, trained_model, causal_model):
        self.forecast_model = trained_model
        self.causal_model = causal_model

    def counterfactual_forecast(self, X, intervention):
        """
        Forecast under hypothetical intervention

        Args:
            X: Current features
            intervention: Dict of {variable: new_value}
        """

        # Apply intervention
        X_intervened = X.copy()
        for var, value in intervention.items():
            X_intervened[var] = value

            # Propagate effects through causal model
            effects = self.causal_model.get_downstream_effects(var, value)
            for affected_var, effect in effects.items():
                X_intervened[affected_var] += effect

        return self.forecast_model.predict(X_intervened)
```

---

## AutoML for Time Series

### Automated Model Selection

```python
class TimeSeriesAutoML:
    """
    Automated machine learning for time series
    """

    def __init__(self, task='forecasting', time_budget=3600):
        self.task = task
        self.time_budget = time_budget
        self.best_model = None
        self.search_history = []

    def get_model_space(self):
        """Define search space of models"""
        return {
            'statistical': [
                {'model': 'arima', 'params': {'p': [1,2,3], 'd': [0,1], 'q': [1,2]}},
                {'model': 'ets', 'params': {'trend': ['add', 'mul'], 'seasonal': ['add', 'mul']}},
                {'model': 'prophet', 'params': {'seasonality_mode': ['additive', 'multiplicative']}}
            ],
            'ml': [
                {'model': 'lightgbm', 'params': {'n_estimators': [100, 500], 'max_depth': [5, 10]}},
                {'model': 'xgboost', 'params': {'n_estimators': [100, 500], 'learning_rate': [0.01, 0.1]}},
                {'model': 'random_forest', 'params': {'n_estimators': [100, 500]}}
            ],
            'dl': [
                {'model': 'lstm', 'params': {'hidden_size': [32, 64], 'num_layers': [1, 2]}},
                {'model': 'tcn', 'params': {'num_channels': [[32, 64], [64, 128]]}},
                {'model': 'transformer', 'params': {'d_model': [32, 64], 'nhead': [4, 8]}}
            ]
        }

    def search(self, train_data, val_data, metric='mape'):
        """
        Search for best model configuration
        """
        import time
        start_time = time.time()
        best_score = float('inf')

        model_space = self.get_model_space()

        for category, models in model_space.items():
            for model_config in models:
                if time.time() - start_time > self.time_budget:
                    break

                # Try different hyperparameter combinations
                for params in self._param_combinations(model_config['params']):
                    try:
                        model = self._build_model(model_config['model'], params)
                        model.fit(train_data)
                        predictions = model.predict(len(val_data))

                        score = self._evaluate(val_data, predictions, metric)

                        self.search_history.append({
                            'model': model_config['model'],
                            'params': params,
                            'score': score
                        })

                        if score < best_score:
                            best_score = score
                            self.best_model = model

                    except Exception as e:
                        continue

        return self.best_model

    def _param_combinations(self, params):
        """Generate all parameter combinations"""
        from itertools import product

        keys = list(params.keys())
        values = list(params.values())

        for combo in product(*values):
            yield dict(zip(keys, combo))

# Using existing libraries
"""

# AutoTS
from autots import AutoTS

model = AutoTS(
    forecast_length=30,
    frequency='D',
    ensemble='simple',
    model_list='fast'
)
model = model.fit(df, date_col='date', value_col='value')
forecast = model.predict()

# Auto-ARIMA
from pmdarima import auto_arima

model = auto_arima(
    y,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True
)

# Neural Prophet
from neuralprophet import NeuralProphet

model = NeuralProphet(auto_select_best=True)
model.fit(df)
"""
```

---

## Summary

Advanced techniques enable:
- **Probabilistic forecasting**: Quantify uncertainty
- **Hierarchical forecasting**: Coherent multi-level predictions
- **Transfer learning**: Leverage related domains
- **Online learning**: Adapt to changing patterns
- **Conformal prediction**: Guaranteed coverage
- **Causal forecasting**: Understand interventions

### Recommended Reading

1. Hyndman & Athanasopoulos - *Forecasting: Principles and Practice*
2. Petropoulos et al. - *Forecasting: Theory and Practice*
3. Lim & Zohren - *Time Series Forecasting with Deep Learning*

### Return to [Main README](../README.md)

---

<div align="center">

**[⬆ Back to Top](#)** | **[📚 Main Repository](https://github.com/Gaurav14cs17/ml_system_design)**

Made with 💜 by [Gaurav14cs17](https://github.com/Gaurav14cs17)

</div>
