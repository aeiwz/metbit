# Test Matrix

| Feature/Module | Scenario | Input Conditions | Expected Output/Error | Test Type | Priority | Test File |
| --- | --- | --- | --- | --- | --- | --- |
| spec_norm.Normalization | Happy path PQN | Small DataFrame with positive values | Rows rescaled against median, shape preserved | unit | P0 | tests/test_spec_norm.py |
| spec_norm.Normalization | Error input | Unconvertible object | `TypeError` raised | unit | P0 | tests/test_spec_norm.py |
| spec_norm.Normalization | Edge zeros | DataFrame containing zeros | No crash, DataFrame returned | unit | P1 | tests/test_spec_norm.py |
| utility.lazypair | Pair generation | DataFrame with 3 groups | Pair names/index lists created, datasets sliced | unit | P0 | tests/test_utility.py |
| utility.lazypair | Error missing column | Missing target column | `KeyError` raised | unit | P0 | tests/test_utility.py |
| utility.lazypair | Error insufficient groups | Single-group data | `ValueError` raised | unit | P0 | tests/test_utility.py |
| utility.Normalise | PQN with imputation | DataFrame containing NaN values | NaNs imputed and PQN-normalised | integration | P1 | tests/test_normalise.py |
| utility.Normalise | Rounding and z-score | Numeric DataFrame | Rounded values, z-score mean 0/std 1 | unit | P0 | tests/test_normalise.py |
| utility.Normalise | Linear normalisation bounds | Positive data | Output within [0,1] | unit | P1 | tests/test_normalise.py |
| scaler.Scaler | Pareto scaling | Small numeric array | Mean-centred output | unit | P0 | tests/test_scaler.py |
| scaler.Scaler | Unit variance scaling | Small numeric array | Mean 0, std 1 | unit | P0 | tests/test_scaler.py |
| scaler.Scaler | Transform before fit | Call transform without fit | `NotFittedError` raised | unit | P0 | tests/test_scaler.py |
