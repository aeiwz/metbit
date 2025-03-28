import pandas as pd
from neuroCombat import neuroCombat

# Your LC-MS data (samples × features)
X = X.copy()

# Your batch info
batch = meta['batch_number']

# Transpose: now features × samples (required by neuroCombat)
X_t = X.T

# Ensure covariates (batch) align with samples
covars = pd.DataFrame({'batch': batch.values}, index=X.index)

# Apply neuroCombat
combat_result = neuroCombat(dat=X_t, covars=covars, batch_col='batch')

# Transpose back: samples × features
data_corrected = pd.DataFrame(combat_result['data'].T, index=X.index, columns=X.columns)

# Save
#data_corrected.to_csv("intensity_corrected.csv")
