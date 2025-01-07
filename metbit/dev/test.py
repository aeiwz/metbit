import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

# Load the data
df = pd.read_csv('/Users/aeiwz/Github/metbit/metbit/dev/test.csv')
df = df.iloc[:, :-1]
ppm_values = df.columns.astype(float)
spectra = df.to_numpy()

# Compute the reference spectrum as the median
ref = np.median(spectra, axis=0)

# Function to align a spectrum using DTW with window constraint
def align_spectrum(reference, target, window=None):
    alignment_path = dtw.warping_path(reference, target, use_c=True, window=window)
    aligned_target = np.zeros_like(reference)
    for i, j in alignment_path:
        aligned_target[i] = target[j]
    return aligned_target

# Perform alignment on full-resolution data
window = int(len(ref) * 0.1)  # 10% of the spectrum length
aligned_spectra = np.zeros_like(spectra)

for i in range(spectra.shape[0]):
    aligned_spectra[i] = align_spectrum(ref, spectra[i], window=window)

# Plot original and aligned spectra
plt.figure(figsize=(12, 8))

# Plot original spectra
plt.subplot(2, 1, 1)
for i in range(spectra.shape[0]):
    plt.plot(ppm_values, spectra[i], label=f'Unaligned Spectrum {i+1}')
plt.gca().invert_xaxis()
plt.title('Original Spectra (Unaligned)')
plt.xlabel('Chemical Shift (ppm)')
plt.ylabel('Intensity')
plt.legend()

# Plot aligned spectra
plt.subplot(2, 1, 2)
for i in range(aligned_spectra.shape[0]):
    plt.plot(ppm_values, aligned_spectra[i], label=f'Aligned Spectrum {i+1}')
plt.gca().invert_xaxis()
plt.title('Aligned Spectra')
plt.xlabel('Chemical Shift (ppm)')
plt.ylabel('Intensity')
plt.legend()

plt.tight_layout()
plt.show()

# Save aligned spectra to CSV for further analysis
aligned_df = pd.DataFrame(aligned_spectra, columns=ppm_values)
aligned_df.to_csv('/Users/aeiwz/Github/metbit/metbit/dev/aligned_spectra.csv', index=False)