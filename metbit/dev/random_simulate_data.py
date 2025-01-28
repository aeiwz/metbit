import pandas as pd
import numpy as np


np.random.seed(456)

# Function to generate random samples with dynamic noise based on the original data
def generate_sample_data_with_dynamic_noise(df: pd.DataFrame, n_sample: int, sd_percentage: float):
    new_samples = []
    groups = df['Group'].unique()

    for group in groups:
        group_data = df[df['Group'] == group]
        
        # Initialize an empty list for the generated samples
        group_samples = []

        for col in group_data.columns[1:]:  # Skip the 'Group' column
            col_data = group_data[col]
            
            # Calculate the dynamic SD as a percentage of the original value
            dynamic_sd = np.abs(col_data.values[0] * sd_percentage)

            # Create a mask for positions where the value is 0.0
            zero_mask = (col_data == 0.0)

            # Generate random noise based on the dynamic SD
            noise = np.abs(np.random.normal(loc=0, scale=dynamic_sd, size=(n_sample,)))

            # Start with the original value repeated for all new samples
            samples = np.tile(col_data.values[0], (n_sample,)) + noise
            
            # Repeat the zero_mask for all n_sample rows
            expanded_zero_mask = np.tile(zero_mask.values[0], (n_sample,))

            # Apply the zero_mask: if the original value is 0.0, ensure the new samples are also 0.0
            samples[expanded_zero_mask] = 0.0  # Apply mask to ensure 0.0 values remain unchanged

            # Append the generated samples for this column
            group_samples.append(samples)

        # Stack all the columns together and create a DataFrame
        group_samples_df = pd.DataFrame(np.column_stack(group_samples), columns=group_data.columns[1:])
        group_samples_df.insert(0, 'Group', group)
        new_samples.append(group_samples_df)

    return pd.concat(new_samples, ignore_index=True)


if __name__ == __main__:
  
  data = df  # Your input DataFrame
  
  # Parameters
  sd_percentage = 5.0  # 60% of the real data value for dynamic SD
  n_new = 10           # Number of new samples per group
  
  # Generate new samples
  new_samples_df = generate_sample_data_with_dynamic_noise(data, n_new, sd_percentage)

