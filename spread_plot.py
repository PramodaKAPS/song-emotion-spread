import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import sys

# Function to spread values (add noise + rescale to [-1, 1])
def spread_and_rescale(series, noise_level=0.05):
    noisy = series + np.random.normal(0, noise_level, len(series))
    noisy = np.clip(noisy, series.min(), series.max())
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(noisy.values.reshape(-1, 1)).flatten()
    return scaled

# Default CSV and column names (adjust defaults based on your CSV)
csv_filename = 'spotify_full_with_predictions.csv' if len(sys.argv) < 2 else sys.argv[1]
arousal_col = 'arousal_final' if len(sys.argv) < 3 else sys.argv[2]
valence_col = 'valence_final' if len(sys.argv) < 4 else sys.argv[3]

df = pd.read_csv(csv_filename)

# Check if columns exist
if arousal_col not in df.columns or valence_col not in df.columns:
    raise ValueError(f"CSV must have '{arousal_col}' and '{valence_col}' columns. Use: python3 spread_plot.py [csv] [arousal_col] [valence_col]")

print(f"Using columns: {arousal_col} and {valence_col}")

# Apply spreading
df['arousal_spread'] = spread_and_rescale(df[arousal_col])
df['valence_spread'] = spread_and_rescale(df[valence_col])

# Save new CSV
new_csv_filename = 'spread_' + csv_filename
df.to_csv(new_csv_filename, index=False)
print(f"New CSV saved: {new_csv_filename}")

# Generate PNG plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='valence_spread', y='arousal_spread', data=df, alpha=0.7, color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title('Spread Arousal vs Valence (Thayer Plot)')
plt.xlabel('Valence (Spread, -1 to 1)')
plt.ylabel('Arousal (Spread, -1 to 1)')
plt.xlim(-1, 1)
plt.ylim(-1, 1)

png_filename = 'spread_thayer_plot.png'
plt.savefig(png_filename, dpi=300)
plt.close()
print(f"Plot saved: {png_filename}")

