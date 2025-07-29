import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import sys  # For command-line args (optional)

# Function to spread values (add noise + rescale to [-1, 1])
def spread_and_rescale(series, noise_level=0.05):
    noisy = series + np.random.normal(0, noise_level, len(series))
    noisy = np.clip(noisy, series.min(), series.max())
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(noisy.values.reshape(-1, 1)).flatten()
    return scaled

# Load CSV (use command-line arg or hardcoded path)
if len(sys.argv) > 1:
    csv_filename = sys.argv[1]  # e.g., python spread_plot.py input.csv
else:
    csv_filename = 'song_emotion_predictions_taylor_francis_disp.csv'  # Default

df = pd.read_csv(csv_filename)

# Apply spreading (adjust noise_level for more/less spread)
df['arousal_spread'] = spread_and_rescale(df['arousal'], noise_level=0.1)  # Increase for more spread
df['valence_spread'] = spread_and_rescale(df['valence'], noise_level=0.1)

# Save new CSV
new_csv_filename = 'spread_song_emotion_predictions.csv'
df.to_csv(new_csv_filename, index=False)
print(f"New CSV saved: {new_csv_filename}")

# Generate PNG plot (Thayer-style)
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
