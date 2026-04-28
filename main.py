import json
import time

import duckdb
from matplotlib import pyplot as plt
import pandas as pd
from ollama import generate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from environment import EnvVars

rawdata = EnvVars.CSV_SENSOR_DATA_PATH
metadata_file = EnvVars.SENSOR_METADATA_PATH
appliance_use_data = EnvVars.CSV_APPLIANCE_DATA_PATH
cleaned_data = EnvVars.CLEANED_CSV_DATA_PATH 
parquet_data = EnvVars.PARQUET_SENSOR_DATA_PATH

# Clean the data for analysis
def prepareData():
    # Import data from filesystem and set headers
    colnames = ['event', 'date', 'time', 'sensor']
    df = pd.read_csv(rawdata, header=None, names=colnames)

    # Remove unlabeled sensors (noise) from the dataset
    df = df[df['sensor']!='YYY']

    # Convert to Datetime
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Sorting for Time-Series Analysis
    df = df.sort_values(by=['timestamp', 'sensor'])

    # Fix Sensor Jitter (Debouncing)
    # If a sensor triggers again within 2 seconds, we treat it as noise/redundancy.
    df['time_diff'] = df.groupby('sensor')['timestamp'].diff().dt.total_seconds()
    df = df[~((df['time_diff'] < 2) & (df['time_diff'] >= 0))]
    df = df.drop(columns=['time_diff', 'date', 'time'])

    # Open the file and load its contents
    with open(metadata_file, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    # Set vibration sensor delays based on the metadata
    for sensor, info in metadata['vibration'].items():
        delay = info['Delay']
        df.loc[df['sensor'] == sensor, 'timestamp'] -= pd.Timedelta(seconds=delay)
    
    # Export appliance usage and motion/presence sensors to seperate files
    df[df['sensor'].isin(metadata['appliances'])].to_csv(appliance_use_data, index=False)
    df[~df['sensor'].isin(metadata['appliances'])].to_csv(cleaned_data, index=False)
    return None

# Assign probabilities to sensor readings using a markov model
def markovProb(df):
    # Determine next reading
    df['next_sensor'] = df['sensor'].shift(-1)

    # Drop the last reading since it won't have a reading
    df = df.dropna(subset=['next_sensor'])

    # Determine the markov probability
    markov_prob = df.groupby('sensor')['next_sensor'].transform(
        lambda x: x.map(x.value_counts(normalize=True))
    )
    return markov_prob

# Determine iforest probability
def iforestProb(df):

    # Encode categorical strings to integers
    le_event = LabelEncoder()
    le_sensor = LabelEncoder()

    df['Event_Enc'] = le_event.fit_transform(df['event'])
    df['Sensor_Enc'] = le_sensor.fit_transform(df['sensor'])

    df['Time_Sec'] = pd.to_datetime(df['timestamp']).astype(int)/ 10**9
    # Create feature array (X)
    X = df[['Time_Sec', 'Event_Enc', 'Sensor_Enc']].values

    # window_size: how many points to keep in the ensemble
    # n_estimators: number of trees
    model = IsolationForest(n_estimators=100)

    # Higher scores indicate higher anomaly probability
    model.fit(X)
    scores = model.decision_function(X)

    return scores

def getScores():
    df = pd.read_csv(cleaned_data)

    # Calculate probabilities
    df['markov_prob'] = markovProb(df)
    df['iforest_score'] = iforestProb(df)

    # Save the result
    start_time = time.time()
    df.to_parquet(parquet_data)
    end_time = time.time()
    print(f"DataFrame written to Parquet in {end_time - start_time:.2f} seconds")
    return None

def markovSignal():
    db = duckdb.connect()
    df = db.execute(f" SELECT timestamp, markov_prob, FROM read_parquet('{parquet_data }') ORDER BY timestamp ASC").df()
    db.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    groups = [group for _, group in df.groupby(pd.Grouper(freq='24h')) if not group.empty]
    if len(groups) < 2:
            print("Not enough 24-hour periods to perform a comparison.")
            return

    max_periods = 30
    # Keep only the last 'max_periods' to ensure the final image is readable
    plot_groups = groups[-max_periods:]
    n_plots = len(plot_groups)

    # Create the combined tiled image
    print(f"Generating a single combined plot for the last {n_plots} periods...")
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 2.5 * n_plots), sharey=True)
    
    # Handle the edge case where subplots returns a single Axes object instead of an array
    if n_plots == 2:
        axes = axes.flatten()

    for i, group in enumerate(plot_groups):
        ax = axes[i]
        is_recent = (i == n_plots - 1) # The very last item in the list
        
        # Visual anchors for the LLM
        line_color = 'red' if is_recent else 'blue'
        line_width = 2.0 if is_recent else 1.0
        title = f"MOST RECENT 24H ({group.index[0].strftime('%Y-%m-%d')})" if is_recent else f"Historical Baseline ({group.index[0].strftime('%Y-%m-%d')})"
        
        ax.plot(group.index, group['markov_prob'], color=line_color, linewidth=line_width, alpha=0.9)
        
        ax.set_title(title, fontweight='bold' if is_recent else 'normal')
        ax.set_ylabel("Markov Prob")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Clean up X-axis to only show hours/minutes so it doesn't clutter
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.savefig(EnvVars.MARKOV_PATH, dpi=150)
    plt.close()

    return None


def main():
    # Prepare the data
    prepareData()

    getScores()
    markovSignal()

    # Prompt the Local Multimodal LLM
    print("Feeding combined image to local llm...")
    
    # The prompt explicitly references our layout and color-coding
    prompt_text = (
        "Attached is a single image containing multiple stacked charts showing probability scores over time. "
        "The top charts (in blue) show historical 24-hour baseline periods. "
        "The bottom chart (in red) shows the most recent 24-hour period. "
        "Compare the overall shape, volatility, and magnitude of spikes in the historical baselines to the most recent period. "
        "Is the most recent period significantly different from the historical baselines? Explain your reasoning based strictly on the visual patterns."
    )
    start_time = time.time()
    try:
        response = generate(
            model=EnvVars.LLM_MODEL,
            prompt=prompt_text,
            images=[EnvVars.MARKOV_PATH] 
        )
        
        print("\n=== LLM Analysis ===")
        print(response['response'])
        print("====================")
        
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
    end_time = time.time()
    print(f"Review complete in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
