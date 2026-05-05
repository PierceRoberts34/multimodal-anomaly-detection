import json
import time

import duckdb
from collections import defaultdict
from matplotlib import pyplot as plt

import pandas as pd
from ollama import generate

from environment import EnvVars
from activities import process_day
from analyze import markovProb, iforestProb

rawdata = EnvVars.CSV_SENSOR_DATA_PATH
metadata_file = EnvVars.SENSOR_METADATA_PATH
appliance_use_data = EnvVars.CSV_APPLIANCE_DATA_PATH
cleaned_data = EnvVars.CLEANED_CSV_DATA_PATH 
parquet_data = EnvVars.PARQUET_ACTIVITY_DATA_PATH


# Clean the data for analysis
def prepareData():
    # Import data from filesystem and set headers
    colnames = ['event', 'date', 'time', 'sensor']
    df = pd.read_csv(rawdata, header=None, names=colnames)

    # Remove unlabeled sensors (noise) from the dataset
    df = df[df['sensor']!='YYY']

    # Convert to Datetime
    dt = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    df.insert(loc = 0,
          column = 'dt',
          value = dt)

    # Sorting for Time-Series Analysis
    df = df.sort_values(by=['dt', 'sensor'])

    # Fix Sensor Jitter (Debouncing)
    # If a sensor triggers again within 2 seconds, we treat it as noise/redundancy.
    df['time_diff'] = df.groupby('sensor')['dt'].diff().dt.total_seconds()
    df = df[~((df['time_diff'] < 2) & (df['time_diff'] >= 0))]
    df = df.drop(columns=['time_diff', 'date', 'time'])

    # Open the file and load its contents
    with open(metadata_file, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    # Set vibration sensor delays based on the metadata
    for sensor, info in metadata['vibration'].items():
        delay = info['Delay']
        df.loc[df['sensor'] == sensor, 'dt'] -= pd.Timedelta(seconds=delay)
    
    # Export appliance usage and motion/presence sensors to seperate files
    df[df['sensor'].isin(metadata['appliances'])].to_csv(appliance_use_data, index=False)
    df[~df['sensor'].isin(metadata['appliances'])].to_csv(cleaned_data, index=False)
    return None

def sensorsToActivities():
    df = pd.read_csv(cleaned_data, parse_dates=['dt'])
    grouped = df.groupby(df['dt'].dt.date)
    actions = []
    for day, window in grouped:
        daily_actions = process_day(window)
        for act in daily_actions:
            actions.append(act)
    out = pd.DataFrame(actions)
    out.sort_values(by=['start_time'])
    out.to_parquet(parquet_data)
    return None


def getScores():
    df = pd.read_parquet(parquet_data)
    # Calculate probabilities
    df['markov_prob'] = markovProb(df)
    df.drop(columns=['next_activity'], inplace=True)
    df['iforest_score'] =iforestProb(df)
    # Save modified file
    df.to_parquet(parquet_data)
    return None


'''
def highlightAnomalies():
    threshold = 0.05
    db = duckdb.connect()
    anomalies = db.execute(f"SELECT start_time, activity, next_activity, markov_prob, FROM read_parquet('{parquet_data }') WHERE markov_prob < {threshold} ORDER BY start_time ASC").df()
    db.close()
    print(anomalies)
'''

def probabilitySignal(signal, threshold, bound='upper'):
    db = duckdb.connect()
    # Get Data
    df = db.execute(f"SELECT start_time, {signal} FROM read_parquet('{parquet_data }') ORDER BY start_time ASC").df()
    # Get labels for potential anomalies
    if bound == 'lower':
        anomalies = db.execute(f"SELECT start_time, {signal} FROM read_parquet('{parquet_data }') WHERE {signal} > {threshold} ORDER BY start_time ASC").df()
    else:
        anomalies = db.execute(f"SELECT start_time, {signal} FROM read_parquet('{parquet_data }') WHERE {signal} < {threshold} ORDER BY start_time ASC").df()
    db.close()

    df['start_time'] = pd.to_datetime(df['start_time'])
    anomalies['start_time'] = pd.to_datetime(anomalies['start_time'])
    df.set_index('start_time', inplace=True)

    groups = [group for _, group in df.groupby(pd.Grouper(freq='24h')) if not group.empty]

    # Keep only the last 'max_periods' to ensure the final image is readable
    max_periods = 30
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
        line_color = 'blue'
        line_width = 1.0
        day_start = group.index.min()
        day_end = group.index.max()
        title = f"{group.index[0].strftime('%Y-%m-%d')}"
        
        ax.plot(group.index, group[f'{signal}'], color=line_color, linewidth=line_width, alpha=0.9)

        mask = (anomalies['start_time'] >= day_start) & (anomalies['start_time'] <= day_end)
        group_anomalies = anomalies.loc[mask]
        
        if not group_anomalies.empty:
            ax.scatter(group_anomalies['start_time'], group_anomalies[signal], 
                       color='red', s=25, label='Anomaly', zorder=5)
        
        ax.set_title(title, fontweight='normal')
        ax.set_ylabel(f"{signal}")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Threshold')
        
        # Clean up X-axis to only show hours/minutes so it doesn't clutter
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    plt.tight_layout()
    outputPath = EnvVars.getGraphPath(signal)
    plt.savefig(outputPath, dpi=150)
    plt.close()

    return outputPath

def promptLLM(imagePath):
    # Prompt the Local Multimodal LLM
    print("Feeding combined image to local llm...")
    
    # The prompt explicitly references our layout and color-coding
    prompt_text = (
        "Attached is a single image containing multiple stacked charts showing probability scores over time."
        "Points with low probability are highlighted red"
        "Decide which graphs, if any, appear significantly different in shape or number of anomalies"
    )
    start_time = time.time()
    try:
        response = generate(
            model=EnvVars.LLM_MODEL,
            prompt=prompt_text,
            images=[imagePath] 
        )
        
        print("\n=== LLM Analysis ===")
        print(response['response'])
        print("====================")
        
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
    end_time = time.time()
    print(f"Review complete in {end_time - start_time:.2f} seconds")
    return None

def main():
    # Prepare the data
    prepareData()

    # Convert sensor data to activity data
    sensorsToActivities()

    # Calculate Probabilities
    getScores()

    # Create graph of last 30 days
    markovPath = probabilitySignal('markov_prob', 0.01)

    # Prompt multimodal llm with probability
    promptLLM(markovPath)
    #promptLLM(iforestPath)


if __name__ == "__main__":
    main()
