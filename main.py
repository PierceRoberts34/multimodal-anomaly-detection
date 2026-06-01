"""
Main file, cleans csv and converts to parquet, then performs analysis
"""

import time

import duckdb
from collections import defaultdict
from matplotlib import pyplot as plt

import pandas as pd
from ollama import generate

from utils.environment import EnvVars
from utils.preprocessing import prepareData
from utils.analyze import markovProb, iforestProb, eifProb

metadata_file = EnvVars.SENSOR_METADATA_PATH
appliance_use_data = EnvVars.APPLIANCE_DATA_PATH
cleaned_data = EnvVars.CLEANED_DATA_PATH

def getScores():
    period = 24.0 * 60.0 * 60.0
    db = duckdb.connect()
    # Sin and Cos transform perioding data to use as ML features
    query = f"""
            CREATE VIEW ml_data AS 
            SELECT *,
                    SIN(epoch(CAST(dt AS TIMESTAMP)) / {period} * 2 * PI()) AS sinTransform,
                    COS(epoch(CAST(dt AS TIMESTAMP)) / {period} * 2 * PI()) AS cosTransform
            FROM read_parquet('{cleaned_data}');
            """
    db.execute(query)
    df = db.execute(f"SELECT dt, sensor, sinTransform, cosTransform FROM ml_data").df()
    df['markov_prob'] = markovProb(df)
    df['iforest_score'] = iforestProb(df)
    df['eif_score'] = eifProb(df)
    db.sql("CREATE TABLE scores AS SELECT * FROM df")
    query = f"""
        COPY (
        SELECT ml_data.*, scores.markov_prob, scores.iforest_score, scores.eif_score
        FROM ml_data
        LEFT JOIN scores
        ON ml_data.dt = scores.dt
        ) TO '{cleaned_data}' (FORMAT 'parquet');
        """
    db.execute(query)
    db.close()
    return None

def probabilitySignal(signal, threshold):
    db = duckdb.connect()
    # Get Data
    df = db.execute(f"SELECT dt, {signal} FROM read_parquet('{cleaned_data }') ORDER BY dt ASC").df()
    # Get labels for potential anomalies
    anomalies = db.execute(f"SELECT dt, sensor, {signal} FROM read_parquet('{cleaned_data }') WHERE {signal} < {threshold} ORDER BY dt ASC").df()
    db.close()
    outputPath = EnvVars.getAnomalyReportPath(signal)
    anomalies.to_csv(outputPath, index=False)

    df['dt'] = pd.to_datetime(df['dt'])
    anomalies['dt'] = pd.to_datetime(anomalies['dt'])
    df.set_index('dt', inplace=True)

    groups = [group for _, group in df.groupby(pd.Grouper(freq='24h')) if not group.empty]

    # Keep only the last 'max_periods' to ensure the final image is readable
    max_periods = 30
    plot_groups = groups[-max_periods:]
    n_plots = len(plot_groups)

    # Create the combined tiled image
    print(f"Generating a single combined plot for the last {n_plots} periods...")
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 2.5 * n_plots), sharey=True, squeeze=False)
    axes = axes.flatten()
    for i, group in enumerate(plot_groups):
        ax = axes[i]
        day_start = group.index.min()
        day_end = group.index.max()
        title = f"{group.index[0].strftime('%Y-%m-%d')}"
        
        ax.scatter(group.index, group[signal], color='blue', s=5, alpha=0.6)

        mask = (anomalies['dt'] >= day_start) & (anomalies['dt'] <= day_end)
        group_anomalies = anomalies.loc[mask]
        
        if not group_anomalies.empty:
            ax.scatter(group_anomalies['dt'], group_anomalies[signal], 
                       color='red', s=5, label='Anomaly', zorder=5)
        
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
    dt = time.time()
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
    print(f"Review complete in {end_time - dt:.2f} seconds")
    return None

def main():
    # Prepare the data
    prepareData()

    # Calculate Probabilities
    getScores()

    # Create graph of last 30 days
    markovPath = probabilitySignal('markov_prob', 0.01)
    iforestPath = probabilitySignal('iforest_score', -0.15)
    eifPath = probabilitySignal('eif_score', -0.15)

    # Prompt multimodal llm with probability
    promptLLM(markovPath)
    promptLLM(iforestPath)


if __name__ == "__main__":
    main()
