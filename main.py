from datetime import datetime
import json
import time
from typing import List

import duckdb
import numpy as np
import pandas as pd
from ollama import chat, create
from pydantic import BaseModel, ValidationError

from environment import EnvVars

rawdata = EnvVars.CSV_SENSOR_DATA_PATH
metadata_file = EnvVars.SENSOR_METADATA_PATH
appliance_use_data = EnvVars.CSV_APPLIANCE_DATA_PATH
cleaned_data = EnvVars.CLEANED_CSV_DATA_PATH 
parquet_data = EnvVars.PARQUET_SENSOR_DATA_PATH


class Decision(BaseModel):
    row_id: int
    reasoning: str
    is_severe: bool

# Define the output structure with Pydantic
# This restricts the llm's responses
class ResponseFormat(BaseModel):
    decisions: List[Decision]

# Clean the data for analysis
def prepareData():
    # Import data from filesystem and set headers
    colnames = ['status', 'date', 'time', 'sensor']
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
def markovProb():
    # Import data from filesystem
    df = pd.read_csv(cleaned_data)

    # Determine next reading
    df['next_sensor'] = df['sensor'].shift(-1)

    # Drop the last reading since it won't have a reading
    df = df.dropna(subset=['next_sensor'])

    # Determine the markov probability
    df['markov_prob'] = df.groupby('sensor')['next_sensor'].transform(
        lambda x: x.map(x.value_counts(normalize=True))
    )

    # Mark values lower than anomaly threshold
    df['anomaly'] = [1 if x <= 0.05 else 0 for x in df['markov_prob']]

    # Save the result
    start_time = time.time()
    df.to_parquet(parquet_data)
    end_time = time.time()
    print(f"DataFrame written to Parquet in {end_time - start_time:.2f} seconds")
    return None

def anomalyFlag(df, batch_size=10):
    df = df.reset_index().rename(columns={'index': 'row_id'})
    results = {}

    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        chunk_json = chunk[['row_id', 'timestamp', 'status', 'sensor', 'anomaly']].to_json(orient='records')
        
        response = chat('data-agent', 
                        messages=[{'role': 'user', 'content': chunk_json}],
                        format=ResponseFormat.model_json_schema())
        
        try:
            output = ResponseFormat.model_validate_json(response.message.content)
            for item in output.decisions:
                results[item.row_id] = {
                    'is_severe': item.is_severe,
                    'reason': item.reasoning
                }

        except Exception as e:
            print(f"Error at batch {i}: {e}")
    df['is_severe'] = df['row_id'].map(lambda x: results.get(x, {}).get('is_severe', False))
    df['reasoning'] = df['row_id'].map(lambda x: results.get(x, {}).get('reason', 'No reason provided'))

    return df

def main():
    # Prepare the data
    prepareData()
    # Compute the markov probabilities
    markovProb()

    # Create a custom model for ease of prompting
    system_prompt = """
    You are a sensor data expert. Analyze these events for anomalies.
    The 'anomaly' field (1=potential, 0=normal) is a Markov-based suggestion.
    Refine this:
    - Flag (true) if the sequence is physically impossible or indicates a safety risk.
    - Ignore (false) if it looks like sensor bounce or common noise.

    Return your answer as a JSON object with 'decisions' containing 'row_id', 'is_severe', and 'reasoning' (only if severe).
    """
    create(model='data-agent', from_=EnvVars.LLM_MODEL, system=system_prompt)

    db = duckdb.connect()
    db.execute(f"CREATE VIEW subject1 AS SELECT * FROM read_parquet('{parquet_data }')")

    query = f"SELECT * FROM subject1 LIMIT 10" # Limit results for testing purposes
    df = db.execute(query).df()
    db.close()

    # Apply the function and create/update the column
    start_time = time.time()
    df = anomalyFlag(df, batch_size=2)
    
    end_time = time.time()
    print(f"All anomalies reviewed in {end_time - start_time:.2f} seconds")

    df.to_csv(EnvVars.ANOMALY_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
