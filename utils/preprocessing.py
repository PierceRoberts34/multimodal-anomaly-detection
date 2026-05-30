import json

import pandas as pd

from utils.environment import EnvVars

# Clean the data for analysis
def prepareData():
    # Import data from filesystem and set headers
    colnames = ['event', 'date', 'time', 'sensor']
    df = pd.read_csv(EnvVars.CSV_SENSOR_DATA_PATH, header=None, names=colnames)

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
    with open(EnvVars.SENSOR_METADATA_PATH, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    # Set vibration sensor delays based on the metadata
    for sensor, info in metadata['vibration'].items():
        delay = info['Delay']
        df.loc[df['sensor'] == sensor, 'dt'] -= pd.Timedelta(seconds=delay)
    
    # Export appliance usage and motion/presence sensors to seperate files
    df[df['sensor'].isin(metadata['appliances'])].to_parquet(EnvVars.APPLIANCE_DATA_PATH, index=False)
    df[~df['sensor'].isin(metadata['appliances'])].to_parquet(EnvVars.CLEANED_DATA_PATH, index=False)
    return None
