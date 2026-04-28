from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

class EnvVars:
    SENSOR_METADATA_PATH = os.getenv('SENSOR_METADATA_PATH')
    CSV_SENSOR_DATA_PATH = os.getenv('CSV_SENSOR_DATA_PATH')
    CLEANED_CSV_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'_cleaned'+'.csv'
    CSV_APPLIANCE_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'_appliance-usage_cleaned'+'.csv'
    PARQUET_SENSOR_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'.parquet'
    ANOMALY_DATA_PATH = os.getenv('ANOMALY_DATA_PATH')
    LLM_MODEL = os.getenv('LLM_MODEL') # Lightweight models are suggested
    GRAPH_OUTPUT_FOLDER = os.getenv('GRAPH_OUTPUT_FOLDER')

    @staticmethod
    def getGraphPath(filename):
        return os.path.join(EnvVars.GRAPH_OUTPUT_FOLDER+f'{filename}.png')

