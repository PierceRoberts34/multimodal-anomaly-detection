from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

class EnvVars:
    SENSOR_METADATA_PATH = os.getenv('SENSOR_METADATA_PATH')
    CSV_SENSOR_DATA_PATH = os.getenv('CSV_SENSOR_DATA_PATH')
    CLEANED_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'_cleaned'+'.parquet'
    APPLIANCE_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'_appliance-usage_cleaned'+'.parquet'
    PARQUET_ACTIVITY_DATA_PATH = os.path.splitext(CSV_SENSOR_DATA_PATH)[0]+'.parquet'
    ANOMALY_REPORT_FOLDER = os.getenv('ANOMALY_REPORT_FOLDER', './reports/csv/')
    LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3-vl:2b') # Lightweight models are suggested
    GRAPH_OUTPUT_FOLDER = os.getenv('GRAPH_OUTPUT_FOLDER', './reports/graphs/')

    @staticmethod
    def getGraphPath(filename):
        return os.path.join(EnvVars.GRAPH_OUTPUT_FOLDER+f'{filename}.png')

    @staticmethod
    def getAnomalyReportPath(filename):
        return os.path.join(EnvVars.ANOMALY_REPORT_FOLDER+f'{filename}_anomalies.csv')