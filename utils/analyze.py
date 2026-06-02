"""
Contains machine learning algorithms used in the project
"""

import duckdb
from sklearn.ensemble import IsolationForest
from treeple import ExtendedIsolationForest
from sklearn.preprocessing import LabelEncoder

from utils.environment import EnvVars

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

def iforestProb(df):
    le_activity = LabelEncoder()

    # Create features for machine learning model
    df['sensorEnc'] = le_activity.fit_transform(df['sensor'])

    X = df[['sensorEnc', 'sinTransform', 'cosTransform']].values

    model = IsolationForest(n_estimators=200)

    # Higher scores indicate higher anomaly probability
    model.fit(X)
    
    scores = model.decision_function(X)
    
    return scores

# Determine extended iforest probability
def eifProb(df):
    # Encode categorical strings to integers
    le_activity = LabelEncoder()

    # Create features for machine learning model
    df['sensorEnc'] = le_activity.fit_transform(df['sensor'])

    # Create feature array (X)
    X = df[['sensorEnc', 'sinTransform', 'cosTransform']].to_numpy()

    # window_size: how many points to keep in the ensemble
    # n_estimators: number of trees
    model = ExtendedIsolationForest(n_estimators=200, 
                                    feature_combinations=2,
                                    random_state=42)

    # Higher scores indicate higher anomaly probability
    model.fit(X)
    scores = model.decision_function(X)

    return scores