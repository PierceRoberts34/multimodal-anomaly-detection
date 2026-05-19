import pandas as pd
from sklearn.ensemble import IsolationForest
from treeple import ExtendedIsolationForest
from sklearn.preprocessing import LabelEncoder


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