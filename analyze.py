import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


# Assign probabilities to sensor readings using a markov model
def markovProb(df):
    # Determine next reading
    df['next_activity'] = df['activity'].shift(-1)
    
    # Drop the last reading since it won't have a reading
    df = df.dropna(subset=['next_activity'])

    # Determine the markov probability
    markov_prob = df.groupby('activity')['next_activity'].transform(
        lambda x: x.map(x.value_counts(normalize=True))
    )
    return markov_prob

# Determine iforest probability
def iforestProb(df):

    # Encode categorical strings to integers
    le_activity = LabelEncoder()

    df['Activity_Enc'] = le_activity.fit_transform(df['activity'])

    df['Time_Sec'] = pd.to_datetime(df['start_time']).astype(int)/ 10**9
    # Create feature array (X)
    X = df[['Time_Sec', 'Activity_Enc']].values

    # window_size: how many points to keep in the ensemble
    # n_estimators: number of trees
    model = IsolationForest(n_estimators=100)

    # Higher scores indicate higher anomaly probability
    model.fit(X)
    scores = model.decision_function(X)

    return scores