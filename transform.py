import pandas as pd

"""
We want to convert the data to the form 
move participant livingroom1 livingroom3 13 23
sit participant couch 26
turn-on-tv 29
"""

sensor_data_raw_path = "./raw/HomeIntellex_1.csv"

df = pd.read_csv(sensor_data_raw_path, 
                names=["output", "date", "time", "sensor"],  # Custom column names
                header=None)

# Start by converting date and time to a single object

df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])

df = df.drop(['date', 'time'], axis=1)

# Extract location from the data

pattern = r'Presence_|_Vibration|_Motion_Sensor|_NoVibration|_Chair|-Computer_Desk|_Computer_Desk'

df['location'] = df['sensor'].str.replace(pattern, '', regex=True)


# The sensors in the garage and refrigerator need special correction. Create a location map
location_map = {
    'VOCOlinc_Contact_Sensor': 'Garage',
    'Eve_Contact_Sensor': 'Kitchen',
    'Aqara_Vibration_Sensor': 'Livingroom',
    'Office': 'Office',
    'Aqara_Sensor': 'Livingroom',
    'Fridge_Sensor': 'Kitchen'
}

# Apply the map to the unknown locations
df['location'] = df['location'].map(location_map).fillna(df['location'])

# This sensor is unknown and should be inspected in the dataset, but we will remove it for now
df = df[df['location']!='YYY']

# Check that location makes sense
print(df['location'].unique())

df['activity'] = 'unknown'

# Create mapping dictionaries
activity_map = {
    'VOCOlinc_Contact_Sensor': 'using-garage',
    'Eve_Contact_Sensor': 'using-refrigerator',
    'Aqara_Vibration_Sensor': 'biking',
    'Bedroom_Chair_Vibration': 'using-personal-computer',
    'Bedroom_Chair_NoVibration': 'using-personal-computer',
    'Office_Chair_Vibration': 'using-office-computer',
    'Office_Chair_NoVibration': 'using-office-computer',
    'Sofa_Sit': 'sitting',
    'Sofa-Leave': 'sitting',
    'Chair-Sit': 'sitting',
    'Chair_Leave': 'sitting',
}

# Apply mappings to create activities
df['activity'] = df['sensor'].map(activity_map).fillna(df['activity'])
df['activity'] = df['output'].map(activity_map).fillna(df['activity'])

# Represents activities with concrete start/stop times (easy to find durations)
pairs = {
    'On_Computer_B': 'Left_Computer_B',
    'On_Computer_O': 'Left_Computer_O',
    'GarageDoor_Open': 'GarageDoor_Closed',
    'FridgeDoor_Open': 'FridgeDoor_Closed',
    'Sofa_Sit': 'Sofa-Leave', 
    'Chair-Sit': 'Chair_Leave',
    'Bike_Running': 'Bike-Stopped',
}

active_events = {}
results = []

for _, row in df.iterrows():
    event = row['output']
    time = row['timestamp']
    activity_name = row['activity']

    for start, end in pairs.items():
        if start in event:
            active_events[activity_name] = time
            break
            
        # Check if this is an "END" event
        elif end in event:
            if activity_name in active_events:
                results.append({
                    'Activity': activity_name,
                    'Start': active_events[activity_name],
                    'End': time
                })
                del active_events[activity_name] # Clear the state
            break

# Create an activity dataframe
activities = pd.DataFrame(results)

# Calculate duration as the difference in seconds between start time and end time.
activities['Duration_(Seconds)'] = (activities['End'] - activities['Start']).dt.total_seconds()

results = []

df['location-change'] = df['location'] != df['location'].shift()

movement = df[['timestamp','location']][df['location-change']]
movement['Start'] = movement['timestamp'].shift()
movement = movement.rename(columns = {'timestamp': 'End'})
movement['Activity'] = 'moving'
movement['Duration_(Seconds)'] = (movement['End'] - movement['Start']).dt.total_seconds()

activities['group'] = (activities['Activity'] != activities['Activity'].shift()).cumsum()

activities = activities.groupby(['group', 'Activity']).agg(
    Start=('Start', 'first'),
    End=('End', 'last'),
    Duration_(Seconds)=('Duration_(Seconds)', 'sum')
).reset_index().drop(columns='group')

output_path = 'activities.csv'
activities.to_csv(output_path)
