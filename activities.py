"""
HomeIntellex → Activity Log Generator
======================================
Converts raw sensor events into a clean, human-readable activity CSV.

Output format (one row per meaningful activity):
  move participant entryway livingroom1 0 11
  move participant livingroom1 livingroom3 13 23
  sit participant couch 26
  turn-on-tv 29
  get-up participant couch 53
  move participant livingroom3 kitchen 56 63
  open-fridge 66
  turn-on-microwave 69
  close-fridge 73
  move participant kitchen livingroom3 75 85
  sit participant couch 87

Each session = one calendar day.
Output file: activity_log.csv  (one row per event, includes day/date column)
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────
#  SENSOR  →  ROOM  mapping
# ─────────────────────────────────────────
SENSOR_TO_ROOM = {
    "entrance":               "entryway",
    "Kitchen_Motion_Sensor":  "kitchen",
    "Office_Motion_Sensor":   "office",
    "Bedroom_Motion_Sensor":  "bedroom",
    "Presence_Livingroom":    "livingroom",   # sub-zones handled below
    "Aqara_Vibration_Sensor": "livingroom",
    "Eve_Contact_Sensor":     "kitchen",
    "Fridge_Vibration_Sensor":"kitchen",
    "VOCOlinc_Contact_Sensor":"garage",
    "Office_Chair_Vibration": "office",
    "Office_Chair_NoVibration":"office",
    "Bedroom_Chair_Vibration":"bedroom",
    "Bedroom_Chair_NoVibration":"bedroom",
    "Office_Computer_Desk":   "office",
    "Bedroom_Computer_Desk":  "bedroom",
    "Bedroom-Computer_Desk":  "bedroom",
}

# Livingroom sub-zones based on Presence_Livingroom events
LR_ENTRY_EVENTS = {
    "Enter_Livingroom", "In_Livingroom", "In/Out_Livingroom",
}
LR_INSIDE_EVENTS = {
    "Inside_Livingroom", "BedDoor_Side",
}
LR_LEAVE_EVENTS = {
    "Leave_Livingroom", "Exit_Livingroom", "Outside_Livingroom",
}

# Furniture / action events → output strings
FURNITURE_SIT = {
    "Sofa_Sit": "couch",
    "Sofa_Sitting": "couch",   # variant
    "Chair-Sit": "chair",
    "Bike_Sit": "exercise-bike",
    "Bile_Sit": "exercise-bike",  # typo
    "On_Computer_O": "computer-office",
    "On_Computer_B": "computer-bedroom",
    "On_Computer": "computer",
}
FURNITURE_LEAVE = {
    "Sofa-Leave": "couch",
    "Chair_Leave": "chair",
    "Bike_Leave": "exercise-bike",
    "Left_Computer_O": "computer-office",
    "Left_Computer_B": "computer-bedroom",
    "Left_Computer": "computer",
}
POINT_OF_INTEREST_ENTER = {
    "TV-Side": "tv-area",
    "Table_Side": "table",
    "Window_Side": "window",
    "Beside_Window": "window",
    "Backyard-Side": "backyard",
    "Bedroom_Side": "bedroom-door",
    "Bike_Side": "exercise-bike",
    "Bike_Running": "exercise-bike",
    "Bike_Used": "exercise-bike",
}
POINT_OF_INTEREST_LEAVE = {
    "TV_Leave": "tv-area",
    "Table_Leave": "table",
    "Window_Leave": "window",
    "Leave_Window": "window",
    "Backyard_Leave": "backyard",
    "Bedroom_Leave": "bedroom-door",
    "Bike-Stopped": "exercise-bike",
}
APPLIANCE_EVENTS = {
    "FridgeDoor_Open":   "open-fridge",
    "FridgeDoor_Closed": "close-fridge",
    "Fridge_Open":       "open-fridge",     # variant / vibration sensor
    "GarageDoor_Open":   "open-garage",
    "GarageDoor_Closed": "close-garage",
}
# open-door (entrance) → arrival
ENTRANCE_EVENTS = {"open"}

# ─────────────────────────────────────────
#  CORE: parse one day into activity lines
# ─────────────────────────────────────────
class ActivityTracker:
    """State machine that tracks user movements and actions throughout the day."""
    
    def __init__(self, day_start):
        self.day_start = day_start
        self.activities = []
        
        # Room state
        self.current_room = None
        self.room_entry_time = None
        self.last_entrance_time = None
        self.last_room_fire = {}
        
        # Furniture / POI state
        self.sitting_on = None
        self.sit_start = None
        self.pending_get_up_time = None  # Tracks potential end of a sit
        
        self.poi_in = None
        self.poi_start = None

    def record_activity(self, act_type, start_time=None, end_time=None, **kwargs):
        """Standardizes activity recording for both durations and point-in-time events."""
        activity = {"activity": act_type}
        
        if start_time is not None and end_time is not None:
            if end_time >= start_time:
                activity["start_time"] = start_time
                activity["duration"] = end_time - start_time
        elif start_time is not None:
            activity["start_time"] = start_time
            
        activity.update(kwargs)
        self.activities.append(activity)

    def _flush_pending_sit(self, end_time=None):
        """Finalizes and records a sitting activity."""
        if self.sitting_on is not None:
            # Use the time they got up, or fallback to the provided end_time
            final_end = self.pending_get_up_time or end_time
            if final_end and final_end >= self.sit_start:
                self.record_activity(
                    f"siting on {self.sitting_on}", 
                    start_time=self.sit_start, 
                    end_time=final_end
                )
            
            # Reset sit state
            self.sitting_on = None
            self.sit_start = None
            self.pending_get_up_time = None

    def finalize(self, last_dt):
        """Called at the end of the day to close out any ongoing activities."""
        self._flush_pending_sit(end_time=last_dt)

    def process_row(self, dt, sensor, event, detected_room):
        """Processes a single sensor event and updates state."""
        
        is_sit_event = event in FURNITURE_SIT
        is_leave_event = event in FURNITURE_LEAVE

        # ── PENDING GET-UP CHECK ──
        # If we got up recently, wait to see what this new event is
        if self.pending_get_up_time is not None:
            if is_sit_event:
                # We sat back down. Cancel the get-up.
                self.pending_get_up_time = None
            elif not is_leave_event:
                # The next event is NOT a sit (e.g., motion in a room).
                # The sit is definitively over. Flush it.
                self._flush_pending_sit()

        # ── SIT events ──
        if is_sit_event:
            furniture = FURNITURE_SIT[event]
            if self.sitting_on != furniture:
                # If we switched to different furniture without a leave event, close the old one
                self._flush_pending_sit(end_time=dt)
                
                self.sitting_on = furniture
                self.sit_start = dt
                self.pending_get_up_time = None

        # ── LEAVE / GET-UP events ──
        elif is_leave_event:
            furniture = FURNITURE_LEAVE[event]
            if self.sitting_on == furniture:
                # Mark the potential end time, but wait for the next sensor
                self.pending_get_up_time = dt

        # ── ENTRANCE OPEN → entryway ──
        elif event in ENTRANCE_EVENTS and sensor == "entrance":
            if self.current_room != "entryway":
                prev = self.current_room
                prev_time = self.room_entry_time
                
                if prev is not None:
                    self.record_activity(
                        f"move from {prev} to entryway", 
                        start_time=prev_time, 
                        end_time=dt
                    )
                self.current_room = "entryway"
                self.room_entry_time = dt
                self.last_entrance_time = dt

        # ── ROOM DETECTION from motion sensors ──
        elif detected_room and detected_room != self.current_room:
            last_fire = self.last_room_fire.get(detected_room)
            if last_fire is not None and (dt - last_fire) < timedelta(minutes=3):
                self.last_room_fire[detected_room] = dt
            else:
                prev = self.current_room
                prev_time = self.room_entry_time if self.room_entry_time is not None else dt

                if prev is not None and prev != detected_room:
                    self.record_activity(
                        f"move from {prev} to {detected_room}", 
                        start_time=prev_time, 
                        end_time=dt
                    )

                self.current_room = detected_room
                self.room_entry_time = dt
                self.last_room_fire[detected_room] = dt

        # ── POINT-OF-INTEREST approach ──
        elif event in POINT_OF_INTEREST_ENTER:
            poi = POINT_OF_INTEREST_ENTER[event]
            if self.poi_in != poi:
                self.poi_in = poi
                self.poi_start = dt
                if poi == "exercise-bike":
                    self.record_activity("start-exercise-bike", start_time=dt)
                elif poi == "tv-area":
                    self.record_activity("turn-on-tv", start_time=dt)

        # ── POINT-OF-INTEREST leave ──
        elif event in POINT_OF_INTEREST_LEAVE:
            poi = POINT_OF_INTEREST_LEAVE[event]
            if self.poi_in == poi:
                if poi == "exercise-bike":
                    self.record_activity("stop-exercise-bike", start_time=dt)
                elif poi == "tv-area":
                    self.record_activity("turn-off-tv", start_time=dt)
                self.poi_in = None

        # ── APPLIANCE events ──
        elif event in APPLIANCE_EVENTS:
            self.record_activity(APPLIANCE_EVENTS[event], start_time=dt)

        # ── COMPUTER events ──
        elif event in ("On_Computer_O", "On_Computer_B", "On_Computer"):
            self.record_activity("start-computer", sensor=sensor, start_time=dt)
        elif event in ("Left_Computer_O", "Left_Computer_B", "Left_Computer"):
            self.record_activity("leave-computer", sensor=sensor, start_time=dt)


def room_from_sensor(sensor, event):
    if sensor == "Presence_Livingroom":
        if event in LR_ENTRY_EVENTS:
            return "livingroom1"
        if event in LR_INSIDE_EVENTS:
            return "livingroom3"
    return SENSOR_TO_ROOM.get(sensor)


def process_day(day_rows):
    """
    Returns list of dicts with keys: activity, start_time, duration, etc.
    """
    if day_rows.empty:
        return []

    day_start = day_rows.iloc[0, 0]
    tracker = ActivityTracker(day_start)

    last_dt = None
    for _, row in day_rows.iterrows():
        dt = row['dt']
        last_dt = dt
        event = row['event']
        sensor = row['sensor']
        detected_room = room_from_sensor(sensor, event)
        
        tracker.process_row(dt, sensor, event, detected_room)
        
    # Ensure any ongoing activities at the end of the day's data are closed out
    tracker.finalize(last_dt)

    return tracker.activities