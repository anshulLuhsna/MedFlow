"""
Generate outbreak and disruption events
"""

import uuid
from datetime import timedelta

def format_events_for_db(outbreak_events: list, supply_disruptions: list) -> list:
    """Format events for database insertion"""
    
    db_events = []
    
    # Outbreak events
    for event in outbreak_events:
        # Get affected hospital IDs (you'll need to filter by region)
        db_event = {
            "id": str(uuid.uuid4()),
            "event_type": "outbreak",
            "event_name": event["name"],
            "affected_region": ", ".join(event["affected_regions"]),
            "start_date": event["start_date"].date(),
            "end_date": (event["start_date"] + timedelta(days=event["duration_days"])).date(),
            "severity": event["severity"],
            "impact_description": f"Admission multiplier: {event['admission_multiplier']}x"
        }
        db_events.append(db_event)
    
    # Supply disruption events
    for event in supply_disruptions:
        db_event = {
            "id": str(uuid.uuid4()),
            "event_type": "supply_disruption",
            "event_name": event["name"],
            "affected_region": ", ".join(event["affected_regions"]),
            "start_date": event["start_date"].date(),
            "end_date": (event["start_date"] + timedelta(days=event["duration_days"])).date(),
            "severity": "medium",
            "impact_description": f"Supply reduction: {int(event['supply_reduction']*100)}%"
        }
        db_events.append(db_event)
    
    return db_events