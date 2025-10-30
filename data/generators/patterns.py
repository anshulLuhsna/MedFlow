"""
Utility functions for generating realistic patterns
"""

import numpy as np
from datetime import timedelta

def add_seasonal_pattern(base_value, date, seasonal_factors):
    """Add seasonal variation to a base value"""
    month = date.month
    factor = seasonal_factors.get(month, 1.0)
    return base_value * factor

def add_weekly_pattern(base_value, date, weekly_pattern):
    """Add weekly cyclical pattern"""
    weekday = date.weekday()
    factor = weekly_pattern.get(weekday, 1.0)
    return base_value * factor

def add_noise(value, noise_level=0.15):
    """Add random noise to a value"""
    noise = np.random.normal(1.0, noise_level)
    return max(0, int(value * noise))

def calculate_trend(day_index, total_days, trend_strength=0.1):
    """Calculate linear trend factor"""
    # Slight upward trend over time (increasing demand)
    return 1.0 + (day_index / total_days) * trend_strength

def is_in_event(date, event):
    """Check if date falls within an event period"""
    event_start = event["start_date"]
    event_end = event_start + timedelta(days=event["duration_days"])
    return event_start <= date < event_end

def get_event_multiplier(date, events, region, metric_type="admission"):
    """Get the multiplier effect from active events"""
    multiplier = 1.0
    
    for event in events:
        if is_in_event(date, event):
            # Check if this event affects the region
            if region in event.get("affected_regions", []):
                if metric_type == "admission":
                    multiplier *= event.get("admission_multiplier", 1.0)
                elif metric_type == "resource":
                    # Resource-specific multipliers
                    resource_mults = event.get("resource_multiplier", {})
                    # Will be applied per resource type
                    pass
    
    return multiplier

def generate_realistic_value(
    base_value,
    date,
    day_index,
    total_days,
    region,
    events,
    seasonal_factors,
    weekly_pattern,
    noise_level=0.15
):
    """Generate a realistic value with all patterns applied"""
    
    value = base_value
    
    # Apply trend
    value *= calculate_trend(day_index, total_days)
    
    # Apply seasonal pattern
    value = add_seasonal_pattern(value, date, seasonal_factors)
    
    # Apply weekly pattern
    value = add_weekly_pattern(value, date, weekly_pattern)
    
    # Apply event multiplier
    value *= get_event_multiplier(date, events, region)
    
    # Add random noise
    value = add_noise(value, noise_level)
    
    return max(0, int(value))