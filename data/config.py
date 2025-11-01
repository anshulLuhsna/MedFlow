"""
Configuration for synthetic data generation
"""

from datetime import datetime, timedelta

# Time period
# Extended to 18 months for better LSTM training (more sequences per hospital)
START_DATE = datetime(2023, 11, 1)  # 18 months ago from Oct 2024
END_DATE = datetime(2024, 10, 30)
TOTAL_DAYS = (END_DATE - START_DATE).days

# Hospital configuration
NUM_HOSPITALS = 100
REGIONS = [
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Kerala",
    "Telangana",
    "Andhra Pradesh",
    "Delhi NCR",
    "Uttar Pradesh",
    "Rajasthan",
    "Gujarat",
    "West Bengal",
    "Bihar",
    "Punjab",
    "Haryana",
    "Madhya Pradesh",
    "Assam"
]

# Hospital size distribution
HOSPITAL_SIZES = {
    "small": {"ratio": 0.50, "beds_range": (50, 150)},
    "medium": {"ratio": 0.35, "beds_range": (150, 300)},
    "large": {"ratio": 0.15, "beds_range": (300, 600)}
}

# Hospital specializations
SPECIALIZATIONS = [
    "general",
    "trauma",
    "pediatric",
    "cardiac",
    "respiratory"
]

# Resource types and their parameters
RESOURCES = {
    "ventilators": {
        "baseline_per_bed": 0.02,  # 2 ventilators per 100 beds
        "critical_threshold": 2,
        "consumption_rate": 0.15,  # 15% of ICU patients need ventilators
        "unit": "units"
    },
    "o2_cylinders": {
        "baseline_per_bed": 0.5,  # 50 cylinders per 100 beds
        "critical_threshold": 10,
        "consumption_rate": 0.8,  # 80% replacement rate per week
        "unit": "cylinders"
    },
    "beds": {
        "baseline_per_bed": 1.0,  # Same as hospital capacity
        "critical_threshold": 5,
        "consumption_rate": 0.7,  # 70% occupancy on average
        "unit": "beds"
    },
    "medications": {
        "baseline_per_bed": 10,  # 10 doses per bed
        "critical_threshold": 100,
        "consumption_rate": 5,  # 5 doses per patient per day on average
        "unit": "doses"
    },
    "ppe": {
        "baseline_per_bed": 5,  # 5 PPE sets per bed
        "critical_threshold": 50,
        "consumption_rate": 2,  # 2 sets per patient per day
        "unit": "sets"
    }
}

# Event configurations
OUTBREAK_EVENTS = [
    {
        "name": "TB Outbreak",
        "start_date": datetime(2024, 6, 15),
        "duration_days": 45,
        "affected_regions": ["Bihar", "Uttar Pradesh", "West Bengal", "Madhya Pradesh"],
        "severity": "high",
        "admission_multiplier": 2.5,
        "resource_multiplier": {"medications": 3.0, "ppe": 2.5}
    },
    {
        "name": "Dengue Post-Monsoon Peak",
        "start_date": datetime(2024, 9, 15),
        "duration_days": 40,
        "affected_regions": ["Delhi NCR", "Uttar Pradesh", "Maharashtra", "West Bengal", "Assam"],
        "severity": "medium",
        "admission_multiplier": 1.8,
        "resource_multiplier": {"medications": 2.5, "beds": 1.5, "ppe": 1.5}
    },
    {
        "name": "Air Pollution Respiratory Surge (Delhi NCR)",
        "start_date": datetime(2024, 10, 1),
        "duration_days": 20,
        "affected_regions": ["Delhi NCR", "Haryana", "Punjab"],
        "severity": "medium",
        "admission_multiplier": 1.6,
        "resource_multiplier": {"ventilators": 2.0, "o2_cylinders": 2.2}
    }
]

# Supply disruption events
SUPPLY_DISRUPTIONS = [
    {
        "name": "Monsoon Logistics Disruption",
        "start_date": datetime(2024, 7, 10),
        "duration_days": 30,
        "affected_regions": ["Maharashtra", "Kerala", "Assam", "West Bengal"],
        "affected_resources": ["o2_cylinders", "medications"],
        "supply_reduction": 0.5  # 50% reduction due to flooding and transport delays
    }
]

# Seasonal patterns (flu season, summer lull, etc.)
# Extended to cover full 18-month period (Nov 2023 - Oct 2024)
SEASONAL_FACTORS = {
    11: 1.15,  # November - post-monsoon respiratory illnesses
    12: 1.10,  # December - winter onset, respiratory cases
    1: 1.05,   # January - winter peak respiratory
    2: 0.95,   # February - transition period
    3: 0.90,   # March - spring, lower baseline
    4: 0.95,   # April - pre-summer baseline
    5: 1.05,   # May - heatwave impacts in parts of India
    6: 1.10,   # June - pre-monsoon/early monsoon onset
    7: 1.15,   # July - monsoon illnesses rise
    8: 1.20,   # August - monsoon peak, water/air-borne diseases
    9: 1.25,   # September - post-monsoon vector-borne peaks (e.g., dengue)
    10: 1.20   # October - elevated respiratory and vector-borne cases
}

# Weekly patterns (lower on weekends)
WEEKLY_PATTERN = {
    0: 1.0,   # Monday
    1: 1.05,  # Tuesday
    2: 1.1,   # Wednesday - peak
    3: 1.05,  # Thursday
    4: 1.0,   # Friday
    5: 0.7,   # Saturday - weekend dip
    6: 0.6    # Sunday - lowest
}

# Random noise parameters
NOISE_LEVEL = 0.15  # 15% random variation