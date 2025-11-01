"""
Generate patient admission patterns
"""

from datetime import timedelta
import random
from .patterns import generate_realistic_value

def calculate_base_admissions(hospital_capacity: int) -> int:
    """Calculate base daily admissions based on hospital size

    IMPORTANT: For ML training, admissions must be:
    1. Realistic (hospitals have patients every day)
    2. Sufficient volume (not mostly zeros)
    3. Correlated with hospital size
    """
    # FIXED: 5-10% of capacity as daily admissions
    # Ensure minimum admissions for small hospitals (realistic floor)
    base = int(hospital_capacity * random.uniform(0.05, 0.10))

    # Minimum admissions even for smallest hospitals
    # Small hospital (50 beds) should have at least 3-5 patients/day
    minimum = max(3, int(hospital_capacity * 0.03))

    return max(base, minimum)

def generate_admission_severity() -> float:
    """Generate average severity score for the day (1-5 scale)"""
    # Normal distribution around 2.5, most patients are not critical
    severity = random.gauss(2.5, 0.8)
    return max(1.0, min(5.0, round(severity, 2)))

def calculate_icu_admissions(total_admissions: int, severity: float) -> int:
    """Calculate ICU admissions based on total and severity

    IMPORTANT: ICU admissions drive ventilator/O2 consumption
    Must have sufficient volume for ML training
    """
    # FIXED: Higher severity = more ICU admissions
    # Base rate: 10-20% (realistic for Indian hospitals)
    # Severity adjustment: ±5% based on patient acuity
    icu_rate = 0.10 + (severity - 2.5) * 0.05  # 10-20% ICU rate
    icu_rate = max(0.08, min(0.25, icu_rate))  # Increased minimum from 5% to 8%

    icu_admissions = int(total_admissions * icu_rate)

    # Ensure minimum ICU admissions for large patient volumes
    # Large hospitals should have at least 1-2 ICU patients/day
    if total_admissions >= 20 and icu_admissions == 0:
        icu_admissions = 1 if random.random() < 0.7 else 2

    return icu_admissions

def calculate_emergency_admissions(total_admissions: int) -> int:
    """Calculate emergency admissions"""
    # 30-50% are emergency admissions
    emergency_rate = random.uniform(0.30, 0.50)
    return int(total_admissions * emergency_rate)

def generate_admissions_data(
    hospitals: list,
    start_date,
    total_days: int,
    events: list,
    seasonal_factors: dict,
    weekly_pattern: dict
) -> list:
    """Generate admission data for all hospitals over time period"""
    
    admissions_data = []
    
    for hospital in hospitals:
        base_admissions = calculate_base_admissions(hospital["capacity_beds"])
        
        for day_idx in range(total_days):
            current_date = start_date + timedelta(days=day_idx)
            
            # Generate total admissions with all patterns
            total_admissions = generate_realistic_value(
                base_value=base_admissions,
                date=current_date,
                day_index=day_idx,
                total_days=total_days,
                region=hospital["region"],
                events=events,
                seasonal_factors=seasonal_factors,
                weekly_pattern=weekly_pattern
            )
            
            severity = generate_admission_severity()
            icu = calculate_icu_admissions(total_admissions, severity)
            emergency = calculate_emergency_admissions(total_admissions)
            
            admission_record = {
                "hospital_id": hospital["id"],
                "admission_date": current_date.date(),
                "total_admissions": total_admissions,
                "icu_admissions": icu,
                "emergency_admissions": emergency,
                "average_severity": severity
            }
            
            admissions_data.append(admission_record)
    
    return admissions_data