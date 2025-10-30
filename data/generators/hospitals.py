"""
Generate hospital network
"""

import uuid
import random
import numpy as np
from typing import List, Dict

def distribute_hospitals_by_region(num_hospitals: int, regions: List[str]) -> Dict[str, int]:
    """Distribute hospitals across regions with some imbalance"""
    # Not perfectly equal - some regions have more hospitals
    weights = np.random.dirichlet(np.ones(len(regions)) * 2)
    distribution = {
        region: max(5, int(num_hospitals * weight))  # At least 5 per region
        for region, weight in zip(regions, weights)
    }
    
    # Adjust to match exact total
    total = sum(distribution.values())
    diff = num_hospitals - total
    if diff != 0:
        # Add/subtract from largest region
        largest = max(distribution, key=distribution.get)
        distribution[largest] += diff
    
    return distribution

def generate_hospital_name(region: str, index: int) -> str:
    """Generate realistic hospital names"""
    prefixes = ["General", "Memorial", "Regional", "Community", "Central", "City"]
    suffixes = ["Hospital", "Medical Center", "Health System", "Care Center"]
    
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    return f"{region} {prefix} {suffix} #{index}"

def assign_hospital_size(size_distribution: Dict) -> tuple:
    """Randomly assign hospital size based on distribution"""
    sizes = list(size_distribution.keys())
    weights = [size_distribution[s]["ratio"] for s in sizes]
    
    size = random.choices(sizes, weights=weights)[0]
    beds_range = size_distribution[size]["beds_range"]
    beds = random.randint(beds_range[0], beds_range[1])
    
    return size, beds

def generate_coordinates(region: str) -> tuple:
    """Generate realistic lat/long for an Indian state/region"""
    # Approximate centers for selected Indian states/regions
    region_centers = {
        "Maharashtra": (19.7515, 75.7139),      # Near Aurangabad
        "Karnataka": (15.3173, 75.7139),        # Near central KA
        "Tamil Nadu": (11.1271, 78.6569),       # Near central TN
        "Kerala": (10.8505, 76.2711),           # Near Palakkad
        "Telangana": (18.1124, 79.0193),        # Near Warangal
        "Andhra Pradesh": (15.9129, 79.7400),   # Near Kurnool
        "Delhi NCR": (28.6139, 77.2090),        # New Delhi
        "Uttar Pradesh": (26.8467, 80.9462),    # Near Lucknow
        "Rajasthan": (26.9124, 75.7873),        # Near Jaipur
        "Gujarat": (22.2587, 71.1924),          # Near Saurashtra
        "West Bengal": (22.9868, 87.8550),      # Near Durgapur
        "Bihar": (25.0961, 85.3131),            # Near Patna
        "Punjab": (31.1471, 75.3412),           # Near Jalandhar
        "Haryana": (29.0588, 76.0856),          # Near Rohtak
        "Madhya Pradesh": (22.9734, 78.6569),   # Near Sagar
        "Assam": (26.2006, 92.9376),            # Near Nagaon
    }
    
    center = region_centers.get(region, (22.3511, 78.6677))  # Default: near India's centroid
    # Add some random variation
    lat = center[0] + random.uniform(-2, 2)
    lon = center[1] + random.uniform(-2, 2)
    
    return round(lat, 6), round(lon, 6)

def generate_hospitals(
    num_hospitals: int,
    regions: List[str],
    size_distribution: Dict,
    specializations: List[str]
) -> List[Dict]:
    """Generate complete hospital network"""
    
    hospitals = []
    distribution = distribute_hospitals_by_region(num_hospitals, regions)
    
    for region in regions:
        count = distribution[region]
        for i in range(count):
            size, beds = assign_hospital_size(size_distribution)
            lat, lon = generate_coordinates(region)
            
            hospital = {
                "id": str(uuid.uuid4()),
                "name": generate_hospital_name(region, i + 1),
                "region": region,
                "latitude": lat,
                "longitude": lon,
                "capacity_beds": beds,
                "hospital_type": random.choice(specializations),
                "size": size  # For internal use, not in DB
            }
            
            hospitals.append(hospital)
    
    return hospitals