# MedFlow API Endpoint Reference

Quick reference for all API endpoints. See API_OVERVIEW.md for architecture details.

---

## Health Endpoints

### GET `/health`
**Purpose:** Basic health check
**Auth:** None
**Response:**
```json
{"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

### GET `/health/ml`
**Purpose:** Verify ML models loaded
**Auth:** None
**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_resources": ["ppe", "o2_cylinders", "ventilators", "medications", "beds"]
}
```

---

## Hospital Endpoints

### GET `/api/v1/hospitals`
**Purpose:** List all hospitals
**Auth:** Required
**Query Params:** `region` (optional)
**Example:**
```bash
curl -H "X-API-Key: key" "http://localhost:8000/api/v1/hospitals?region=North"
```

### GET `/api/v1/hospitals/{hospital_id}`
**Purpose:** Get hospital details
**Auth:** Required
**Path Param:** `hospital_id` (UUID)

### GET `/api/v1/hospitals/{hospital_id}/inventory`
**Purpose:** Get current inventory levels
**Auth:** Required
**Returns:** All resource types with quantities

### GET `/api/v1/hospitals/{hospital_id}/status`
**Purpose:** Complete status (inventory + predictions + risks)
**Auth:** Required
**ML Core Method:** `ml_core.get_hospital_status()`

---

## Prediction Endpoints

### POST `/api/v1/predict/demand` ⚠️
**Purpose:** Forecast demand using LSTM
**Auth:** Required
**Status:** Temporarily unavailable (feature engineering issue)
**Body:**
```json
{
  "hospital_id": "uuid",
  "resource_type": "ppe",
  "days_ahead": 14
}
```

### GET `/api/v1/shortages`
**Purpose:** Detect shortage risks using Random Forest
**Auth:** Required
**Query Params:** `resource_type` (optional), `limit` (1-100, optional)
**Example:**
```bash
curl -H "X-API-Key: key" \
  "http://localhost:8000/api/v1/shortages?resource_type=ventilators&limit=20"
```
**Response:**
```json
{
  "shortages": [...],
  "count": 12,
  "summary": {
    "by_risk_level": {"critical": 3, "high": 5, "medium": 4}
  }
}
```

### POST `/api/v1/optimize`
**Purpose:** Generate single optimal allocation strategy
**Auth:** Required
**Body:**
```json
{
  "resource_type": "ventilators",
  "shortage_hospital_ids": ["uuid1", "uuid2"],  // optional
  "limit": 50  // optional
}
```
**Response:**
```json
{
  "status": "optimal",
  "allocations": [...],
  "summary": {
    "total_transfers": 5,
    "total_cost": 2625.0,
    "hospitals_helped": 8,
    "shortage_reduction_pct": 85.5
  }
}
```

### POST `/api/v1/strategies`
**Purpose:** Generate multiple allocation strategies
**Auth:** Required
**Body:**
```json
{
  "resource_type": "ventilators",
  "n_strategies": 3,
  "limit": 50
}
```
**Response:** Array of 3 strategies (Cost-Efficient, Maximum Coverage, Balanced)

---

## Preference Learning Endpoints

### POST `/api/v1/preferences/score`
**Purpose:** Rank strategies by user preferences
**Auth:** Required
**Scoring:** Hybrid (40% RF + 30% LLM + 30% Vector DB)
**Body:**
```json
{
  "user_id": "user_123",
  "recommendations": [...],
  "past_interactions": [...]  // optional
}
```
**Response:**
```json
{
  "ranked_strategies": [
    {
      "strategy_name": "Cost-Efficient",
      "preference_score": 0.85,
      "llm_explanation": "This aligns with your cost-conscious approach..."
    }
  ]
}
```

### POST `/api/v1/preferences/update`
**Purpose:** Learn from user decision
**Auth:** Required
**Body:**
```json
{
  "user_id": "user_123",
  "interaction": {
    "selected_recommendation_index": 0,
    "recommendations": [...],
    "timestamp": "2025-11-05T10:00:00"
  }
}
```

---

## Outbreak Endpoints

### GET `/api/v1/outbreaks`
**Purpose:** List outbreaks with filters
**Auth:** Required
**Query Params:**
- `start_date`, `end_date` (ISO format)
- `region`, `event_type`, `severity`
- `limit` (1-100)

### GET `/api/v1/outbreaks/{outbreak_id}`
**Purpose:** Get outbreak details
**Auth:** Required

### GET `/api/v1/outbreaks/active`
**Purpose:** Get currently active outbreaks
**Auth:** Required
**Query Params:** `region` (optional)

### GET `/api/v1/outbreaks/impact/{outbreak_id}`
**Purpose:** Impact analysis (shortages during event)
**Auth:** Required
**Query Params:** `resource_type` (optional)

---

## Common Parameters

### Resource Types (Valid Values)
- `ppe` - Personal Protective Equipment
- `o2_cylinders` - Oxygen Cylinders
- `ventilators` - Ventilators
- `medications` - Medications
- `beds` - Hospital Beds

### Risk Levels
- `critical` - < 2 days until shortage
- `high` - 2-5 days
- `medium` - 5-10 days
- `low` - > 10 days

### Event Types
- `outbreak` - Disease outbreak
- `supply_disruption` - Supply chain issue

---

## Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request successful |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Missing API key |
| 404 | Not Found | Hospital ID doesn't exist |
| 500 | Server Error | ML model error |
| 503 | Service Unavailable | ML models not loaded |

---

## Quick Examples

### Detect Shortages → Generate Strategies → Rank by Preferences
```bash
# 1. Find shortages
curl -H "X-API-Key: key" http://localhost:8000/api/v1/shortages > shortages.json

# 2. Generate strategies
curl -X POST http://localhost:8000/api/v1/strategies \
  -H "X-API-Key: key" -H "Content-Type: application/json" \
  -d '{"resource_type": "ventilators", "n_strategies": 3}' > strategies.json

# 3. Rank by preferences
curl -X POST http://localhost:8000/api/v1/preferences/score \
  -H "X-API-Key: key" -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "recommendations": [/*strategies*/]}' > ranked.json
```
