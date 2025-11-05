# MedFlow API Overview

**Version:** 1.0
**Status:** Production Ready (Phase 4 Complete)

## Quick Info

- **Framework:** FastAPI 0.104.1
- **Base URL:** `http://localhost:8000`
- **Docs:** `http://localhost:8000/docs` (Swagger UI)
- **Auth:** API Key header (`X-API-Key`) or JWT Bearer token
- **Total Endpoints:** 16

## Architecture

```
┌─────────────────────────────────────────────┐
│            FastAPI Backend                   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Health (2)    │  Hospitals (4)      │   │
│  │  Predictions (4) │ Preferences (2)   │   │
│  │  Outbreaks (4)                       │   │
│  └──────────────────────────────────────┘   │
│                   ↓                          │
│  ┌──────────────────────────────────────┐   │
│  │         ML Core Integration          │   │
│  │  • LSTM Forecasting                  │   │
│  │  • Random Forest Shortage Detection  │   │
│  │  • Linear Programming Optimization   │   │
│  │  • Hybrid Preference Learning        │   │
│  └──────────────────────────────────────┘   │
│                   ↓                          │
│  ┌──────────────────────────────────────┐   │
│  │      Supabase (PostgreSQL)           │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Endpoint Categories

### Health (No Auth Required)
- `GET /health` - Basic health check
- `GET /health/ml` - ML models status

### Hospitals (Auth Required)
- `GET /api/v1/hospitals` - List all hospitals
- `GET /api/v1/hospitals/{id}` - Hospital details
- `GET /api/v1/hospitals/{id}/inventory` - Current inventory
- `GET /api/v1/hospitals/{id}/status` - Complete status with predictions

### Predictions (Auth Required)
- `POST /api/v1/predict/demand` - LSTM demand forecasting ⚠️
- `GET /api/v1/shortages` - Random Forest shortage detection
- `POST /api/v1/optimize` - Single allocation strategy
- `POST /api/v1/strategies` - Multiple allocation strategies

### Preferences (Auth Required)
- `POST /api/v1/preferences/score` - Rank strategies by user preferences
- `POST /api/v1/preferences/update` - Learn from user feedback

### Outbreaks (Auth Required)
- `GET /api/v1/outbreaks` - List outbreaks with filters
- `GET /api/v1/outbreaks/{id}` - Outbreak details
- `GET /api/v1/outbreaks/active` - Currently active outbreaks
- `GET /api/v1/outbreaks/impact/{id}` - Impact analysis

## Quick Start

```bash
# 1. Start server
cd backend
uvicorn app.main:app --reload --port 8000

# 2. Check health
curl http://localhost:8000/health

# 3. Test with Postman
# Import: backend/MedFlow_API.postman_collection.json

# 4. Make your first API call
curl -H "X-API-Key: your_key" http://localhost:8000/api/v1/hospitals
```

## Known Issues

⚠️ **Demand Prediction Endpoint** - Temporarily unavailable due to feature engineering mismatch (4 features vs 17 expected). Fix in progress.

✅ **All Other Endpoints** - Fully functional and tested (22/22 tests passing)

## For Phase 5 (Agentic Layer)

The API is designed for LangGraph agent integration:

1. **Data Analyst Agent** → `/hospitals`, `/shortages`, `/outbreaks/active`
2. **Forecasting Agent** → `/predict/demand`
3. **Optimization Agent** → `/strategies`
4. **Preference Agent** → `/preferences/score`, `/preferences/update`
5. **Reasoning Agent** → Combines outputs, generates explanations

See `AGENT_WORKFLOW.md` for detailed integration guide.
