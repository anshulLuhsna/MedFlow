"""
MedFlow FastAPI Application
Main application entry point
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from contextlib import asynccontextmanager
import time

from .config import get_settings
from .routes import health, predictions, preferences, hospitals, outbreaks, interactions

# Get settings
settings = get_settings()


# ============================================
# Lifespan Events (Startup/Shutdown)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📝 Environment: {settings.environment}")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print(f"🛑 Shutting down {settings.app_name}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RESTful API for medical resource allocation AI",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================
# CORS Configuration
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request Logging Middleware
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests with timing information
    """
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log request details
    print(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {duration:.2f}s"
    )

    return response


# ============================================
# Global Exception Handler
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions and return structured error response
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# Include Routers
# ============================================

# Health check routes (no auth required)
app.include_router(health.router)

# API routes (auth required)
app.include_router(predictions.router)
app.include_router(preferences.router)
app.include_router(hospitals.router)
app.include_router(outbreaks.router)
app.include_router(interactions.router)


# ============================================
# Root Endpoint
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": f"{settings.app_name} v{settings.app_version}",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.now().isoformat()
    }
