import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import init_db
from backend.app.api import routes_health, routes_analysis, routes_rag, routes_assistant, routes_experiments, routes_reports

# Initialize database tables
init_db()
logger.info("VisionIQ Database initialized successfully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration "
        "and Cost-Aware Query Routing. Suitable for academic evaluation and IEEE Access submission."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Storage directory
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")

# Register API Routers
app.include_router(routes_health.router, prefix=settings.API_PREFIX, tags=["Diagnostics & Health"])
app.include_router(routes_analysis.router, prefix=settings.API_PREFIX, tags=["Analysis & Reports"])
app.include_router(routes_rag.router, prefix=settings.API_PREFIX, tags=["RAG & Knowledge Base"])
app.include_router(routes_assistant.router, prefix=settings.API_PREFIX, tags=["In-App AI Assistant"])
app.include_router(routes_experiments.router, prefix=settings.API_PREFIX, tags=["Research Experiments"])
app.include_router(routes_reports.router, prefix=settings.API_PREFIX, tags=["PDF Reports & Records"])

@app.get("/", tags=["Root"])
def root():
    return {
        "framework": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "phase": "Phase 1: Foundation",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_PREFIX}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
