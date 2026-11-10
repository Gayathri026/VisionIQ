import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.session import init_db
from backend.api import (
    routes_health,
    routes_analysis,
    routes_video,
    routes_camera,
    routes_reports,
    routes_rag,
    routes_evaluation
)

# Initialize Database Schema
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Storage directory for visual artifacts and reports
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")

# Register API Routers
app.include_router(routes_health.router, prefix=settings.API_PREFIX, tags=["Health"])
app.include_router(routes_analysis.router, prefix=settings.API_PREFIX, tags=["Analysis"])
app.include_router(routes_video.router, prefix=settings.API_PREFIX, tags=["Video"])
app.include_router(routes_camera.router, prefix=settings.API_PREFIX, tags=["Live Camera"])
app.include_router(routes_reports.router, prefix=settings.API_PREFIX, tags=["Reports"])
app.include_router(routes_rag.router, prefix=settings.API_PREFIX, tags=["RAG Knowledge"])
app.include_router(routes_evaluation.router, prefix=settings.API_PREFIX, tags=["Scientific Evaluation"])

@app.get("/")
def root():
    return {
        "framework": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_prefix": settings.API_PREFIX
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
