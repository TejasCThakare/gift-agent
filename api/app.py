"""
api/app.py

FastAPI application factory.

Mounts:
  POST /run              — start workflow
  POST /review/{tid}     — submit review action
  GET  /status/{tid}     — poll current state
  GET  /runs             — list all runs
  GET  /                 — serve review UI (ui/index.html)
  GET  /health           — health check

CORS is configured to allow all origins for development.
In production, restrict to your frontend domain.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()


import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes.run import router as run_router
from api.routes.review import router as review_router
from api.routes.status import router as status_router
from services.llm.factory import get_provider_info
from storage.artifact_store import ARTIFACTS_DIR

logger = logging.getLogger("api.app")

# Ensure artifacts directory exists at startup
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

UI_DIR = Path(__file__).parent.parent / "ui"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Gift Agent API",
        description=(
            "Hyper-personalised AI gift recommendation system. "
            "Analyzes LinkedIn profiles to recommend relevant, purchasable gifts "
            "with full evidence grounding and human-in-the-loop review."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(run_router, tags=["Workflow"])
    app.include_router(review_router, tags=["Review"])
    app.include_router(status_router, tags=["Status"])

    # ── Static files (artifacts downloads) ───────────────────────────────
    if ARTIFACTS_DIR.exists():
        app.mount(
            "/artifacts",
            StaticFiles(directory=str(ARTIFACTS_DIR)),
            name="artifacts",
        )

    # ── UI serving ────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        """Serve the single-file review UI."""
        ui_path = UI_DIR / "index.html"
        if ui_path.exists():
            return FileResponse(str(ui_path))
        return JSONResponse(
            {"message": "Gift Agent API is running. UI not found at ui/index.html."},
            status_code=200,
        )

    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        """Check API health and LLM provider availability."""
        provider_info = get_provider_info()
        any_available = any(provider_info.values())

        return {
            "status": "healthy" if any_available else "degraded",
            "llm_providers": provider_info,
            "artifacts_dir": str(ARTIFACTS_DIR),
            "artifacts_dir_exists": ARTIFACTS_DIR.exists(),
        }

    # ── Startup event ─────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        logger.info("Gift Agent API starting up")
        provider_info = get_provider_info()
        for name, available in provider_info.items():
            status = "✓ available" if available else "✗ not available"
            logger.info("  LLM provider %s: %s", name, status)

        any_available = any(provider_info.values())
        if not any_available:
            logger.warning(
                "WARNING: No LLM provider is available. "
                "Start Ollama (ollama serve) or set GROQ_API_KEY / GEMINI_API_KEY."
            )

    return app


# Module-level app instance for uvicorn
app = create_app()
