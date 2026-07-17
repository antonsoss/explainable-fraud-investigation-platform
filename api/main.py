"""FastAPI entry point for the fraud investigation platform."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import Settings
from api.schemas import (
    HealthResponse,
    PredictionResponse,
    RawTransactionBatch,
)
from api.services.metrics_service import MetricsService
from api.services.model_service import ModelService
from api.services.xai_service import XAIService


API_VERSION = "1.1.0"
settings = Settings.from_environment()

app = FastAPI(
    title="Fraud Investigation API",
    version=API_VERSION,
    description=(
        "Read-only access to frozen evaluation artifacts plus raw-transaction "
        "scoring with the champion fraud model."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
router = APIRouter(prefix="/api/v1")
app.mount(
    "/artifacts/xai",
    StaticFiles(
        directory=settings.project_root / "results" / "xai" / "figures",
        check_dir=False,
    ),
    name="xai-artifacts",
)


@lru_cache(maxsize=1)
def model_service() -> ModelService:
    return ModelService(settings.project_root, settings.max_prediction_records)


@lru_cache(maxsize=1)
def metrics_service() -> MetricsService:
    return MetricsService(settings.project_root)


@lru_cache(maxsize=1)
def xai_service() -> XAIService:
    return XAIService(settings.project_root)


def required_artifacts(project_root: Path) -> list[Path]:
    xai_figure_names = [
        "shap_global_bar.png",
        "shap_global_beeswarm.png",
        "shap_vs_native_importance.png",
        "shap_dependence_TransactionDT.png",
        "shap_dependence_C13.png",
        "shap_dependence_C1.png",
        "shap_waterfall_3519397.png",
        "shap_waterfall_3524909.png",
        "shap_waterfall_3541077.png",
        "shap_waterfall_3551357.png",
        "lime_3519397.png",
        "lime_3524909.png",
        "lime_3541077.png",
        "lime_3551357.png",
    ]
    required = [
        project_root / "models" / "trained" / "champion_manifest.json",
        project_root / "models" / "preprocessing" / "selected_features.pkl",
        project_root / "results" / "xai" / "xai_metadata.json",
        project_root / "results" / "xai" / "shap_global_importance.csv",
        project_root / "results" / "xai" / "local_shap_values.parquet",
        project_root / "results" / "xai" / "local_lime_values.parquet",
    ]
    required.extend(
        project_root / "results" / "xai" / "figures" / filename
        for filename in xai_figure_names
    )
    return required


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "Fraud Investigation API",
        "health": "/api/v1/health",
        "documentation": "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["Operations"])
def health() -> HealthResponse:
    missing = [
        str(path.relative_to(settings.project_root))
        for path in required_artifacts(settings.project_root)
        if not path.exists()
    ]
    return HealthResponse(
        status="ready" if not missing else "degraded",
        service="fraud-investigation-api",
        version=API_VERSION,
        artifacts_ready=not missing,
        missing_artifacts=missing,
    )


@router.get("/model", tags=["Model"])
def model() -> dict:
    try:
        return metrics_service().model_info()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/metrics", tags=["Model"])
def metrics() -> dict:
    try:
        return metrics_service().metrics()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/xai", tags=["Model Explanations"])
def xai(top_n: int = 20) -> dict:
    try:
        return xai_service().global_summary(top_n=top_n)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/xai/{transaction_id}", tags=["Model Explanations"])
def local_xai(transaction_id: int, top_n: int = 10) -> dict:
    try:
        return xai_service().local_explanation(transaction_id, top_n=top_n)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(payload: RawTransactionBatch) -> dict:
    try:
        return model_service().predict_records(payload.transactions)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


app.include_router(router)
