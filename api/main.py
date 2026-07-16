"""FastAPI entry point for the fraud investigation platform."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings
from api.schemas import (
    HealthResponse,
    InvestigationRequest,
    PredictionResponse,
    RawTransactionBatch,
    SimilarTransactionsRequest,
)
from api.services.investigation_service import InvestigationService
from api.services.metrics_service import MetricsService
from api.services.model_service import ModelService


API_VERSION = "1.0.0"
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


@lru_cache(maxsize=1)
def model_service() -> ModelService:
    return ModelService(settings.project_root, settings.max_prediction_records)


@lru_cache(maxsize=1)
def metrics_service() -> MetricsService:
    return MetricsService(settings.project_root)


@lru_cache(maxsize=1)
def investigation_service() -> InvestigationService:
    return InvestigationService(settings.project_root)


def required_artifacts(project_root: Path) -> list[Path]:
    return [
        project_root / "models" / "trained" / "champion_manifest.json",
        project_root / "models" / "preprocessing" / "selected_features.pkl",
        project_root / "models" / "investigation" / "tfidf_vectorizer.joblib",
        project_root / "models" / "investigation" / "validation_case_matrix.npz",
        project_root / "models" / "investigation" / "validation_case_library.parquet",
        project_root / "results" / "investigation" / "assistant_summaries.parquet",
    ]


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


@router.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(payload: RawTransactionBatch) -> dict:
    try:
        return model_service().predict_records(payload.transactions)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/investigations", tags=["Investigation"])
def investigations() -> dict:
    try:
        return {"transactions": investigation_service().list_demo_cases()}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/similar-transactions", tags=["Investigation"])
def similar_transactions(payload: SimilarTransactionsRequest) -> dict:
    try:
        return {
            "query_transaction_id": payload.transaction_id,
            "neighbors": investigation_service().similar_cases(
                payload.transaction_id, payload.top_n
            ),
            "warning": (
                "Similarity is descriptive evidence and is not a calibrated fraud probability."
            ),
        }
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/investigate", tags=["Investigation"])
def investigate(payload: InvestigationRequest) -> dict:
    try:
        return investigation_service().investigate(payload.transaction_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


app.include_router(router)

