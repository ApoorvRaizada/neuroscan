from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.schemas import HealthResponse, PredictionResponse
from app.security import require_api_key
from app.validation import read_image
from model.inference import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.predictor = Predictor(settings.model_path)  # loaded ONCE at startup
    yield


app = FastAPI(title="NeuroScan", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    img = await read_image(file, settings)
    return await run_in_threadpool(app.state.predictor.predict, img)