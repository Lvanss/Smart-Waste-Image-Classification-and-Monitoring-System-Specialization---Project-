from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.model.predictor import WastePredictor


app = FastAPI(
    title="Smart Waste Classification API",
    description="Classifies waste images into six TrashNet categories.",
    version="1.0.0",
)

# Load the trained model once when the API starts.
predictor = WastePredictor()


@app.get("/")
def home():
    return {
        "message": "Smart Waste Classification API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "BaselineCNN",
        "device": str(predictor.device),
        "classes": [
            "cardboard",
            "glass",
            "metal",
            "paper",
            "plastic",
            "trash",
        ],
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload an image.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        with Image.open(BytesIO(contents)) as image:
            result = predictor.predict(image)

        return {
            "filename": file.filename,
            **result,
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a valid image.",
        )

    finally:
        await file.close()