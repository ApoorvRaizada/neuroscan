"""Validate uploads by actually decoding them, not by trusting names or headers."""
import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings

ALLOWED_FORMATS = {"JPEG", "PNG"}


async def read_image(file: UploadFile, s: Settings) -> Image.Image:
    data = await file.read(s.max_upload_bytes + 1)  # never read more than limit + 1
    if len(data) > s.max_upload_bytes:
        raise HTTPException(413, f"File larger than {s.max_upload_bytes // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(data))  # lazy: reads the header only
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Not a readable image") from None
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(415, "Only JPEG and PNG are accepted")
    w, h = img.size
    if not (s.min_dim <= w <= s.max_dim and s.min_dim <= h <= s.max_dim):
        raise HTTPException(422, f"Image sides must be {s.min_dim} to {s.max_dim} px, got {w}x{h}")
    try:
        img.load()  # full decode; catches truncated or corrupt files
    except (OSError, SyntaxError, ValueError):
        raise HTTPException(400, "Corrupt image") from None
    return img