"""Pydantic models for the HTTP contract (see docs/PRD.md, section 5)."""
from typing import Literal

from pydantic import BaseModel


class Detection(BaseModel):
    type: Literal["pothole", "crack"]
    label_es: str
    model_class: str
    confidence: float
    bbox: list[int]
    area_pct: float


class Summary(BaseModel):
    damage_found: bool
    max_confidence: float | None


class Privacy(BaseModel):
    faces_blurred: int
    plates_blurred: int


class Timing(BaseModel):
    anonymize: int
    damage: int
    total: int


class AnalyzeResponse(BaseModel):
    image: str
    width: int
    height: int
    detections: list[Detection]
    summary: Summary
    privacy: Privacy
    models: dict[str, str]
    timing_ms: Timing


class Health(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
