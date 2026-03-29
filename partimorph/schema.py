from typing import TypedDict


class CircleData(TypedDict):
    x: float
    y: float
    r: float


class EllipseData(TypedDict):
    major: float
    minor: float
    x: float
    y: float
    angle: float
    w: float
    h: float
    bbox: list[list[float]]


class RoundnessResult(TypedDict):
    val: float


class CircularityResult(TypedDict):
    val: float


class SphericityResult(TypedDict):
    val: float
    inscribed: CircleData
    enclosing: CircleData


class AspectRatioResult(TypedDict):
    val: float
    ellipse: EllipseData


class AnalysisResult(TypedDict, total=False):
    roundness: RoundnessResult | None
    circularity: CircularityResult | None
    sphericity: SphericityResult | None
    aspect_ratio: AspectRatioResult | None
