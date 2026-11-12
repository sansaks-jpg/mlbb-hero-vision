"""MLBB Hero Vision — Offline ONNX hero recognition distilled from Gemini Embedding 2."""

from .student import HeroVision, ensure_model_files, is_student_ready

__all__ = ["HeroVision", "ensure_model_files", "is_student_ready"]
__version__ = "1.0.0"
