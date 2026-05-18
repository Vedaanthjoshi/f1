"""Local asset helpers for UI rendering."""

from __future__ import annotations

import base64
from pathlib import Path

from ui.metadata import PROJECT_ROOT

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def asset_path(relative_path: str) -> Path:
    """Return an absolute path for a project-local asset."""
    return (PROJECT_ROOT / relative_path).resolve()


def image_data_uri(relative_path: str) -> str | None:
    """Return a base64 data URI for a project-local image."""
    path = asset_path(relative_path)
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    if not path.exists():
        return None

    mime_type = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

