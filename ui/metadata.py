"""Metadata and local asset resolution for the redesigned UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class AssetCheck:
    """A missing local asset found in metadata."""

    kind: str
    identifier: str
    configured_path: str


def _read_json(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a JSON list.")
    return data


@lru_cache(maxsize=1)
def drivers() -> list[dict[str, Any]]:
    return _read_json("drivers.json")


@lru_cache(maxsize=1)
def teams() -> list[dict[str, Any]]:
    return _read_json("teams.json")


@lru_cache(maxsize=1)
def cars() -> list[dict[str, Any]]:
    return _read_json("cars.json")


@lru_cache(maxsize=1)
def tracks() -> list[dict[str, Any]]:
    return _read_json("tracks.json")


def clear_metadata_cache() -> None:
    """Clear cached metadata after JSON files are edited."""
    drivers.cache_clear()
    teams.cache_clear()
    cars.cache_clear()
    tracks.cache_clear()


def get_driver(code: str | None) -> dict[str, Any] | None:
    if not code:
        return None
    normalized = code.upper()
    return next((driver for driver in drivers() if driver.get("code", "").upper() == normalized), None)


def get_team(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    normalized = name.casefold()
    return next((team for team in teams() if team.get("name", "").casefold() == normalized), None)


def get_car(car_id: str | None) -> dict[str, Any] | None:
    if not car_id:
        return None
    normalized = car_id.casefold()
    return next((car for car in cars() if car.get("id", "").casefold() == normalized), None)


def get_track(race_name: str | None) -> dict[str, Any] | None:
    if not race_name:
        return None
    normalized = race_name.casefold()
    for track in tracks():
        names = [track.get("raceName", ""), *track.get("fastf1Names", [])]
        if any(str(name).casefold() == normalized for name in names):
            return track
    return None


def resolve_asset(configured_path: str | None) -> Path | None:
    """Return an absolute local asset path if it exists inside the project."""
    if not configured_path:
        return None

    candidate = (PROJECT_ROOT / configured_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    return candidate if candidate.exists() else None


def resolve_driver_photo(code: str, configured_path: str | None = None) -> Path | None:
    """Find driver photo across supported image extensions."""
    if configured_path:
        p = resolve_asset(configured_path)
        if p and p.exists():
            return p
        # Check if configured path exists with a different common extension
        base_name = Path(configured_path).stem
        parent_dir = Path(configured_path).parent
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
            candidate = (PROJECT_ROOT / parent_dir / f"{base_name}{ext}").resolve()
            if candidate.exists():
                return candidate

    # Try driver code directly in assets/drivers/
    normalized_code = code.upper()
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"]:
        candidate = (ASSETS_DIR / "drivers" / f"{normalized_code}{ext}").resolve()
        if candidate.exists():
            return candidate

    return None


def resolve_car_image(
    driver_code: str | None = None,
    car_id: str | None = None,
    team_name: str | None = None,
    configured_path: str | None = None,
) -> Path | None:
    """Find car image across supported image extensions, car ID, team name, or driver code."""
    candidates_to_try: list[str] = []
    if configured_path:
        candidates_to_try.append(configured_path)
    if car_id:
        candidates_to_try.append(f"assets/cars/{car_id}")
    if team_name:
        candidates_to_try.extend([
            f"assets/cars/{team_name}",
            f"assets/cars/{team_name.replace(' ', '')}",
            f"assets/cars/{team_name.replace(' ', '_')}",
            f"assets/cars/{team_name.replace(' Racing', '')}",
            f"assets/cars/{team_name.replace(' Racing', '').replace(' ', '')}",
            f"assets/cars/{team_name.replace('Kick ', '')}",
            f"assets/cars/{team_name.replace('Kick ', '').replace(' ', '')}",
            f"assets/cars/{team_name.lower().replace(' ', '')}",
            f"assets/cars/{team_name.lower().replace(' ', '_')}",
        ])
    if driver_code:
        candidates_to_try.extend([
            f"assets/cars/{driver_code.upper()}",
            f"assets/cars/{driver_code.lower()}",
        ])

    extensions = [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP", ""]

    for base in candidates_to_try:
        for ext in extensions:
            test_path = f"{base}{ext}" if not base.endswith(ext) else base
            p = resolve_asset(test_path)
            if p and p.exists():
                return p

    # Fallback scan directory case-insensitively
    cars_dir = ASSETS_DIR / "cars"
    if cars_dir.exists():
        existing_files = list(cars_dir.iterdir())
        match_keys = []
        if team_name:
            clean_team = "".join(c for c in team_name.lower() if c.isalnum())
            match_keys.extend([clean_team, clean_team.replace("racing", ""), clean_team.replace("kick", "")])
        if car_id:
            match_keys.append("".join(c for c in car_id.lower() if c.isalnum()))
        if driver_code:
            match_keys.append(driver_code.lower())

        for file_path in existing_files:
            file_clean = "".join(c for c in file_path.stem.lower() if c.isalnum())
            for key in match_keys:
                if key and (key == file_clean or key in file_clean or file_clean in key):
                    return file_path

    return None


def driver_profile(code: str) -> dict[str, Any]:
    """Return driver, team, car, and resolved asset paths with fallbacks."""
    driver = get_driver(code) or {
        "code": code.upper(),
        "name": code.upper(),
        "number": None,
        "team": "Unknown Team",
        "country": "Unknown",
        "photo": None,
        "carId": None,
    }
    team = get_team(driver.get("team")) or {
        "name": driver.get("team", "Unknown Team"),
        "shortName": driver.get("team", "Unknown Team"),
        "accentColor": "#E10600",
        "secondaryColor": "#FFFFFF",
    }
    car = get_car(driver.get("carId")) or {
        "id": driver.get("carId"),
        "team": team.get("name"),
        "name": "Unknown Car",
        "engine": "Unknown",
        "image": None,
    }
    return {
        "driver": driver,
        "team": team,
        "car": car,
        "driverPhotoPath": resolve_driver_photo(code, driver.get("photo")),
        "carImagePath": resolve_car_image(code, car.get("id"), team.get("name"), car.get("image")),
    }


def missing_assets() -> list[AssetCheck]:
    """List configured image paths that do not exist yet."""
    missing: list[AssetCheck] = []

    for driver in drivers():
        photo = driver.get("photo")
        if photo and resolve_asset(photo) is None:
            missing.append(AssetCheck("driver", driver.get("code", "unknown"), photo))

    for car in cars():
        image = car.get("image")
        if image and resolve_asset(image) is None:
            missing.append(AssetCheck("car", car.get("id", "unknown"), image))

    for track in tracks():
        photo = track.get("photo")
        if photo and resolve_asset(photo) is None:
            missing.append(AssetCheck("track", track.get("raceName", "unknown"), photo))

    return missing

