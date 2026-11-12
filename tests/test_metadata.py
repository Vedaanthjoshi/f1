from ui.metadata import driver_profile, get_driver, get_track, missing_assets, resolve_asset


def test_driver_lookup_by_fastf1_code():
    driver = get_driver("ver")

    assert driver is not None
    assert driver["code"] == "VER"
    assert driver["team"] == "Red Bull Racing"


def test_unknown_driver_profile_falls_back_cleanly():
    profile = driver_profile("zzz")

    assert profile["driver"]["code"] == "ZZZ"
    assert profile["team"]["name"] == "Unknown Team"
    assert profile["driverPhotoPath"] is None


def test_track_lookup_accepts_fastf1_alias():
    track = get_track("Great Britain")

    assert track is not None
    assert track["raceName"] == "British Grand Prix"


def test_missing_assets_are_reported_without_crashing():
    missing = missing_assets()

    assert missing
    assert all(item.configured_path.startswith("assets/") for item in missing)


def test_asset_resolution_rejects_project_escape():
    assert resolve_asset("../outside.png") is None

