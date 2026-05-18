# UI Iteration Notes

This redesign is isolated from `app.py` until integration is explicitly requested.

## Current Rule

- Do not change `app.py` for UI work unless the user approves it.
- Do not fetch images from the internet.
- Whenever a page needs imagery, ask the user for local image input first.
- The three user-provided references are for future visual direction:
  - Homepage background: `C:/Users/ambas/Downloads/homepage.png`
  - Cars page reference: `C:/Users/ambas/Downloads/Racing Game UI.jpg`
  - Map pages reference: `C:/Users/ambas/Downloads/mapUI.jpg`

## Added Structure

```text
ui_app.py
ui/
  theme.py
  components.py
  charts.py
  metadata.py
  pages/
    asset_library.py
    home.py
    race_setup.py
    driver_comparison.py
    track_profile.py
    strategy_analysis.py
assets/
  drivers/
  cars/
  tracks/
data/
  drivers.json
  teams.json
  cars.json
  tracks.json
```

## Iteration 1 Status

The visual shell now exists as a separate Streamlit entrypoint. It includes:

- Shared theme constants.
- Global dark motorsport CSS.
- Reusable hero, panel, stat, navigation, and state components.
- Separate page modules for the future homepage, race setup, driver comparison, track profile, and strategy analysis.

The existing backend pipeline remains untouched in `app.py`.

## Iteration 2 Status

The local asset and metadata system now exists. It includes:

- Local asset directories for user-provided driver, car, and track imagery.
- JSON metadata files for drivers, teams, cars, and tracks.
- `ui.metadata` helpers for loading metadata, resolving local asset paths, matching FastF1 driver codes, and matching race names.
- Fallback profiles for unknown drivers or missing metadata.
- Missing asset reporting in the separate `Asset Library` page.

Configured image paths are intentionally allowed to be missing. Missing images should prompt a request for user-provided local files, never an internet fetch.

## Iteration 3 Status

The collaborative homepage is now integrated into `app.py`.

It includes:

- A cinematic homepage hero using the user-provided local image copied to `assets/home/homepage.png`.
- NFS Unbound-inspired white, grey, black, and bright-red visual treatment.
- Main-page race setup controls for season and Grand Prix.
- Sidebar race setup controls for quick access.
- Event summary cards for season, Grand Prix, matchup, and circuit.
- Driver-versus-driver preview cards with metadata fallback states.
- Track preview using local track metadata.
- Existing FastF1 data loading, ML training, charts, and undercut simulation preserved below the homepage flow.

No internet images were fetched. Future driver, car, or track imagery still requires user-provided local files.

## Iteration 3 Refinement

The homepage image now behaves as the seamless page background instead of sitting inside a framed hero box. The main UI theme has shifted away from black/red dominance to:

- White and light grey as the primary surface.
- Black for contrast and typography.
- Bright red only for emphasis, buttons, and active racing accents.
- A lighter NFS-style sidebar rail.
- Light chart styling for the analysis section.

The original full-size homepage image remains in `assets/home/homepage.png`; the app uses a web-optimized derivative at `assets/home/homepage_hero.webp` for faster rendering.
