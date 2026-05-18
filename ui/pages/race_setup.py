"""Race setup page placeholder for the redesigned UI."""

from ui.components import panel, state_message


def render() -> None:
    panel("Race Setup", "This page will host season, Grand Prix, Driver A, and Driver B setup modules.")
    state_message(
        "info",
        "Image Input Required Later",
        "When this page needs driver, car, or track imagery, ask the user for local files before adding any assets.",
    )

