"""Strategy analysis page placeholder for the redesigned UI."""

from ui.components import panel, state_message


def render() -> None:
    panel("Strategy Analysis", "Future chart panels will wrap tyre degradation, gap evolution, and undercut simulation outputs.")
    state_message(
        "empty",
        "Pipeline Preserved",
        "The existing ML and simulation logic remains in the current app until integration is requested.",
    )

