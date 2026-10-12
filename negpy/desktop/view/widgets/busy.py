from contextlib import contextmanager
from typing import Iterator

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractButton


@contextmanager
def busy_button(button: QAbstractButton, text: str | None = None) -> Iterator[None]:
    """Show a blocking GUI-thread action as busy on the button that started it.

    Never an override wait cursor: on macOS, QGuiApplication::setOverrideCursor can crash in
    CGImageCreate while it builds the cursor image, in a dialog or in the main window alike.
    """
    label = button.text()
    # The button's own flag, not isEnabled(), which also reads a disabled parent.
    was_enabled = not button.testAttribute(Qt.WidgetAttribute.WA_ForceDisabled)
    button.setEnabled(False)
    if text is not None:
        button.setText(text)
    button.repaint()
    try:
        yield
    finally:
        if text is not None:
            button.setText(label)
        button.setEnabled(was_enabled)
