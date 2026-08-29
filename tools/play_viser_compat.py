"""Launch mjlab play with a compatibility fix for fixed-range Viser sliders."""

from viser._gui_api import GuiApi


_add_slider = GuiApi.add_slider


def _add_slider_clamped(self, label, *, min, max, step, initial_value, **kwargs):
    # mjlab creates dashboard controls for fixed command axes whose configured
    # maximum is zero. Viser requires the initial value to be within its GUI
    # slider bounds. Clamp only the displayed control; simulation ranges stay
    # unchanged.
    initial_value = min if initial_value < min else max if initial_value > max else initial_value
    return _add_slider(
        self,
        label,
        min=min,
        max=max,
        step=step,
        initial_value=initial_value,
        **kwargs,
    )


GuiApi.add_slider = _add_slider_clamped

from mjlab.scripts.play import main


if __name__ == "__main__":
    main()
