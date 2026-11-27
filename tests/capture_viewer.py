"""A real scrollable window for the OS parts (``windows``, ``macos``, ``linux``): a known tall image in a Tk canvas.

``python capture_viewer.py <title> <seed>`` says its own pid on stdout first (the window's owner), then shows
:func:`picture` ``(seed)`` one pixel a pixel (on Windows the process is made per-monitor DPI aware, so Windows does not
rescale it). Each wheel event scrolls :data:`NOTCH` pixels a notch -- Windows' and Tk 9's ``<MouseWheel>`` count 120 a
notch, Tk 8.6 on macOS one event a notch, X11's are ``<Button-4>``/``<Button-5>`` -- and the Down arrow :data:`ARROW`
pixels, both stopping at the end, as a document viewer does.
"""

import ctypes
import os
import sys

import numpy as np

WIDTH, HEIGHT, VIEW = 360, 1900, 300
NOTCH, ARROW = 40, 37
_WHEEL_DELTA = 120
_SHOWN_MS = 300  # after this the window is up, and its view is drawn again


def picture(seed: int) -> np.ndarray:
    """Distinct noise rows: any misplaced row shows."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def notches(delta: int) -> int:
    """The notches a ``<MouseWheel>`` ``delta`` scrolls, down when > 0: 120 a notch, else one an event."""
    if abs(delta) >= _WHEEL_DELTA:
        return -(delta // _WHEEL_DELTA)
    return (delta < 0) - (delta > 0)


def main(title: str, seed: int) -> None:
    sys.stdout.write(f'{os.getpid()}\n')  # the window's owner: on Windows a venv launcher's child
    sys.stdout.flush()
    import tkinter as tk  # noqa: PLC0415 -- only the window needs Tk; a runner without one collects this module

    if sys.platform == 'win32':
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    root = tk.Tk()
    root.title(title)
    root.attributes('-topmost', True)
    root.resizable(width=False, height=False)
    canvas = tk.Canvas(root, width=WIDTH, height=VIEW, highlightthickness=0, borderwidth=0, yscrollincrement=1)
    canvas.pack()
    header = f'P6 {WIDTH} {HEIGHT} 255 '.encode('ascii')
    photo = tk.PhotoImage(data=header + picture(seed).tobytes(), format='ppm')
    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.configure(scrollregion=(0, 0, WIDTH, HEIGHT))

    def scrolled(event: object, units: int) -> None:
        """Scroll ``units`` pixels, and say on stderr what came and where the view now starts (shown on a failure)."""
        canvas.yview_scroll(units, 'units')
        top = round(canvas.canvasy(0))
        sys.stderr.write(f'viewer: {event.type} {getattr(event, "delta", "")} {event.keysym}: {units:+d} -> {top}\n')
        sys.stderr.flush()

    root.bind_all('<MouseWheel>', lambda e: scrolled(e, NOTCH * notches(e.delta)))
    root.bind_all('<Button-4>', lambda e: scrolled(e, -NOTCH))
    root.bind_all('<Button-5>', lambda e: scrolled(e, NOTCH))
    root.bind_all('<Down>', lambda e: scrolled(e, ARROW))
    root.geometry('+200+200')
    root.focus_force()
    root.after(_SHOWN_MS, _redrawn, root, canvas)
    root.mainloop()


def _redrawn(root: object, canvas: object) -> None:
    """Draw the view again once the window is up: Tk 9 on macOS may leave a window blank until it is next drawn."""
    root.focus_force()
    canvas.yview_scroll(1, 'units')
    canvas.yview_scroll(-1, 'units')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
