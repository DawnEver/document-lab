"""A real scrollable window for the ``desktop`` part: a known tall image in a Tk canvas, run as its own process.

``python viewer.py <title> <seed>`` shows :func:`picture` ``(seed)`` at physical pixels (the process is made
per-monitor DPI aware, so Windows does not rescale it). The wheel scrolls :data:`NOTCH` pixels per notch and the Down
arrow :data:`ARROW` pixels, both stopping at the end, as a document viewer does.
"""

import ctypes
import sys
import tkinter as tk

import numpy as np

WIDTH, HEIGHT, VIEW = 360, 1900, 300
NOTCH, ARROW = 40, 37


def picture(seed: int) -> np.ndarray:
    """Distinct noise rows: any misplaced row shows."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def main(title: str, seed: int) -> None:
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
    root.bind_all('<MouseWheel>', lambda e: canvas.yview_scroll(-NOTCH * (e.delta // 120), 'units'))
    root.bind_all('<Down>', lambda _: canvas.yview_scroll(ARROW, 'units'))
    root.geometry('+200+200')
    root.focus_force()
    root.mainloop()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
