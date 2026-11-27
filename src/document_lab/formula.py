"""An equation crop's pixels to LaTeX, in-process: the ``formula`` extra's model, loaded only when asked.

``document-lab paper`` / ``pages`` with ``--formulas latex`` write each equation crop's LaTeX beside it (the crop is
kept: the picture is the evidence, the LaTeX a model's reading of it). The model is TexTeller (`MODEL`, pinned at
`REVISION`; code and weights Apache-2.0): a ViT encoder and a RoBERTa decoder that transformers loads as a
VisionEncoderDecoderModel, run by torch on a CUDA GPU when one answers, else on the CPU -- no server, no ONNX
runtime, no image library. Its weights download once into the Hugging Face cache (``HF_HUB_CACHE``, by default
``~/.cache/huggingface/hub``); the evaluation that chose it is decision D85 of the document-lab-pdf-reader task.

torch and transformers are imported inside :func:`transcriber`, so the base package -- and this module -- import
without them; asking for formulas without the extra, or without weights the cache holds or can fetch, raises
:class:`FormulaUnavailable` by name. Nothing falls back to no LaTeX when LaTeX was asked for.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = ['MODEL', 'REVISION', 'UNREAD', 'FormulaUnavailable', 'Formulas', 'Transcribe', 'beside', 'transcriber']

#: What an equation crop is written as beside its image: ``latex``.
Formulas = Literal['latex']
#: An equation crop's RGB pixels (rows, columns, 3; uint8) to its LaTeX; None where the model gives no reading: a
#: blank crop, or a reading that never ends (it repeats until the token budget runs out, measured on 2% of crops).
Transcribe = Callable[[np.ndarray], str | None]
#: What stands beside a crop the model gave no reading of: never silently nothing.
UNREAD: Final = '<!-- formulas: no LaTeX read from this crop (blank, or the reading never ended) -->'

MODEL: Final = 'OleehyO/TexTeller'
REVISION: Final = '7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea'  # 2024-06-22, TexTeller 3

# TexTeller's own inference transform (texteller/constants.py, utils/image.py): grayscale, the background border
# trimmed, scaled to fit a 448-pixel square, normalized by its training images' mean and spread, padded at the right
# and bottom.
_SIZE: Final = 448
_MEAN: Final = 0.9545467
_STD: Final = 0.15394445
_BORDER: Final = 15  # gray levels from the background color a pixel must differ by to be ink
_MAX_TOKENS: Final = 1024
_GRAY: Final = np.array([0.2989, 0.587, 0.114])  # ITU-R 601-2, as torchvision's Grayscale


class FormulaUnavailable(RuntimeError):
    """Formulas were asked for and the model cannot run here: the extra or its weights are missing."""


def transcriber(formulas: Formulas, *, device: str | None = None) -> Transcribe:
    """The model, loaded once: each call reads one equation crop's pixels as LaTeX (:data:`Transcribe`).

    ``device``: a torch device; by default CUDA when torch sees a GPU, else the CPU. Raises
    :class:`FormulaUnavailable` naming what is missing.
    """
    if formulas != 'latex':
        msg = f'formulas {formulas!r}: the one form is latex'
        raise ValueError(msg)
    try:
        import torch  # noqa: PLC0415 -- the extra is imported only when formulas are asked for
        from transformers import RobertaTokenizerFast, VisionEncoderDecoderModel  # noqa: PLC0415
    except ImportError as missing:
        msg = f"--formulas latex needs the 'formula' extra (torch, transformers): document-lab[formula]: {missing}"
        raise FormulaUnavailable(msg) from missing
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL, revision=REVISION)
    except OSError as missing:
        msg = (
            f'--formulas latex needs the weights of {MODEL} at {REVISION[:12]}: neither in the Hugging Face cache '
            f'nor downloadable ({missing})'
        )
        raise FormulaUnavailable(msg) from missing
    model = model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu')).eval()

    def transcribe(pixels: np.ndarray) -> str | None:
        square = _square(pixels)
        if square is None:
            return None
        with torch.inference_mode():
            ids = model.generate(
                square[None, None].to(model.device),
                max_new_tokens=_MAX_TOKENS,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
        if ids.shape[1] > _MAX_TOKENS and ids[0, -1] != tokenizer.eos_token_id:
            return None  # the budget ran out mid-reading: a loop, not a formula
        return tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _square(pixels: np.ndarray) -> torch.Tensor | None:
        gray = pixels[..., :3].astype(np.float64) @ _GRAY
        corners = (gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1])
        background = Counter(np.round(corners)).most_common(1)[0][0]
        rows, cols = np.nonzero(np.abs(gray - background) > _BORDER)
        if not len(rows):
            return None
        ink = torch.from_numpy(gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] / 255)
        height, width = ink.shape
        scale = min((_SIZE - 1) / min(height, width), _SIZE / max(height, width))
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        fitted = torch.nn.functional.interpolate(ink[None, None], size=size, mode='bicubic', antialias=True)[0, 0]
        square = torch.zeros((_SIZE, _SIZE), dtype=torch.float32)
        square[: size[0], : size[1]] = (fitted.clamp(0, 1) - _MEAN) / _STD
        return square

    return transcribe


def beside(latex: str | None, number: str | None) -> str:
    r"""What is written beside an equation crop: ``latex`` as markdown display blocks; :data:`UNREAD` for None.

    The model's own display delimiters are dropped -- a reading of several displays (``\] \[`` inside it) is one
    block each -- and the equation's ``number`` (``3`` of ``(3)``) where the model set it after a formula: the crop
    shows it, the label already names it. Any other number is the formula's own.
    """
    if latex is None:
        return UNREAD
    body = re.sub(r'^\s*(?:\\\[|\$\$)\s*|\s*(?:\\\]|\$\$)\s*$', '', latex.strip())
    return '\n\n'.join(f'$$\n{_unnumbered(d, number)}\n$$' for d in re.split(r'\s*\\\]\s*\\\[\s*', body))


#: Spacing a model sets between a formula and its number.
_SPACE: Final = r'\\q?quad|\\[,;:!]|~|\\hskip\s*-?[\d.]+\s*[a-z]{2}|\\hspace\*?\{[^}]*\}'


def _unnumbered(display: str, number: str | None) -> str:
    r"""``display`` without the equation's own ``number`` set after it (``\tag{3}``, ``\qquad (3)``)."""
    if number is not None:
        own = re.escape(number)
        display = re.sub(rf'\s*\\tag\*?\{{\s*{own}\s*\}}\s*$', '', display)
        display = re.sub(rf'(?:\s*(?:{_SPACE}))*\s*(?:\\left)?\(\s*{own}\s*(?:\\right)?\)\s*$', '', display)
    return display.strip()
