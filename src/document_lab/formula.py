"""An equation crop's pixels to LaTeX, in-process: the ``formula`` extra's model, loaded only when asked.

``document-lab paper`` / ``pages`` with ``--formulas latex`` write each equation crop's LaTeX beside it (the crop is
kept: the picture is the evidence, the LaTeX a model's reading of it). Two models read it (:data:`MODELS`, each pinned
at a revision), run by torch through transformers on a CUDA GPU when one answers, else on the CPU -- no server, no
ONNX runtime:

- ``texteller`` (the default): TexTeller, a ViT encoder and a RoBERTa decoder (code and weights Apache-2.0); 0.56 s an
  equation on the GPU, no image library (decision D85 of the document-lab-pdf-reader task).
- ``codeformula-v2``: docling's CodeFormulaV2, an Idefics3 vision-language model asked ``<formula>`` (weights
  CDLA-Permissive-2.0, fetched at run time, never redistributed); 0.03 more glyph F1 than TexTeller on 309 real
  crops, equal by hand, 3.4x slower on the GPU, and its image processor needs Pillow (D85, D87).

Weights download once into the Hugging Face cache (``HF_HUB_CACHE``, by default ``~/.cache/huggingface/hub``).
torch, transformers and Pillow are imported inside :func:`transcriber`, so the base package -- and this module --
import without them; asking for formulas without the extra, or without weights the cache holds or can fetch, raises
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

__all__ = [
    'MODELS',
    'UNREAD',
    'FormulaUnavailable',
    'Formulas',
    'Model',
    'Transcribe',
    'beside',
    'transcriber',
]

#: What an equation crop is written as beside its image: ``latex``.
Formulas = Literal['latex']
#: The model that reads it: TexTeller (the default) or docling's CodeFormulaV2.
Model = Literal['texteller', 'codeformula-v2']
#: Each model's Hugging Face repository and the revision it is pinned at.
MODELS: Final[dict[str, tuple[str, str]]] = {
    'texteller': ('OleehyO/TexTeller', '7b96df06b9d81cdb129c3bef68b7250bc3e2b0ea'),  # 2024-06-22, TexTeller 3
    'codeformula-v2': ('docling-project/CodeFormulaV2', 'ecedbe111d15c2dc60bfd4a823cbe80127b58af4'),  # 2025-08-11
}
#: An equation crop's pixels as written (`document_lab.png`: rows, columns and 3 RGB channels, or rows and columns
#: of gray; uint8) to its LaTeX; None where the model gives no reading: a blank crop, or a reading that never ends
#: (it repeats until the token budget runs out, measured on 2% of crops).
Transcribe = Callable[[np.ndarray], str | None]
#: What stands beside a crop the model gave no reading of: never silently nothing.
UNREAD: Final = '<!-- formulas: no LaTeX read from this crop (blank, or the reading never ended) -->'

_MAX_TOKENS: Final = 1024  # a reading longer than this never ended
_BORDER: Final = 15  # gray levels from the background color a pixel must differ by to be ink
_GRAY: Final = np.array([0.2989, 0.587, 0.114])  # ITU-R 601-2, as torchvision's Grayscale
_GRAY_AXES: Final = 2  # a gray crop's: rows and columns, no channel axis

# TexTeller's own inference transform (texteller/constants.py, utils/image.py): grayscale, the background border
# trimmed, scaled to fit a 448-pixel square, normalized by its training images' mean and spread, padded at the right
# and bottom.
_SIZE: Final = 448
_MEAN: Final = 0.9545467
_STD: Final = 0.15394445


class FormulaUnavailable(RuntimeError):
    """Formulas were asked for and the model cannot run here: the extra or its weights are missing."""


def transcriber(formulas: Formulas, *, model: Model = 'texteller', device: str | None = None) -> Transcribe:
    """The ``model``, loaded once: each call reads one equation crop's pixels as LaTeX (:data:`Transcribe`).

    ``device``: a torch device; by default CUDA when torch sees a GPU, else the CPU. Raises
    :class:`FormulaUnavailable` naming what is missing.
    """
    if formulas != 'latex':
        msg = f'formulas {formulas!r}: the one form is latex'
        raise ValueError(msg)
    if model not in MODELS:
        msg = f'formula model {model!r}: one of {", ".join(MODELS)}'
        raise ValueError(msg)
    try:
        import torch  # noqa: PLC0415 -- the extra is imported only when formulas are asked for
        import transformers  # noqa: PLC0415, F401

        if model == 'codeformula-v2':
            import PIL  # noqa: PLC0415, F401 -- its image processor's backend
    except ImportError as missing:
        msg = f"--formulas latex needs the 'formula' extra (torch, transformers, pillow): {missing}"
        raise FormulaUnavailable(msg) from missing
    repo, revision = MODELS[model]
    where = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        return (_texteller if model == 'texteller' else _codeformula)(repo, revision, where)
    except OSError as missing:
        msg = (
            f'--formulas latex needs the weights of {repo} at {revision[:12]}: neither in the Hugging Face cache '
            f'nor downloadable ({missing})'
        )
        raise FormulaUnavailable(msg) from missing


def _ink(pixels: np.ndarray) -> np.ndarray | None:
    """The crop's gray levels (0..255) cut to its ink; None for a blank crop (its corners' one color throughout)."""
    gray = pixels.astype(np.float64)
    if gray.ndim != _GRAY_AXES:
        gray = gray[..., :3] @ _GRAY
    corners = (gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1])
    background = Counter(np.round(corners)).most_common(1)[0][0]
    rows, cols = np.nonzero(np.abs(gray - background) > _BORDER)
    if not len(rows):
        return None
    return gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _texteller(repo: str, revision: str, device: str) -> Transcribe:
    import torch  # noqa: PLC0415
    from transformers import RobertaTokenizerFast, VisionEncoderDecoderModel  # noqa: PLC0415

    model = VisionEncoderDecoderModel.from_pretrained(repo, revision=revision, use_safetensors=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(repo, revision=revision)
    model = model.to(device).eval()

    def square(ink: np.ndarray) -> torch.Tensor:
        gray = torch.from_numpy(ink / 255)
        height, width = gray.shape
        scale = min((_SIZE - 1) / min(height, width), _SIZE / max(height, width))
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        fitted = torch.nn.functional.interpolate(gray[None, None], size=size, mode='bicubic', antialias=True)[0, 0]
        out = torch.zeros((_SIZE, _SIZE), dtype=torch.float32)
        out[: size[0], : size[1]] = (fitted.clamp(0, 1) - _MEAN) / _STD
        return out

    def transcribe(pixels: np.ndarray) -> str | None:
        ink = _ink(pixels)
        if ink is None:
            return None
        with torch.inference_mode():
            ids = model.generate(
                square(ink)[None, None].to(model.device),
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

    return transcribe


def _codeformula(repo: str, revision: str, device: str) -> Transcribe:
    """CodeFormulaV2 as docling asks it (docling/models/stages/code_formula): the crop and ``<formula>``, greedy."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(repo, revision=revision, use_fast=False)
    dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=revision, dtype=dtype).to(device).eval()
    asked = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '<formula>'}]}]
    prompt = processor.apply_chat_template(asked, add_generation_prompt=True)
    end = '<end_of_utterance>'

    def transcribe(pixels: np.ndarray) -> str | None:
        if _ink(pixels) is None:
            return None
        rgb = np.repeat(pixels[..., None], 3, axis=2) if pixels.ndim == _GRAY_AXES else pixels[..., :3]
        inputs = processor(text=[prompt], images=[np.ascontiguousarray(rgb)], return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=_MAX_TOKENS, do_sample=False)
        text = processor.batch_decode(ids[:, inputs['input_ids'].shape[1] :], skip_special_tokens=False)[0]
        if end not in text and ids.shape[1] - inputs['input_ids'].shape[1] >= _MAX_TOKENS:
            return None  # the budget ran out mid-reading: a loop, not a formula
        return re.sub(r'</?formula>|<loc_\d+>|<\|[a-z_]+\|>', '', text.split(end)[0]).strip() or None

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
    r"""``display`` without the equation's own ``number`` set after it (``\tag{3}``, ``\qquad (3)``, ``( 1 2 )``)."""
    if number is not None:
        own = r'\s*'.join(re.escape(c) for c in number)  # a tokenizer may space a number's characters
        display = re.sub(rf'\s*\\tag\*?\{{\s*{own}\s*\}}\s*$', '', display)
        display = re.sub(rf'(?:\s*(?:{_SPACE}))*\s*(?:\\left)?\(\s*{own}\s*(?:\\right)?\)\s*$', '', display)
    return display.strip()
