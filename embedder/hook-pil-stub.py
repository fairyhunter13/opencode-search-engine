"""PyInstaller runtime hook: minimal Pillow + fastembed image stubs.

This runs before any application code is loaded.

fastembed imports Pillow types at module import time (even when we only use
text models). Newer fastembed versions also import image preprocessing helpers
from `fastembed.image.transform.*` inside tokenizer utilities, which pulls in
Pillow enums like `Image.Resampling`.

We exclude Pillow from the PyInstaller build to keep the bundle small, so we
provide tiny stubs that satisfy these imports. Image embedding/preprocessing is
not supported in this build.
"""

import sys
import types


def _make_stub(name, *, pkg=False):
    mod = types.ModuleType(name)
    if pkg:
        mod.__path__ = []  # Make it a package so sub-imports work
    mod.__loader__ = None
    return mod


pil = sys.modules.get("PIL")
if pil is None:
    pil = _make_stub("PIL", pkg=True)
    sys.modules["PIL"] = pil

image = sys.modules.get("PIL.Image")
if image is None:
    image = _make_stub("PIL.Image")
    sys.modules["PIL.Image"] = image

if not hasattr(image, "Image"):

    class Image:  # noqa: N801 - mimic Pillow's class name
        pass

    image.Image = Image

if not hasattr(image, "Resampling"):
    import enum

    class Resampling(enum.IntEnum):
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1

    image.Resampling = Resampling

# Ensure `from PIL import Image` returns the `PIL.Image` module.
if not hasattr(pil, "Image"):
    pil.Image = image


def _stub_fastembed_images():
    # fastembed imports image helpers even for text-only code paths.
    # Stub them so fastembed can import without bundling Pillow.
    if "fastembed.image" not in sys.modules:
        mod = _make_stub("fastembed.image", pkg=True)

        class ImageEmbedding:
            def __init__(self, *_, **__):
                raise RuntimeError("fastembed image embeddings are not supported in this build")

        mod.ImageEmbedding = ImageEmbedding
        mod.__all__ = ["ImageEmbedding"]
        sys.modules["fastembed.image"] = mod

    if "fastembed.image.transform" not in sys.modules:
        sys.modules["fastembed.image.transform"] = _make_stub("fastembed.image.transform", pkg=True)

    if "fastembed.image.transform.operators" not in sys.modules:
        mod = _make_stub("fastembed.image.transform.operators")

        class Compose:
            @staticmethod
            def from_config(*_, **__):
                raise RuntimeError("fastembed image preprocessing is not supported in this build")

            def __call__(self, *_, **__):
                raise RuntimeError("fastembed image preprocessing is not supported in this build")

        mod.Compose = Compose
        mod.__all__ = ["Compose"]
        sys.modules["fastembed.image.transform.operators"] = mod


_stub_fastembed_images()
