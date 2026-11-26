"""Regression: 8-bit frames must decode correctly for every frame-store mix.

``get_components_internal`` creates two frame stores per decode (one for the
image data, one for the alpha channel) and each one asks the aimdo host buffer
for address space independently. They can therefore disagree: the image store
may be a ``HostFrames`` while the alpha store fell back to ``FrameChunks``.
That mix used to copy ``uint8`` frames into a buffer that is read as
``float32`` and raised ``cannot reshape array of size N into shape (H, W, 3)``.

The tests below decode real PNGs through ``VideoFromFile.get_components()`` for
all four reservation outcomes and compare the decoded values exactly. The
allocator boundary is replaced with a bounded ctypes-backed byte store so the
matrix is deterministic and needs neither a GPU nor a working aimdo binary;
PyAV, the production frame classes and ``hostbuf_to_tensor`` are the real ones.
"""

from __future__ import annotations

import ctypes
import io

import numpy as np
import pytest
import torch
from PIL import Image

from comfy_api.latest._input_impl import video_types

HEIGHT = 16
WIDTH = 32
CAPACITY = 1 << 20


class BoundedHostBuffer:
    """A real byte store that only exposes what ``HostFrames`` needs.

    It stands in for ``comfy_aimdo.host_buffer.HostBuffer``: fixed capacity,
    stable address, and the same ``extend``/``size``/``get_raw_address`` surface
    the production code uses.
    """

    def __init__(self, capacity: int = CAPACITY) -> None:
        self._storage = ctypes.create_string_buffer(capacity)
        self.size = 0

    def extend(self, size: int, reallocate: bool = False, register: bool = False) -> int:
        if reallocate:
            self.size = 0
        if self.size + size > len(self._storage):
            raise RuntimeError("test host buffer exhausted")
        address = ctypes.addressof(self._storage) + self.size
        self.size += size
        return address

    def get_raw_address(self) -> int:
        return ctypes.addressof(self._storage)

    def truncate(self, size: int = 0, do_unregister: bool = True) -> None:
        self.size = 0


def _png(with_alpha: bool) -> tuple[np.ndarray, io.BytesIO]:
    channels = 4 if with_alpha else 3
    values = np.arange(HEIGHT * WIDTH * channels, dtype=np.uint16)
    pixels = (values.reshape(HEIGHT, WIDTH, channels) % 256).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGBA" if with_alpha else "RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return pixels, buffer


def _decode(payload: io.BytesIO, frame_ok: bool, alpha_ok: bool):
    """Decode one PNG with the two reservations forced to the given outcomes."""
    outcomes = (frame_ok, alpha_ok)
    calls: list[bool] = []

    def reserve() -> BoundedHostBuffer:
        ok = outcomes[len(calls)]
        calls.append(ok)
        if not ok:
            raise RuntimeError("deliberate host buffer reservation failure")
        return BoundedHostBuffer()

    with (
        pytest.MonkeyPatch.context() as patcher,
    ):
        patcher.setattr(video_types.comfy.memory_management, "aimdo_enabled", True)
        patcher.setattr(video_types, "reserve_host_buffer", reserve)
        components = video_types.VideoFromFile(payload).get_components()
    return components, calls


RESERVATION_OUTCOMES = [
    (False, False),
    (True, True),
    (True, False),
    (False, True),
]


@pytest.mark.parametrize("with_alpha", [False, True], ids=["rgb", "rgba"])
@pytest.mark.parametrize(
    ("frame_ok", "alpha_ok"),
    RESERVATION_OUTCOMES,
    ids=["both_fallback", "both_host", "frames_host_alpha_fallback", "frames_fallback_alpha_host"],
)
def test_eight_bit_frames_survive_every_store_mix(with_alpha: bool, frame_ok: bool, alpha_ok: bool):
    pixels, payload = _png(with_alpha)

    components, calls = _decode(payload, frame_ok, alpha_ok)

    # frame store is asked first, the alpha store second
    assert calls == [frame_ok, alpha_ok]

    expected = torch.from_numpy(pixels[..., :3].astype(np.float32) / 255.0).unsqueeze(0)
    assert components.images.shape == (1, HEIGHT, WIDTH, 3)
    assert components.images.dtype == torch.float32
    torch.testing.assert_close(components.images, expected, rtol=0, atol=0)

    if with_alpha:
        expected_alpha = torch.from_numpy(pixels[..., 3:4].astype(np.float32) / 255.0).unsqueeze(0)
        assert components.alpha is not None
        torch.testing.assert_close(components.alpha, expected_alpha, rtol=0, atol=0)
    else:
        assert components.alpha is None
