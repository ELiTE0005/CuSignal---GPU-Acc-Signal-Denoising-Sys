import os
import struct
import tempfile

import numpy as np
from data_loader.mstar_phoenix import load_mstar_chip, read_phoenix_chip


def write_minimal_phoenix(path, mag_u16_be):
    rows, cols = mag_u16_be.shape
    header = (
        f"PhoenixHeaderVer = 1.0\nNumberOfRows = {rows}\nNumberOfColumns = {cols}\n"
        "EndofPhoenixHeader\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for v in mag_u16_be.ravel():
            f.write(struct.pack(">H", int(v)))


def test_roundtrip_phoenix_mag_only():
    mag = np.arange(12, dtype=np.uint16).reshape(3, 4)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".001") as tf:
        path = tf.name
    try:
        write_minimal_phoenix(path, mag)
        chip = read_phoenix_chip(path)
        assert chip.magnitude.shape == (3, 4)
        np.testing.assert_allclose(chip.magnitude, mag.astype(np.float32), rtol=1e-5)
    finally:
        os.unlink(path)


def test_load_mstar_chip_dispatches():
    mag = np.ones((2, 2), dtype=np.uint16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".001") as tf:
        path = tf.name
    try:
        write_minimal_phoenix(path, mag)
        chip = load_mstar_chip(path)
        assert chip.magnitude.shape == (2, 2)
    finally:
        os.unlink(path)
