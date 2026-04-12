import os

import pytest

from data_loader.radarscenes_loader import resolve_radar_data_h5


def test_resolve_prefers_direct_sequence(tmp_path):
    seq = tmp_path / "sequence_7"
    seq.mkdir()
    h5 = seq / "radar_data.h5"
    h5.write_bytes(b"")
    found = resolve_radar_data_h5(str(tmp_path), 7)
    assert os.path.samefile(found, str(h5))


def test_resolve_data_subfolder(tmp_path):
    data = tmp_path / "data" / "sequence_2"
    data.mkdir(parents=True)
    h5 = data / "radar_data.h5"
    h5.write_bytes(b"")
    found = resolve_radar_data_h5(str(tmp_path), 2)
    assert os.path.samefile(found, str(h5))


def test_resolve_missing():
    with pytest.raises(FileNotFoundError):
        resolve_radar_data_h5("/nonexistent/path/that/does/not/exist", 1)
