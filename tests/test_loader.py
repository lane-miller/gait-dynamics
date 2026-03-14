"""
Unit tests for gait_dynamics.data.loader module.

Filesystem structure is built with pytest's tmp_path fixture using empty
.hea stub files (no WFDB data content). wfdb.rdrecord is patched for every
test that would otherwise attempt to parse those stubs.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from gait_dynamics.data.loader import list_subjects, load_subject

_PATCH_RDRECORD = "gait_dynamics.data.loader.wfdb.rdrecord"


@pytest.fixture
def mock_record():
    """WFDB record stub: 3 acc channels, fs=100, (500, 3) p_signal."""
    record = MagicMock()
    record.p_signal = np.zeros((500, 3))
    record.sig_name = ["acc_x", "acc_y", "acc_z"]
    record.fs = 100
    return record


def _make_subject(root, subject_id, trials=("trial_01",)):
    """Create a subject directory with one empty .hea stub per trial."""
    subj_dir = root / subject_id
    subj_dir.mkdir()
    for trial in trials:
        (subj_dir / f"{subject_id}_{trial}.hea").touch()
    return subj_dir


class TestListSubjects:

    def test_returns_sorted_subject_list(self, tmp_path):
        """Directories created out of order must be returned alphabetically."""
        for subject_id in ["AB03", "AB01", "AB02"]:
            _make_subject(tmp_path, subject_id)

        assert list_subjects(tmp_path) == ["AB01", "AB02", "AB03"]

    def test_raises_fnf_when_data_dir_missing(self):
        """Non-existent data_dir must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list_subjects("/nonexistent/camargo/dataset")


class TestLoadSubject:

    def test_returns_dict_with_correct_keys(self, tmp_path, mock_record):
        """Returned dict must contain exactly: subject_id, acc, sample_rate."""
        _make_subject(tmp_path, "AB01")

        with patch(_PATCH_RDRECORD, return_value=mock_record):
            result = load_subject(tmp_path, "AB01")

        assert set(result.keys()) == {"subject_id", "acc", "sample_rate"}

    def test_acc_shape_is_n_samples_by_3(self, tmp_path, mock_record):
        """acc array must have shape (500, 3) matching the mock p_signal."""
        _make_subject(tmp_path, "AB01")

        with patch(_PATCH_RDRECORD, return_value=mock_record):
            result = load_subject(tmp_path, "AB01")

        assert result["acc"].shape == (500, 3)

    def test_raises_fnf_for_missing_subject_directory(self, tmp_path):
        """Subject directory that was never created must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_subject(tmp_path, "AB99")

    def test_raises_fnf_for_unknown_trial_id_lists_available(self, tmp_path):
        """Error message for an unknown trial_id must list available trial names."""
        _make_subject(tmp_path, "AB01", trials=("LG_R01",))
        with pytest.raises(FileNotFoundError, match="AB01_LG_R01"):
            load_subject(tmp_path, "AB01", trial_id="AB01_SD_R01")

    def test_raises_value_error_for_wrong_acc_channel_count(self, tmp_path):
        """Record with only 2 acc channels must raise ValueError."""
        _make_subject(tmp_path, "AB01")

        bad_record = MagicMock()
        bad_record.p_signal = np.zeros((500, 2))
        bad_record.sig_name = ["acc_x", "acc_y"]
        bad_record.fs = 100

        with patch(_PATCH_RDRECORD, return_value=bad_record):
            with pytest.raises(ValueError, match="exactly 3"):
                load_subject(tmp_path, "AB01")
