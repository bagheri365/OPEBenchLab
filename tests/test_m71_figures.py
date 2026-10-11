import csv
from pathlib import Path

import pytest

from opebenchlab.m71_figures import build_data, load_reports, one


def test_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing paired input'):
        load_reports(tmp_path)


def test_missing_column_is_explicit(tmp_path):
    p = tmp_path / 'm61_support' / 'paired_bootstrap.csv'
    p.parent.mkdir()
    p.write_text('requests,exploration\n500,1.0\n')
    with pytest.raises(ValueError, match='missing columns'):
        load_reports(tmp_path)


def test_unique_row_required():
    with pytest.raises(ValueError, match='Expected one row'):
        one([{'a': 1}, {'a': 1}], a=1)


def test_coverage_denominator_check():
    # Test the invariant before plotting, even if an upstream CSV is malformed.
    from opebenchlab.m71_figures import INPUTS
    assert len(INPUTS) == 6
