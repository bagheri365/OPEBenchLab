import csv
import json
from pathlib import Path

import pytest

from opebenchlab.m60_benchmark import EXPERIMENTS, execute, inspect_csv, main


def test_registry_modules_importable():
    import importlib
    for module, _ in EXPERIMENTS.values():
        assert importlib.import_module(module)


def test_csv_metadata(tmp_path):
    path = tmp_path / 'a.csv'
    path.write_text('x,y\n1,2\n3,4\n')
    info = inspect_csv(path)
    assert info['rows'] == 2
    assert info['columns'] == ['x', 'y']
    assert len(info['sha256']) == 64


def test_run_smoke_and_refuse_overwrite(tmp_path):
    dest = tmp_path / 'run'
    m = execute('m512-support', dest, ['--trials', '1', '--requests', '1202', '--explorations', '0.2', '--ridges', '0.1'])
    assert m['returncode'] == 0
    assert set(m['artifacts']) == {'summary.csv', 'trials.csv', 'paired.csv'}
    assert m['artifacts']['trials.csv']['rows'] == 5
    assert json.loads((dest / 'manifest.json').read_text())['experiment'] == 'm512-support'
    with pytest.raises(ValueError, match='empty'):
        execute('m512-support', dest, [])


def test_list(capsys):
    main(['list'])
    assert 'm512-support' in capsys.readouterr().out


def test_unknown_experiment(tmp_path):
    with pytest.raises(ValueError, match='Unknown'):
        execute('unknown', tmp_path / 'x', [])
