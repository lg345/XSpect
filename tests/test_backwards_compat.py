"""Tests for backwards compatibility of import paths."""

import pytest


def test_import_experiment_old_path():
    from XSpect.XSpect_Analysis import experiment
    assert experiment is not None
    assert callable(experiment)


def test_import_spectroscopy_run_old_path():
    from XSpect.XSpect_Analysis import spectroscopy_run
    assert spectroscopy_run is not None
    assert callable(spectroscopy_run)


def test_import_spectroscopy_experiment_old_path():
    from XSpect.XSpect_Analysis import spectroscopy_experiment
    assert spectroscopy_experiment is not None
    assert callable(spectroscopy_experiment)


def test_import_from_new_model_path():
    from XSpect.model import experiment, spectroscopy_experiment, spectroscopy_run
    assert experiment is not None
    assert spectroscopy_experiment is not None
    assert spectroscopy_run is not None


def test_import_registry_from_analysis():
    from XSpect.analysis import register_step, register_reduction, get_step, list_steps
    assert callable(register_step)
    assert callable(register_reduction)
    assert callable(get_step)
    assert callable(list_steps)


def test_import_pipeline_from_controller():
    from XSpect.controller import Pipeline
    assert Pipeline is not None
    assert hasattr(Pipeline, 'from_yaml')
    assert hasattr(Pipeline, 'run')


def test_import_from_top_level():
    from XSpect import Pipeline, register_step, spectroscopy_run
    assert Pipeline is not None
    assert callable(register_step)
    assert spectroscopy_run is not None
