"""Tests for XSpect.controller.config_parser"""

import pytest
import os
import tempfile
import yaml

from XSpect.controller.config_parser import (
    parse_yaml, ConfigValidationError, _expand_runs,
    PipelineConfig, ExperimentConfig, DataConfig, StepConfig, OutputConfig,
)


@pytest.fixture
def valid_yaml(tmp_path):
    content = {
        'experiment': {
            'hutch': 'xcs',
            'experiment_id': 'xcsp23820',
            'lcls_run': 21,
        },
        'data': {
            'runs': [1, 2, 3],
            'keys': {'ipm4/sum': 'ipm', 'tt/ttCorr': 'time_tool_correction'},
        },
        'pipeline': [
            {'step': 'placeholder_add', 'on': 'data', 'value': 1.0},
            {'step': 'placeholder_multiply', 'on': 'data', 'value': 2.0},
        ],
        'output': {'format': 'hdf5', 'path': './results/'},
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)
    return str(path)


def test_parse_valid_yaml(valid_yaml):
    config = parse_yaml(valid_yaml)
    assert isinstance(config, PipelineConfig)
    assert config.experiment.hutch == 'xcs'
    assert config.experiment.experiment_id == 'xcsp23820'
    assert config.experiment.lcls_run == 21
    assert config.data.runs == [1, 2, 3]
    assert len(config.pipeline) == 2
    assert config.pipeline[0].step == 'placeholder_add'
    assert config.output.format == 'hdf5'


def test_missing_experiment_section(tmp_path):
    content = {
        'data': {'runs': [1], 'keys': {'a': 'b'}},
        'pipeline': [{'step': 'placeholder_add'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    with pytest.raises(ConfigValidationError, match="experiment"):
        parse_yaml(str(path))


def test_missing_pipeline_section(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {'runs': [1], 'keys': {'a': 'b'}},
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    with pytest.raises(ConfigValidationError, match="pipeline"):
        parse_yaml(str(path))


def test_missing_data_section(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'pipeline': [{'step': 'placeholder_add'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    with pytest.raises(ConfigValidationError, match="data"):
        parse_yaml(str(path))


def test_unknown_step_name(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {'runs': [1], 'keys': {'a': 'b'}},
        'pipeline': [{'step': 'totally_fake_step'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    with pytest.raises(ConfigValidationError, match="unknown step"):
        parse_yaml(str(path))


def test_step_missing_step_field(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {'runs': [1], 'keys': {'a': 'b'}},
        'pipeline': [{'on': 'data', 'value': 1.0}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    with pytest.raises(ConfigValidationError, match="missing required 'step' field"):
        parse_yaml(str(path))


def test_run_range_expansion():
    assert _expand_runs(['162-165', 170]) == [162, 163, 164, 165, 170]
    assert _expand_runs([1, 2, 3]) == [1, 2, 3]
    assert _expand_runs(['5-7']) == [5, 6, 7]


def test_output_section_defaults(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {'runs': [1], 'keys': {'a': 'b'}},
        'pipeline': [{'step': 'placeholder_add'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    config = parse_yaml(str(path))
    assert config.output.format == 'hdf5'
    assert config.output.path == './results/'


def test_reduction_section_optional(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {'runs': [1], 'keys': {'a': 'b'}},
        'pipeline': [{'step': 'placeholder_add'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    config = parse_yaml(str(path))
    assert config.reduction == []


def test_file_not_found():
    with pytest.raises(ConfigValidationError, match="not found"):
        parse_yaml("/nonexistent/path/to/file.yaml")


def test_detector_keys_parsed(tmp_path):
    content = {
        'experiment': {'hutch': 'x', 'experiment_id': 'y', 'lcls_run': 1},
        'data': {
            'runs': [1],
            'keys': {'a': 'b'},
            'detector_keys': {
                'epix_1/ROI_0_area': {
                    'name': 'epix',
                    'rois': [[0, -1]],
                    'combine_rois': True,
                }
            },
        },
        'pipeline': [{'step': 'placeholder_add'}],
    }
    path = tmp_path / "test.yaml"
    with open(path, 'w') as f:
        yaml.dump(content, f)

    config = parse_yaml(str(path))
    assert len(config.data.detector_keys) == 1
    assert config.data.detector_keys[0].name == 'epix'
    assert config.data.detector_keys[0].combine_rois is True
