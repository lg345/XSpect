"""Integration tests for the Pipeline class."""

import pytest
import os

from XSpect.controller.pipeline import Pipeline
from XSpect.controller.config_parser import PipelineConfig


def test_pipeline_from_yaml_basic(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline.yaml')
    pipeline = Pipeline.from_yaml(path)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.config, PipelineConfig)
    assert pipeline.config.experiment.hutch == 'test'


def test_pipeline_run_placeholder_steps(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline.yaml')
    pipeline = Pipeline.from_yaml(path)
    pipeline.run(cores=1, batch_size=100)

    assert len(pipeline.analyzed_runs) == 1
    run = pipeline.analyzed_runs[0]
    assert "epix.added" in run.results
    assert "epix.multiplied" in run.results
    assert run.results["epix.added"] == 1.0
    assert run.results["epix.multiplied"] == 0


def test_pipeline_reduction_phase(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline.yaml')
    pipeline = Pipeline.from_yaml(path)
    pipeline.run(cores=1, batch_size=100)

    assert "summed" in pipeline.results


def test_pipeline_multiple_runs(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline_no_reduction.yaml')
    pipeline = Pipeline.from_yaml(path)
    pipeline.run(cores=1, batch_size=100)

    assert len(pipeline.analyzed_runs) == 2
    for run in pipeline.analyzed_runs:
        assert "data.added" in run.results
        assert run.results["data.added"] == 5.0


def test_pipeline_results_contains_run_keys(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline_no_reduction.yaml')
    pipeline = Pipeline.from_yaml(path)
    pipeline.run(cores=1, batch_size=100)

    assert "run_1.data.added" in pipeline.results
    assert "run_2.data.added" in pipeline.results


def test_pipeline_status_log(fixtures_dir):
    path = os.path.join(fixtures_dir, 'test_pipeline.yaml')
    pipeline = Pipeline.from_yaml(path)
    pipeline.run(cores=1, batch_size=100)

    assert "Pipeline execution started" in pipeline._status_log
    assert "Pipeline execution complete" in pipeline._status_log
