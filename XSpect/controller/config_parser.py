"""
YAML configuration parser for XSpect pipelines.

Parses a YAML file into frozen dataclass structures and validates
that all referenced steps exist in the registry.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path

from XSpect.analysis.registry import (
    get_step,
    get_reduction,
    StepNotFoundError,
    ReductionNotFoundError,
)


class ConfigValidationError(ValueError):
    """Raised when YAML configuration is invalid."""

    pass


@dataclass(frozen=True)
class ExperimentConfig:
    hutch: str
    experiment_id: str
    lcls_run: int
    smalldata_dir: "str | None" = None  # Optional override for the smalldata
    # directory. If set, XSpect looks for {exp}_Run{run:04d}.h5 here instead of
    # the default /sdf/data/lcls/ds/{hutch}/{exp}/hdf5/smalldata search list.


@dataclass(frozen=True)
class DataKeyConfig:
    hdf5_path: str
    friendly_name: str


@dataclass(frozen=True)
class DetectorKeyConfig:
    hdf5_path: str
    name: str
    rois: list
    combine_rois: bool = True
    transpose: bool = False
    row_range: "list | None" = None  # [start, end] crop on the row axis at import time
    # Applied BEFORE any column ROIs.
    # In the transposed frame (when transpose=True) this
    # selects rows 170-280 of the (768, 704) array, which
    # corresponds to slicing axis=2 of the raw HDF5 frame.


@dataclass(frozen=True)
class DataConfig:
    runs: list
    keys: list
    detector_keys: list = field(default_factory=list)
    max_shots: "int | None" = None  # limit total shots loaded per run (None = all)


@dataclass(frozen=True)
class StepConfig:
    step: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    format: str = "hdf5"
    path: str = "./results/"


@dataclass(frozen=True)
class PipelineConfig:
    experiment: ExperimentConfig
    data: DataConfig
    pipeline: list
    reduction: list = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)


def _expand_runs(run_list) -> list[int]:
    """Expand run ranges like '162-181' into individual integers."""
    expanded = []
    for item in run_list:
        item_str = str(item)
        if "-" in item_str:
            parts = item_str.split("-")
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                expanded.extend(range(start, end + 1))
            else:
                expanded.append(int(item_str))
        else:
            expanded.append(int(item_str))
    return expanded


def _parse_experiment(raw: dict) -> ExperimentConfig:
    required = ["hutch", "experiment_id", "lcls_run"]
    for key in required:
        if key not in raw:
            raise ConfigValidationError(
                f"experiment section missing required field: '{key}'"
            )
    return ExperimentConfig(
        hutch=str(raw["hutch"]),
        experiment_id=str(raw["experiment_id"]),
        lcls_run=int(raw["lcls_run"]),
        smalldata_dir=(str(raw["smalldata_dir"]) if raw.get("smalldata_dir") else None),
    )


def _parse_data(raw: dict) -> DataConfig:
    if "runs" not in raw:
        raise ConfigValidationError("data section missing required field: 'runs'")
    if "keys" not in raw:
        raise ConfigValidationError("data section missing required field: 'keys'")

    runs = _expand_runs(raw["runs"])

    keys = []
    for hdf5_path, friendly_name in raw["keys"].items():
        keys.append(
            DataKeyConfig(hdf5_path=str(hdf5_path), friendly_name=str(friendly_name))
        )

    detector_keys = []
    if "detector_keys" in raw:
        for hdf5_path, config in raw["detector_keys"].items():
            detector_keys.append(
                DetectorKeyConfig(
                    hdf5_path=str(hdf5_path),
                    name=config.get("name", hdf5_path),
                    rois=config.get("rois", None),
                    combine_rois=config.get("combine_rois", True),
                    transpose=config.get("transpose", False),
                    row_range=config.get("row_range", None),
                )
            )

    max_shots = None
    if "max_shots" in raw:
        val = raw["max_shots"]
        if val is not None:
            max_shots = int(val)

    return DataConfig(
        runs=runs, keys=keys, detector_keys=detector_keys, max_shots=max_shots
    )


def _normalize_yaml_keys(d: dict) -> dict:
    """
    PyYAML parses 'on' as boolean True and 'off' as boolean False.
    Convert any boolean keys back to their string equivalents.
    """
    normalized = {}
    for k, v in d.items():
        if k is True:
            normalized["on"] = v
        elif k is False:
            normalized["off"] = v
        else:
            normalized[str(k)] = v
    return normalized


def _parse_steps(raw_list: list, section_name: str) -> list[StepConfig]:
    """Parse pipeline or reduction step list."""
    steps = []
    for i, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            raise ConfigValidationError(
                f"{section_name}[{i}]: each entry must be a mapping, got {type(entry).__name__}"
            )
        entry = _normalize_yaml_keys(entry)
        if "step" not in entry:
            raise ConfigValidationError(
                f"{section_name}[{i}]: missing required 'step' field"
            )
        step_name = entry["step"]
        args = {k: v for k, v in entry.items() if k != "step"}
        steps.append(StepConfig(step=step_name, args=args))
    return steps


def _validate_step_names(steps: list[StepConfig], section: str):
    """Check that all step names are registered."""
    for step_config in steps:
        if section == "pipeline":
            try:
                get_step(step_config.step)
            except StepNotFoundError as e:
                raise ConfigValidationError(
                    f"pipeline references unknown step '{step_config.step}': {e}"
                )
        elif section == "reduction":
            try:
                get_reduction(step_config.step)
            except ReductionNotFoundError as e:
                raise ConfigValidationError(
                    f"reduction references unknown reduction '{step_config.step}': {e}"
                )


def parse_yaml(path: str, validate_steps: bool = True) -> PipelineConfig:
    """
    Parse a YAML pipeline configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.
    validate_steps : bool
        If True, validate that all step names exist in the registry.

    Returns
    -------
    PipelineConfig
        Frozen dataclass with all configuration sections.

    Raises
    ------
    ConfigValidationError
        If the YAML is missing required sections or contains invalid entries.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigValidationError("YAML root must be a mapping")

    if "experiment" not in raw:
        raise ConfigValidationError("Missing required section: 'experiment'")
    if "data" not in raw:
        raise ConfigValidationError("Missing required section: 'data'")
    if "pipeline" not in raw:
        raise ConfigValidationError("Missing required section: 'pipeline'")

    experiment = _parse_experiment(raw["experiment"])
    data = _parse_data(raw["data"])
    pipeline_steps = _parse_steps(raw["pipeline"], "pipeline")

    reduction_steps = []
    if "reduction" in raw and raw["reduction"]:
        reduction_steps = _parse_steps(raw["reduction"], "reduction")

    output = OutputConfig()
    if "output" in raw and raw["output"]:
        output = OutputConfig(
            format=raw["output"].get("format", "hdf5"),
            path=raw["output"].get("path", "./results/"),
        )

    if validate_steps:
        _validate_step_names(pipeline_steps, "pipeline")
        _validate_step_names(reduction_steps, "reduction")

    return PipelineConfig(
        experiment=experiment,
        data=data,
        pipeline=pipeline_steps,
        reduction=reduction_steps,
        output=output,
    )
