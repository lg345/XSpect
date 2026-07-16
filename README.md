# XSpect

Processes X-ray spectroscopy data from LCLS (XES, XAS, RIXS). Reads smalldata HDF5, runs a configurable analysis pipeline in parallel across runs, and reduces the result into binned spectra. An analysis is defined by one YAML file, not a Python subclass.

## Installation

Clone the repository:

```bash
git clone https://github.com/lg345/XSpect.git
```

Install dependencies (numpy, h5py, scikit-learn, pyyaml, and the LCLS analysis environment on S3DF).

## Quick start

```python
from XSpect import Pipeline

pipe = Pipeline.from_yaml("my_analysis.yaml")
pipe.run(cores=16, batch_size=2000)

print(pipe.results.keys())
```

The YAML file defines the full workflow: which experiment and runs to load, which HDF5 keys to read, which analysis steps to run in order, and how to reduce across runs. `cores` and `batch_size` control parallelism at runtime without editing the recipe.

See [`docs/YAML_PIPELINE_GUIDE.md`](docs/YAML_PIPELINE_GUIDE.md) for the full reference: all five YAML sections, every registered step, naming conventions, and complete examples.

## Architecture

XSpect separates the analysis into model, controller, and analysis layers.

```
XSpect/
  model/         data containers
    experiment.py    experiment metadata, run collection
    run.py           a single run: loaded arrays + computed results
    von_hamos.py     crystal geometry, pixel-to-energy conversion
  controller/    orchestration
    pipeline.py      Pipeline class: reads YAML, dispatches steps
    config_parser.py parses and validates the YAML recipe
    batch_manager.py loads HDF5, runs per-run pipelines under multiprocessing.Pool
    pipeline_runner.py
  analysis/      the steps themselves
    registry.py      @register_step / @register_reduction, get_step, get_reduction
    spectroscopy.py  detector filtering, binning, normalization, patching
    xes.py, xas.py, droplet.py
```

### The registry pattern

Analysis operations are stateless functions registered by name:

```python
from XSpect import register_step

@register_step("normalize_xes")
def normalize_xes(run, **kwargs):
    """Normalize a spectrum so each row sums to 1 over a pixel range."""
    ...
```

A step takes a `run` object and keyword arguments, reads named attributes off the run, and writes results back as new attributes. Steps do not hold state, so each one is unit-testable in isolation (see `tests/`). A reduction has the same shape but runs across all runs after the per-run pipelines finish and returns a dict.

The `Pipeline` reads the ordered step list from YAML, looks each name up with `get_step`, and calls it. Adding a new operation means writing one function and decorating it. Adding a new experiment means writing one YAML file.

### Data flow

1. `config_parser` reads the YAML and expands run ranges.
2. `batch_manager` loads the requested HDF5 keys and detector ROIs for each run, then farms runs out to a `multiprocessing.Pool`.
3. For each run, the `pipeline` steps run in the listed order, mutating the run object.
4. If a `reduction` section is present, its steps run once across all completed runs.
5. Results land in `pipe.results` and, if an `output` section is given, on disk.

## Use cases

The same pipeline machinery covers the common LCLS spectroscopy modes. Example configs live in [`examples/`](examples/):

- **Static XES**: single-state emission spectra. `examples/mfx101080524_static_xes.yaml`
- **Time-resolved (pump-probe) XES**: laser-on minus laser-off, binned by delay. `examples/mfxl1027922_ultrafast_xes.yaml`
- **XAS energy scans**: absorption vs incident energy, including 2D and temporal. `examples/xcs101591326_2d_xas.yaml`, `examples/xcs101591326_temporal_xas.yaml`
- **Droplet / photon-counting**: sparse reconstruction from the MFX droplet HDF5 layout. `examples/mfx101609126_droplet_pershot_xes.yaml`
- **RIXS**: `examples/mfx101609126_static_rixs.yaml`

## Registered steps (selected)

Full list and arguments in the [YAML pipeline guide](docs/YAML_PIPELINE_GUIDE.md).

- Shot filtering: `filter_shots`, `filter_detector_adu`, `filter_detector_variance` (sklearn VarianceThreshold, data-driven alternative to ADU/keV thresholds)
- Detector prep: `patch_pixels` (manual or auto-detected bad columns), `rotate_detector`, `find_rotation_angle`, `droplet_reconstruction`
- Binning and reduction: `reduce_detector_temporal`, `combine_runs`
- Spectra: `make_energy_axis`, `normalize_xes`

## Skills

Agent skills for common workflows live under [`skills/`](skills/):

- [`XSpect-setup-XES`](skills/XSpect-setup-XES/SKILL.md): end-to-end setup of a new XES experiment. LUTE smalldata pipeline, LUTE YAML, XSpect pipeline YAML, and diagnostic/analysis notebooks. Covers static, time-resolved (pump-probe), CCM-scanned, and droplet/photon-counted XES.

## Documentation

- [`docs/YAML_PIPELINE_GUIDE.md`](docs/YAML_PIPELINE_GUIDE.md): YAML pipeline reference.
- [Source docs](https://lg345.github.io/XSpect/index.html): API details and examples.

## License

Copyright 2025 XSpecT Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
