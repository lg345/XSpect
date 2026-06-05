from XSpect.analysis.registry import (
    register_step,
    register_reduction,
    get_step,
    get_reduction,
    list_steps,
    list_reductions,
    clear_registry,
    StepNotFoundError,
    ReductionNotFoundError,
)

# Import step modules to trigger registration
import XSpect.analysis.spectroscopy  # noqa: F401
import XSpect.analysis.xes  # noqa: F401
import XSpect.analysis.xas  # noqa: F401
