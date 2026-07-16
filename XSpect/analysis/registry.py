"""
Step and reduction registry for the XSpect pipeline.

Steps are stateless functions with signature: step(run, **kwargs) -> None
Reductions receive all runs: reduction(runs, **kwargs) -> dict
"""

_STEP_REGISTRY: dict[str, callable] = {}
_REDUCTION_REGISTRY: dict[str, callable] = {}


class StepNotFoundError(KeyError):
    """Raised when a step name is not found in the registry."""
    pass


class ReductionNotFoundError(KeyError):
    """Raised when a reduction name is not found in the registry."""
    pass


def register_step(name: str):
    """Decorator that registers a function as a pipeline step."""
    def decorator(func):
        if name in _STEP_REGISTRY:
            raise ValueError(f"Step '{name}' is already registered")
        _STEP_REGISTRY[name] = func
        func._step_name = name
        return func
    return decorator


def register_reduction(name: str):
    """Decorator that registers a function as a reduction step."""
    def decorator(func):
        if name in _REDUCTION_REGISTRY:
            raise ValueError(f"Reduction '{name}' is already registered")
        _REDUCTION_REGISTRY[name] = func
        func._reduction_name = name
        return func
    return decorator


def get_step(name: str) -> callable:
    """Look up a registered step by name."""
    try:
        return _STEP_REGISTRY[name]
    except KeyError:
        raise StepNotFoundError(f"Step '{name}' not found. Available: {list(_STEP_REGISTRY.keys())}")


def get_reduction(name: str) -> callable:
    """Look up a registered reduction by name."""
    try:
        return _REDUCTION_REGISTRY[name]
    except KeyError:
        raise ReductionNotFoundError(
            f"Reduction '{name}' not found. Available: {list(_REDUCTION_REGISTRY.keys())}"
        )


def list_steps() -> list[str]:
    """Return all registered step names."""
    return list(_STEP_REGISTRY.keys())


def list_reductions() -> list[str]:
    """Return all registered reduction names."""
    return list(_REDUCTION_REGISTRY.keys())


def clear_registry():
    """Clear all registered steps and reductions. Mainly for testing."""
    _STEP_REGISTRY.clear()
    _REDUCTION_REGISTRY.clear()
