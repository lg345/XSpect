"""Tests for XSpect.analysis.registry"""

import pytest
from XSpect.analysis.registry import (
    register_step, register_reduction,
    get_step, get_reduction,
    list_steps, list_reductions,
    StepNotFoundError, ReductionNotFoundError,
    _STEP_REGISTRY, _REDUCTION_REGISTRY,
)


def test_register_step_adds_to_registry():
    assert 'placeholder_add' in _STEP_REGISTRY


def test_get_step_returns_callable():
    step_fn = get_step('placeholder_add')
    assert callable(step_fn)


def test_get_step_raises_on_missing():
    with pytest.raises(StepNotFoundError):
        get_step('nonexistent_step_xyz')


def test_list_steps_returns_all_names():
    names = list_steps()
    assert 'placeholder_add' in names
    assert 'placeholder_multiply' in names


def test_register_reduction_separate():
    assert 'placeholder_sum' in _REDUCTION_REGISTRY
    assert 'placeholder_sum' not in _STEP_REGISTRY


def test_get_reduction_returns_callable():
    fn = get_reduction('placeholder_sum')
    assert callable(fn)


def test_get_reduction_raises_on_missing():
    with pytest.raises(ReductionNotFoundError):
        get_reduction('nonexistent_reduction_xyz')


def test_list_reductions():
    names = list_reductions()
    assert 'placeholder_sum' in names


def test_duplicate_step_name_raises():
    with pytest.raises(ValueError, match="already registered"):
        @register_step("placeholder_add")
        def duplicate_step(run, **kwargs):
            pass


def test_duplicate_reduction_name_raises():
    with pytest.raises(ValueError, match="already registered"):
        @register_reduction("placeholder_sum")
        def duplicate_reduction(runs, **kwargs):
            pass


def test_step_decorator_preserves_function():
    step_fn = get_step('placeholder_add')
    assert hasattr(step_fn, '_step_name')
    assert step_fn._step_name == 'placeholder_add'
