"""
validation for parsing steps
"""

import functools
from typing import Any, Callable, TypeVar, overload

import numpy as np
import pydantic
import torch


_Self = Any
T_contra = TypeVar("T_contra", contravariant=True, bound=pydantic.BaseModel)
T_co = TypeVar("T_co", covariant=True, bound=pydantic.BaseModel)


@overload
def validate_batch(  # method on class
    f: Callable[[_Self, list[T_contra]], T_co]
) -> Callable[[_Self, dict[str, np.ndarray]], dict[str, Any]]: ...


@overload
def validate_batch(  # function
    f: Callable[[list[T_contra]], T_co]
) -> Callable[[dict[str, np.ndarray]], dict[str, Any]]: ...


def validate_batch(f: Callable) -> Callable:
    """ser / deser for ray.data steps"""
    validated_f = pydantic.validate_call(validate_return=True)(f)

    @functools.wraps(f)
    def decorator(*data: dict[str, np.ndarray]) -> dict[str, Any]:
        nonlocal validated_f
        data_list = list[Any](data)
        data_list[-1] = _unbatch(data[-1])
        result = validated_f(*data_list)
        return result.model_dump()

    return decorator


def _unbatch(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    lens = {len(data[key]) for key in data}
    assert len(lens) == 1, "All arrays must have the same length"
    thelen = lens.pop()
    return [{key: data[key][i] for key in data.keys()} for i in range(thelen)]


@overload
def validate(  # method on class
    f: Callable[[_Self, T_contra], T_co]
) -> Callable[[_Self, dict[str, Any]], dict[str, Any]]: ...


@overload
def validate(  # function
    f: Callable[[T_contra], T_co]
) -> Callable[[dict[str, Any]], dict[str, Any]]: ...


def validate(f: Callable) -> Callable:
    """ser / deser for ray.data steps"""
    validated_f = pydantic.validate_call(validate_return=True)(f)

    @functools.wraps(f)
    def decorator(*data: Any) -> dict[str, Any]:
        nonlocal validated_f
        result = validated_f(*data)
        return result.model_dump()

    return decorator


def validate_tensor(v: torch.Tensor) -> torch.Tensor:
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    return v


TENSOR_VALIDATOR = pydantic.BeforeValidator(validate_tensor)
