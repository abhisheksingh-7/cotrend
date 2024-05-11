"""
validation for parsing steps
"""

import functools
from typing import Any, Callable, TypeVar, overload

import pydantic


_Self = Any
T_contra = TypeVar("T_contra", contravariant=True, bound=pydantic.BaseModel)
T_co = TypeVar("T_co", covariant=True, bound=pydantic.BaseModel)


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
