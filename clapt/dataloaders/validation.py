import functools
from typing import Any, Callable, TypeVar, cast

import pydantic

T_contra = TypeVar("T_contra", contravariant=True, bound=pydantic.BaseModel)
T_co = TypeVar("T_co", covariant=True, bound=pydantic.BaseModel)


def validate(
    f: Callable[[T_contra], T_co]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """ser / deser for ray.data steps"""
    validated_f = pydantic.validate_call(validate_return=True)(f)

    @functools.wraps(f)
    def decorator(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal validated_f
        result = validated_f(data)  # type: ignore
        return result.model_dump(mode="json")

    return decorator
