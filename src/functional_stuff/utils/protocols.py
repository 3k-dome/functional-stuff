__all__ = ("Comparable", "SupportsGreaterThan", "SupportsLessThan")


from typing import Any, Protocol, TypeVar, runtime_checkable

T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class SupportsLessThan(Protocol[T_contra]):
    def __lt__(self, other: T_contra, /) -> bool: ...


@runtime_checkable
class SupportsGreaterThan(Protocol[T_contra]):
    def __gt__(self, other: T_contra, /) -> bool: ...


Comparable = SupportsLessThan[Any] | SupportsGreaterThan[Any]
ComparableT = TypeVar("ComparableT", bound=Comparable)
