__all__ = ("Maybe", "Nothing", "Some", "maybe")


from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import total_ordering, wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast, final

from functional_stuff.monads.base import AbstractMonad, MonadT, T, U


class AbstractMaybe(AbstractMonad[T]):
    """Abstract base for a maybe monad."""

    @abstractmethod
    def is_some(self) -> bool:
        """Returns `True` if some value is contained."""
        ...

    @abstractmethod
    def is_nothing(self) -> bool:
        """Returns `True` if no value is contained."""
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the contained value if any, else raises."""
        ...

    @abstractmethod
    def unwrap_or_default(self, default: T) -> T:
        """Returns the contained value if any, else default."""
        ...

    @abstractmethod
    def __lt__(self, other: "Maybe[T]") -> bool: ...


MaybeT = TypeVar("MaybeT", bound="AbstractMaybe[Any]")


@final
@total_ordering
@dataclass(slots=True, frozen=True)
class Some(AbstractMaybe[T]):
    """Some value of type `T`.

    Being implemented as frozen dataclass automatically provides methods like `__str__` and `__repr__`,
    as well as tuple-based `__eq__` and `__hash__` methods. An `__lt__` method, which proxies the `__lt__`
    operator of the inner value (if any), is also provided and is used to apply `functools.total_ordering`.
    An instance of `Nothing` is **always** less than an instance `Some`.
    """

    value: T

    def map(self, func: Callable[[T], U | None]) -> "Maybe[U]":
        if result := func(self.value):
            return Some(result)
        return Nothing()

    def bind(self, func: Callable[[T], MaybeT]) -> MaybeT:
        return func(self.value)

    def join(self: "Some[MonadT]") -> MonadT:
        return self.value

    def is_some(self) -> Literal[True]:
        return True

    def is_nothing(self) -> Literal[False]:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or_default(self, default: T) -> T:  # noqa: ARG002
        return self.value

    def __lt__(self, other: "Some[T] | Nothing[T]") -> bool:
        # annotating this with `Maybe[T]` breaks when `functools.total_ordering` is applied
        match other:
            case Some(value):
                return cast(bool, self.value < value)  # pyright: ignore[reportOperatorIssue]
            case _:
                return False


@final
@total_ordering
@dataclass(slots=True, frozen=True)
class Nothing(AbstractMaybe[T]):
    """No value of type `T`.

    Being implemented as frozen dataclass automatically provides methods like `__str__` and `__repr__`,
    as well as tuple-based `__eq__` and `__hash__` methods. An `__lt__` method, which proxies the `__lt__`
    operator of the inner value (if any), is also provided and is used to apply `functools.total_ordering`.
    An instance of `Nothing` is **always** less than an instance `Some`.
    """

    def map(self, func: Callable[[T], U | None]) -> "Maybe[U]":  # noqa: ARG002
        return cast(Nothing[U], self)

    def bind(self, func: Callable[[T], MaybeT]) -> MaybeT:  # noqa: ARG002
        return cast(MaybeT, self)

    def join(self) -> "Nothing[T]":
        return self

    def is_some(self) -> Literal[False]:
        return False

    def is_nothing(self) -> Literal[True]:
        return True

    def unwrap(self) -> T:
        raise TypeError

    def unwrap_or_default(self, default: T) -> T:
        return default

    def __lt__(self, other: "Some[T] | Nothing[T]") -> bool:
        # annotating this with `Maybe[T]` breaks when `functools.total_ordering` is applied
        match other:
            case Some():
                return True
            case _:
                return False


Maybe = Some[T] | Nothing[T]
"""Maybe some value of type `T` or nothing."""

P = ParamSpec("P")


def maybe(func: Callable[P, T | None]) -> Callable[P, Maybe[T]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Maybe[T]:
        match result := func(*args, **kwargs):
            case Nothing() | None:
                return Nothing()
            case _:
                return Some(result)

    return wrapper
