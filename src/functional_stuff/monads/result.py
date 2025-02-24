__all__ = ("Error", "Ok", "Result", "result")

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import total_ordering, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    final,
)

if TYPE_CHECKING:
    from functional_stuff.monads.maybe import Maybe

from functional_stuff.monads.base import AbstractMonad, MonadT, T, U


class AbstractResult(AbstractMonad[T]):
    """Abstract base for a result monad."""

    @abstractmethod
    def is_ok(self) -> bool:
        """Returns `True` if the result is ok."""
        ...

    @abstractmethod
    def is_error(self) -> bool:
        """Returns `True` if the result is an exception."""
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the contained result if ok, else raises the contained error."""
        ...

    @abstractmethod
    def unwrap_or_default(self, default: T) -> T:
        """Returns the contained result if ok, else default."""
        ...

    @abstractmethod
    def inspect(self, func: Callable[[T], None]) -> "Self":
        """Calls a function with the contained value if `is_ok`."""
        ...

    @abstractmethod
    def inspect_error(self, func: Callable[[Exception], None]) -> "Self":
        """Calls a function with the contained value if `is_error`."""
        ...

    @abstractmethod
    def maybe(self) -> "Maybe[T]":
        """Converts to `Maybe[T]`, replaces any error with `Nothing`."""
        ...

    @abstractmethod
    def __lt__(self, other: "Result[T]") -> bool: ...


ResultT = TypeVar("ResultT", bound="AbstractResult[Any]")


@final
@total_ordering
@dataclass(slots=True, frozen=True)
class Ok(AbstractResult[T]):
    """A success along with a value of type `T`.

    Being implemented as frozen dataclass automatically provides methods like `__str__` and `__repr__`,
    as well as tuple-based `__eq__` and `__hash__` methods. An `__lt__` method, which proxies the `__lt__`
    operator of the inner value (if ok), is also provided and is used to apply `functools.total_ordering`.
    An instance of `Error` is **always** less than an instance `Ok`.
    """

    value: T

    def map(self, func: Callable[[T], U]) -> "Result[U]":
        try:
            return Ok(func(self.value))
        except Exception as error:  # noqa: BLE001
            return Error(error)

    def bind(self, func: Callable[[T], ResultT]) -> ResultT:
        try:
            return func(self.value)
        except Exception as error:  # noqa: BLE001
            return cast(ResultT, Error(error))

    def join(self: "Ok[MonadT]") -> MonadT:
        return self.value

    def is_ok(self) -> Literal[True]:
        return True

    def is_error(self) -> Literal[False]:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or_default(self, default: T) -> T:  # noqa: ARG002
        return self.value

    def inspect(self, func: Callable[[T], None]) -> Self:
        func(self.value)
        return self

    def inspect_error(self, func: Callable[[Exception], None]) -> Self:  # noqa: ARG002
        return self

    def maybe(self) -> "Maybe[T]":
        from functional_stuff.monads import Some

        return Some(self.value)

    def __lt__(self, other: "Ok[T] | Error[T]") -> bool:
        # annotating this with `Result[T]` breaks when `functools.total_ordering` is applied
        match other:
            case Ok(value):
                return cast(bool, self.value < value)  # pyright: ignore[reportOperatorIssue]
            case _:
                return False


@final
@dataclass(frozen=True, slots=True)
class Error(AbstractResult[T]):
    """An error instead of a value of type `T`.

    Being implemented as frozen dataclass automatically provides methods like `__str__` and `__repr__`,
    as well as tuple-based `__eq__` and `__hash__` methods. An `__lt__` method, which proxies the `__lt__`
    operator of the inner value (if ok), is also provided and is used to apply `functools.total_ordering`.
    An instance of `Error` is **always** less than an instance `Ok`.
    """

    error: Exception

    def map(self, func: Callable[[T], U]) -> "Result[U]":  # noqa: ARG002
        return cast(Error[U], self)

    def bind(self, func: Callable[[T], ResultT]) -> ResultT:  # noqa: ARG002
        return cast(ResultT, self)

    def join(self) -> "Error[T]":
        return self

    def is_ok(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        raise self.error

    def unwrap_or_default(self, default: T) -> T:
        return default

    def inspect(self, func: Callable[[T], None]) -> Self:  # noqa: ARG002
        return self

    def inspect_error(self, func: Callable[[Exception], None]) -> Self:
        func(self.error)
        return self

    def maybe(self) -> "Maybe[T]":
        from functional_stuff.monads import Nothing

        return Nothing[T]()

    def __lt__(self, other: "Ok[T] | Error[T]") -> bool:
        # annotating this with `Result[T]` breaks when `functools.total_ordering` is applied
        match other:
            case Ok():
                return True
            case _:
                return False


Result = Ok[T] | Error[T]
"""A success with a value of type `T` or an error."""

P = ParamSpec("P")


def result(func: Callable[P, T]) -> Callable[P, Result[T]]:
    """Wraps the result of a function in a `Result` monad."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T]:
        try:
            result = func(*args, **kwargs)
            return Ok(result)
        except Exception as error:  # noqa: BLE001
            return Error(error)

    return wrapper
