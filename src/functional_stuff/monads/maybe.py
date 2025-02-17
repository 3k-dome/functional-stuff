__all__ = ("Maybe", "Nothing", "Some", "to_maybe")

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, NoReturn, ParamSpec, TypeGuard, final

from functional_stuff.monads.base import AbstractMonad, M, T, U

if TYPE_CHECKING:
    from functional_stuff.monads.result import Error, Ok, Result

P = ParamSpec("P")


class AbstractMaybe(AbstractMonad[T]):
    @abstractmethod
    def is_some(self) -> bool: ...

    @abstractmethod
    def is_nothing(self) -> bool: ...

    @abstractmethod
    def unwrap(self) -> T: ...

    @abstractmethod
    def unwrap_or_default(self, default: T) -> T: ...

    @abstractmethod
    def to_result(self) -> "Result[T, TypeError]": ...

    @abstractmethod
    def __bool__(self) -> bool: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __repr__(self) -> str: ...


@final
@dataclass(slots=True, frozen=True)
class Some(AbstractMaybe[T]):
    value: T

    def bind(self, func: Callable[[T], U | None]) -> "Maybe[U]":
        match result := func(self.value):
            case Nothing() | None:
                return Nothing()
            case _:
                return Some(result)

    def join(self: "Some[M]") -> "M":
        return self.value

    def is_some(self) -> Literal[True]:
        return True

    def is_nothing(self) -> Literal[False]:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or_default(self, default: T) -> T:  # noqa: ARG002
        return self.value

    def to_result(self) -> "Ok[T]":
        from functional_stuff.monads.result import Ok

        return Ok(self.value)

    def __bool__(self) -> Literal[True]:
        return self.is_some()

    def __eq__(self, other: object) -> "TypeGuard[Some[T]]":
        match other:
            case Some(value):  # pyright: ignore[reportUnknownVariableType]
                return value == self.value  # pyright: ignore[reportUnknownVariableType]
            case _:
                return False

    def __repr__(self) -> str:
        return f"Some({self.value!r})"


@final
@dataclass(slots=True, frozen=True)
class Nothing(AbstractMaybe[NoReturn]):
    def bind(self, func: Callable[[Any], U]) -> "Nothing":  # noqa: ARG002
        return self

    def join(self) -> "Nothing":
        return self

    def is_some(self) -> Literal[False]:
        return False

    def is_nothing(self) -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        error = "An instance of Nothing can't be unwrapped."
        raise TypeError(error)

    def unwrap_or_default(self, default: T) -> T:
        return default

    def to_result(self) -> "Error[TypeError]":
        from functional_stuff.monads.result import Error

        error = "An instance of Nothing can't be unwrapped."
        return Error(TypeError(error))

    def __bool__(self) -> Literal[False]:
        return self.is_some()

    def __eq__(self, other: object) -> "TypeGuard[Nothing]":
        match other:
            case Nothing():
                return True
            case _:
                return False

    def __repr__(self) -> str:
        return "Nothing()"


Maybe = Some[T] | Nothing


def to_maybe(func: Callable[P, T | None]) -> Callable[P, Maybe[T]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Maybe[T]:
        match result := func(*args, **kwargs):
            case Nothing() | None:
                return Nothing()
            case _:
                return Some(result)

    return wrapper
