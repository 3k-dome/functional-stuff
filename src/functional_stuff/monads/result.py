__all__ = ("Error", "Ok", "Result", "to_result")

from abc import abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    ParamSpec,
    Self,
    TypeGuard,
    TypeVar,
    cast,
    final,
    overload,
)

from functional_stuff.monads.base import AbstractDeferrableMonad, AbstractDeferredMonad, AbstractMonad, M, T, U, V, W

if TYPE_CHECKING:
    from functional_stuff.monads.maybe import Maybe, Nothing, Some

E = TypeVar("E", bound=Exception)
F = TypeVar("F", bound=Exception)


P = ParamSpec("P")


class AbstractResult(AbstractMonad[T], AbstractDeferrableMonad[T]):
    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def is_error(self) -> bool: ...

    @abstractmethod
    def unwrap(self) -> T: ...

    @abstractmethod
    def unwrap_or_default(self, default: T | Any) -> T: ...  # noqa: ANN401

    @abstractmethod
    def inspect(self, func: Callable[[T], None]) -> "Self": ...

    @abstractmethod
    def inspect_error(self, func: Callable[[T], None]) -> "Self": ...

    @abstractmethod
    def maybe(self) -> "Maybe[T]": ...

    @abstractmethod
    def __bool__(self) -> bool: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __repr__(self) -> str: ...


@final
@dataclass(frozen=True, slots=True)
class Ok(AbstractResult[T]):
    value: T

    def bind(self, func: Callable[[T], U]) -> "Result[U, Exception]":
        try:
            value = func(self.value)
            return Ok(value)
        except Exception as error:  # noqa: BLE001
            return Error(error)

    def bind_deferred(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "DeferredOk[T, T, U]":
        return DeferredOk(self, func)

    async def bind_async(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "Result[U, Exception]":
        try:
            value = await func(self.value)
            return Ok(value)
        except Exception as error:  # noqa: BLE001
            return Error(error)

    @overload
    def join(self: "Ok[Ok[U]]") -> "Ok[U]": ...

    @overload
    def join(self: "Ok[Error[F]]") -> "Error[F]": ...

    def join(self: "Ok[M]") -> "M":
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

    def inspect_error(self, func: Any) -> Self:  # noqa: ANN401, ARG002
        return self

    def maybe(self) -> "Some[T]":
        from functional_stuff.monads.maybe import Some

        return Some(self.value)

    def __bool__(self) -> Literal[True]:
        return self.is_ok()

    def __eq__(self, other: object) -> "TypeGuard[Ok[T]]":
        match other:
            case Ok(value):  # pyright: ignore[reportUnknownVariableType]
                return value == self.value  # pyright: ignore[reportUnknownVariableType]
            case _:
                return False

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@final
@dataclass(frozen=True, slots=True)
class Error(AbstractResult[E]):
    error: E

    def bind(self, func: Callable[[T], U]) -> "Error[E]":  # noqa: ARG002
        return self

    def bind_deferred(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "DeferredError[E, T, U]":  # noqa: ARG002
        return DeferredError(self)

    async def bind_async(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "Error[E]":  # noqa: ARG002
        return self

    def join(self) -> "Error[E]":
        return self

    def is_ok(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[True]:
        return True

    def unwrap(self) -> NoReturn:
        raise self.error

    def unwrap_or_default(self, default: Any) -> Any:  # noqa: ANN401
        return default

    def inspect(self, func: Any) -> Self:  # noqa: ANN401, ARG002
        return self

    def inspect_error(self, func: Callable[[E], None]) -> Self:
        func(self.error)
        return self

    def maybe(self) -> "Nothing":
        from functional_stuff.monads.maybe import Nothing

        return Nothing()

    def __bool__(self) -> Literal[False]:
        return self.is_ok()

    def __eq__(self, other: object) -> "TypeGuard[Error[E]]":
        match other:
            case Error(error):  # pyright: ignore[reportUnknownVariableType]
                return error == self.error  # pyright: ignore[reportUnknownVariableType]
            case _:
                return False

    def __repr__(self) -> str:
        return f"Error({self.error!r})"


Result = Ok[T] | Error[E]


def to_result(func: Callable[P, T]) -> Callable[P, Result[T, Exception]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
        try:
            result = func(*args, **kwargs)
            return Ok(result)
        except Exception as error:  # noqa: BLE001
            return Error(error)

    return wrapper


@final
class DeferredOk(AbstractDeferredMonad[T, U, V]):
    __slots__ = ("_coroutines", "_ok")

    def __init__(
        self,
        ok: Ok[T],
        awaitable: Callable[[U], Coroutine[None, None, V]],
        coroutines: tuple[Callable[..., Coroutine[None, None, Any]], ...] = (),
    ) -> None:
        self._ok = ok
        self._coroutines = (*coroutines, awaitable)

    def bind(self, awaitable: Callable[[V], Coroutine[None, None, W]]) -> "DeferredOk[T, V, W]":
        return DeferredOk(self._ok, awaitable, self._coroutines)

    async def wait(self) -> Result[V, Exception]:
        result: Result[Any, Exception] = self._ok
        for awaitable in self._coroutines:
            if not result:
                break

            try:
                value = await awaitable(result.value)
                result = Ok(value)
            except Exception as error:  # noqa: BLE001
                result = Error(error)

        return result


@final
class DeferredError(AbstractDeferredMonad[E, U, V]):
    __slots__ = ("_error",)

    def __init__(self, error: Error[E]) -> None:
        self._error = error

    def bind(self, awaitable: Callable[[V], Coroutine[None, None, W]]) -> "DeferredError[E, V, W]":  # noqa: ARG002
        return cast("DeferredError[E, V, W]", self)

    async def wait(self) -> Result[V, E]:
        return self._error


DeferredResult = DeferredOk[Any, Any, T] | DeferredError[E, Any, Any]


def to_result_async(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, Result[T, Exception]]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
        try:
            result = await func(*args, **kwargs)
            return Ok(result)
        except Exception as error:  # noqa: BLE001
            return Error(error)

    return wrapper
