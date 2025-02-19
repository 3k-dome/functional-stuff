from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=object)
U = TypeVar("U", bound=object)


class AbstractMonad(ABC, Generic[T]):
    @abstractmethod
    def bind(self, func: Callable[[T], U]) -> "AbstractMonad[U]": ...

    @abstractmethod
    def join(self: "AbstractMonad[AbstractMonad[U]]") -> "AbstractMonad[U]": ...


M = TypeVar("M", bound=AbstractMonad[Any])


class AbstractAsyncMonad(ABC, Generic[T]):
    """A monadic type that allows async bindings."""

    @abstractmethod
    async def bind_async(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "AbstractMonad[U]": ...


class AbstractDeferrableMonad(ABC, Generic[T]):
    """A monadic type that allows deferred bindings."""

    @abstractmethod
    def bind_deferred(self, func: Callable[[T], Coroutine[Any, Any, U]]) -> "AbstractDeferredMonad[T, T, U]": ...


V = TypeVar("V", bound=object)
W = TypeVar("W", bound=object)


class AbstractDeferredMonad(ABC, Generic[T, U, V]):
    """A deferred monadic type.

    A deferred monad tracks its bindings and executes them in serial using a single `wait` call.

    The generics `T`, `U` and `V` refer to the held monads generic type, the result type of the *previous
    deferred* binding and the result type of the *latest deferred* binding respectively.
    """

    @abstractmethod
    def bind(self, awaitable: Callable[[V], Coroutine[None, None, W]]) -> "AbstractDeferredMonad[T, V, W]": ...

    @abstractmethod
    async def wait(self) -> AbstractMonad[Any]: ...
