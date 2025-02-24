from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=object)
U = TypeVar("U", bound=object)


class AbstractMonad(ABC, Generic[T]):
    """Abstract base for monadic types."""

    @abstractmethod
    def map(self, func: Callable[[T], U]) -> "AbstractMonad[U]":
        """Apply a function to the value inside the monad and return a new monad of same context."""
        ...

    @abstractmethod
    def bind(self, func: Callable[[T], "AbstractMonad[U]"]) -> "AbstractMonad[U]":
        """Apply a monadic function of same context to the value inside the monad and return is result."""
        ...

    @abstractmethod
    def join(self: "AbstractMonad[MonadT]") -> "MonadT":
        """Flattens a monad of monads into a single monad."""
        ...


MonadT = TypeVar("MonadT", bound="AbstractMonad[Any]")
