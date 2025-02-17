from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=object)
U = TypeVar("U", bound=object)


class AbstractMonad(ABC, Generic[T]):
    @abstractmethod
    def bind(self, func: Callable[[T], U]) -> "AbstractMonad[U]": ...

    @abstractmethod
    def join(self: "AbstractMonad[AbstractMonad[U]]") -> "AbstractMonad[U]": ...


M = TypeVar("M", bound=AbstractMonad[Any])
