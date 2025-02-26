from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, TypeVar, cast, final

from functional_stuff.monads.base import AbstractMonad, T, U


@final
@dataclass(slots=True, frozen=True)
class Enumerable(Iterable[T], AbstractMonad[T]):
    """A LINQ inspired `Iterable` over type `T`."""

    iterable: Iterable[T] = tuple[T]()

    def map(self, func: Callable[[T], U]) -> "Enumerable[U]":
        """Proxies `select`."""
        return self.select(func)

    def bind(self, func: Callable[[T], "EnumerableT"]) -> "EnumerableT":
        """Proxies `select_many`."""
        return cast("EnumerableT", self.select_many(func))

    def join(self: "Enumerable[EnumerableT]") -> "EnumerableT":
        """Proxies `select_many` for inner `Enumerable`s using an identity function."""
        return cast("EnumerableT", self.select_many(lambda x: x))

    def __iter__(self) -> Iterator[T]:
        return iter(self.iterable)

    def select(self, selector: Callable[[T], U]) -> "Enumerable[U]":
        """Projects each value of the underlying iterable into a new form of `U`."""
        return Enumerable(selector(x) for x in self)

    def select_many(self, selector: Callable[[T], Iterable[U]]) -> "Enumerable[U]":
        """Projects each value of the underlying iterable into a new form of `U` and flattens the result."""
        return Enumerable(chain.from_iterable(selector(x) for x in self))


EnumerableT = TypeVar("EnumerableT", bound=Enumerable[Any])
