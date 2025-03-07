__all__ = ("Enumerable", "enumerable")


import operator
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import InitVar, dataclass, field
from functools import reduce, wraps
from itertools import chain, tee
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec, TypeVar, cast, final, overload

from functional_stuff.monads.base import AbstractMonad, T, U

if TYPE_CHECKING:
    from functional_stuff.utils.protocols import ComparableT

K = TypeVar("K", bound=object)
P = ParamSpec("P")
Predicate = Callable[[T], bool]


def preserve(func: Callable[Concatenate["Enumerable[T]", P], U]) -> Callable[Concatenate["Enumerable[T]", P], U]:
    """Enables a consuming method to preserve its elements if `preserve=True` is passed.

    Replaces the underlying iterable with `functools.tee` if it is an `Iterator` (i.e. items
    will probably get consumed) and `preserve=True`, otherwise assumes that the inner iterable
    is some form of non consuming collection type and does nothing.
    """

    @wraps(func)
    def wrapper(instance: "Enumerable[T]", *args: P.args, **kwargs: P.kwargs) -> U:
        preserve = kwargs.get("preserve")
        match (instance._iterable, preserve):  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
            case (Iterator() as iterable, True):
                consumed, preserved = tee(iterable)
                instance._iterable = consumed  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
                result = func(instance, *args, **kwargs)
                instance._iterable = preserved  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
                return result
            case _:
                return func(instance, *args, **kwargs)

    return wrapper


@final
@dataclass(slots=True)
class Enumerable(Iterable[T], AbstractMonad[T]):
    """A LINQ inspired `Iterable` over type `T`."""

    iterable: InitVar[Iterable[T]] = tuple[T]()
    _iterable: Iterable[T] = field(init=False)

    # region base

    def __post_init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[T]:
        return iter(self._iterable)

    def map(self, func: Callable[[T], U]) -> "Enumerable[U]":
        """Proxies `select`."""
        return self.select(func)

    def bind(self, func: Callable[[T], "EnumerableT"]) -> "EnumerableT":
        """Proxies `select_many`."""
        return cast("EnumerableT", self.select_many(func))

    def join(self: "Enumerable[EnumerableT]") -> "EnumerableT":
        """Proxies `select_many` for inner `Enumerable`s using an identity function."""
        return cast("EnumerableT", self.select_many(lambda x: x))

    # endregion

    def select(self, selector: Callable[[T], U]) -> "Enumerable[U]":
        """Projects each value of the underlying iterable into a new form of `U`."""
        return Enumerable(selector(x) for x in self)

    def select_many(self, selector: Callable[[T], Iterable[U]]) -> "Enumerable[U]":
        """Projects each value of the underlying iterable into a new form of `U` and flattens the result."""
        return Enumerable(chain.from_iterable(selector(x) for x in self))

    def where(self, predicate: Predicate[T]) -> "Enumerable[T]":
        """Filters the underlying iterable base on `predicate`."""
        return Enumerable(x for x in self if predicate(x))

    @overload
    def any(self, predicate: None, *, preserve: bool = False) -> bool: ...

    @overload
    def any(self, predicate: None = None, *, preserve: bool = False) -> bool: ...

    @overload
    def any(self, predicate: Predicate[T], *, preserve: bool = False) -> bool: ...

    @preserve
    def any(self, predicate: Predicate[T] | None = None, *, preserve: bool = False) -> bool:  # noqa: ARG002
        """Return `True` if any element satisfies the predicate or if any element exists."""
        match predicate:
            case Callable():
                return any(predicate(x) for x in self)
            case _:
                for _ in self:
                    return True
                return False

    @preserve
    def all(self, predicate: Predicate[T], *, preserve: bool = False) -> bool:  # noqa: ARG002
        """Return `True` if all elements satisfy the predicate."""
        return all(predicate(x) for x in self)

    def contains(self, element: T, *, comparer: Literal["eq", "is"] = "eq", preserve: bool = False) -> bool:
        """Returns `True` if the enumerable contains the given element.

        Dispatches `Enumerable.any` using a predicate that either checks for equality `eq` or identity `is`.
        """
        match comparer:
            case "eq":
                return self.any(lambda x: operator.eq(x, element), preserve=preserve)
            case "is":
                return self.any(lambda x: operator.is_(x, element), preserve=preserve)

    @preserve
    def min(self: "Enumerable[ComparableT]", *, preserve: bool = False) -> "ComparableT":  # noqa: ARG002
        """Returns the smallest element of the enumerable."""
        return min(self)

    @preserve
    def min_by(self, key: Callable[[T], "ComparableT"], *, preserve: bool = False) -> T:  # noqa: ARG002
        """Returns the smallest element of the enumerable determined by the given key selector."""
        return min(self, key=key)

    @preserve
    def max(self: "Enumerable[ComparableT]", *, preserve: bool = False) -> "ComparableT":  # noqa: ARG002
        """Returns the largest element of the enumerable."""
        return max(self)

    @preserve
    def max_by(self, key: Callable[[T], "ComparableT"], *, preserve: bool = False) -> T:  # noqa: ARG002
        """Returns the largest element of the enumerable determined by the given key selector."""
        return max(self, key=key)

    @overload
    def aggregate(self, reducer: Callable[[T, T], T]) -> T: ...

    @overload
    def aggregate(self, reducer: Callable[[T, T], T], initial: None) -> T: ...

    @overload
    def aggregate(self, reducer: Callable[[U, T], U], initial: U) -> U: ...

    def aggregate(self, reducer: Callable[[U, T], U] | Callable[[T, T], T], initial: U | None = None) -> U | T:
        """Aggregates the underlying iterable using the given `reducer`."""
        match initial:
            case None:
                reducer = cast(Callable[[T, T], T], reducer)
                return reduce(reducer, self)
            case _:
                reducer = cast(Callable[[U, T], U], reducer)
                return reduce(reducer, self, initial)

    def order(self: "Enumerable[ComparableT]", *, reverse: bool = False) -> "Enumerable[ComparableT]":
        """Orders the elements of the underlying enumerable in ascending order."""
        return Enumerable(sorted(self, reverse=reverse))

    def order_by(self: "Enumerable[T]", key: Callable[[T], "ComparableT"], *, reverse: bool = False) -> "Enumerable[T]":
        """Orders the elements of the underlying enumerable in ascending order using the given key selector.

        `Enumerable` does not implement methods like `order_by_desc` and `then_by` or `then_by_desc`.

        To order an `Enumerable` in descending order, the `reverse` keyword can be used. To order by one or more
        properties use either a key function that returns a `tuple` of those properties or use multiple serial
        calls to `order_by` to make use of Pythons stable sort.
        """
        return Enumerable(sorted(self, key=key, reverse=reverse))

    def distinct(self) -> "Enumerable[T]":
        """Returns an enumerable of distinct elements."""
        container = set[T]()
        return Enumerable(x for x in self if x not in container and not container.add(x))

    def distinct_by(self, selector: Callable[[T], U]) -> "Enumerable[T]":
        """Returns an enumerable of distinct elements determined using the given key selector."""
        container = set[U]()
        return Enumerable(x for x in self if (y := selector(x)) not in container and not container.add(y))

    def prepend(self, *elements: T) -> "Enumerable[T]":
        """Adds one or more elements to the beginning of the enumerable."""
        return Enumerable(chain(elements, self))

    def append(self, *elements: T) -> "Enumerable[T]":
        """Adds one or more elements to the end of the enumerable."""
        return Enumerable(chain(self, elements))

    def concat(self, *iterables: Iterable[T]) -> "Enumerable[T]":
        """Adds the elements of one or more iterables to the end of the enumerable."""
        return Enumerable(chain(self, chain.from_iterable(iterables)))

    def zip(self, iterable: Iterable[U], *, strict: bool = False) -> "Enumerable[tuple[T, U]]":
        """Combines two iterables to a single enumerable of tuples.

        Proxies `zip` and therefore raises `ValueError` if one iterable has less elements than
        the other if `strict=True`, otherwise combines elements until the first iterable is exhausted.
        """
        return Enumerable(zip(self, iterable, strict=strict))

    @classmethod
    def empty(cls) -> "Enumerable[T]":
        """Creates an empty enumerable."""
        return cls()

    # region conversion

    def cast(self, dtype: type[U]) -> "Enumerable[U]":  # noqa: ARG002
        """Proxies `typing.cast`, should be used with caution."""
        return cast("Enumerable[U]", self)

    def of_type(self, dtype: type[U]) -> "Enumerable[U]":
        """Filters the underlying iterable using `isinstance`, internally chains `where` and `cast`."""
        return self.where(lambda x: isinstance(x, dtype)).cast(dtype)

    def to_deque(self) -> deque[T]:
        """Creates a new `deque[T]` by consuming the underlying iterable."""
        return deque(self)

    def to_list(self) -> list[T]:
        """Creates a new `list[T]` by consuming the underlying iterable."""
        return list(self)

    def to_set(self) -> set[T]:
        """Creates a new `set[T]` by consuming the underlying iterable."""
        return set(self)

    @overload
    def to_dict(self, key: Callable[[T], K]) -> dict[K, T]: ...

    @overload
    def to_dict(self, key: Callable[[T], K], func: None) -> dict[K, T]: ...

    @overload
    def to_dict(self, key: Callable[[T], K], func: Callable[[T], U]) -> dict[K, U]: ...

    def to_dict(self, key: Callable[[T], K], func: Callable[[T], U] | None = None) -> dict[K, U] | dict[K, T]:
        """Creates a new `dict[K, U]` or `dict[K, T]` by consuming the underlying iterable."""
        match func:
            case Callable():
                return {key(x): func(x) for x in self}
            case _:
                return {key(x): x for x in self}

    def preserve(self) -> "Enumerable[T]":
        """Returns a new enumerable with its elements unwrapped into a collection type.

        Converts an inner iterator into a static collection type (using `Enumerable.to_list`) which allows
        multiple iterations over its contents without consuming any of its elements.

        The effect of this method is similar to the `preserve` decorator and therefore any method that uses the
        preserve keyword. But unlike those which rely on `itertools.tee` to only cache consumed elements, this
        method unwraps the whole graph at once which may consume more memory.
        """
        match self._iterable:
            case Iterator():
                return Enumerable(self.to_list())
            case _:
                return self

    # endregion


EnumerableT = TypeVar("EnumerableT", bound=Enumerable[Any])


@overload
def enumerable(func: Callable[P, Iterable[T]]) -> Callable[P, Enumerable[T]]: ...


@overload
def enumerable(func: Callable[P, Iterable[T] | None]) -> Callable[P, Enumerable[T] | None]: ...


def enumerable(func: Callable[P, Iterable[T] | None]) -> Callable[P, Enumerable[T] | None]:
    """Wraps the result of a function in an `Enumerable` monad."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Enumerable[T] | None:
        match result := func(*args, **kwargs):
            case Iterable():
                return Enumerable(result)
            case _:
                return result

    return wrapper
