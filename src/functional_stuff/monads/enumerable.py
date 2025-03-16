__all__ = ("Enumerable", "enumerable")


import contextlib
import operator
import warnings
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Reversible, Sequence
from functools import reduce, wraps
from itertools import batched, chain, groupby, islice, takewhile, tee
from typing import TYPE_CHECKING, Any, Concatenate, Generic, Literal, ParamSpec, Self, TypeVar, cast, overload

from functional_stuff.monads.base import AbstractMonad, T, U

if TYPE_CHECKING:
    from functional_stuff.utils.protocols import Comparable, ComparableT

K = TypeVar("K", bound=object)
P = ParamSpec("P")
Predicate = Callable[[T], bool]


def preserve(func: Callable[Concatenate["Enumerable[Any]", P], U]) -> Callable[Concatenate["Enumerable[Any]", P], U]:
    """Enables a consuming method to preserve its elements if `preserve=True` is passed.

    Replaces the underlying iterable with `functools.tee` if it is an `Iterator` (i.e. items
    will probably get consumed) and `preserve=True`, otherwise assumes that the inner iterable
    is some form of non consuming collection type and does nothing.
    """

    if "preserve" not in func.__annotations__:
        warning = f"Preserved function does not have an `preserve` key word, {func=}."
        warnings.warn(warning, SyntaxWarning, stacklevel=2)

    @wraps(func)
    def wrapper(instance: "Enumerable[Any]", *args: P.args, **kwargs: P.kwargs) -> U:
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


class Enumerable(Iterable[T], AbstractMonad[T]):
    """A LINQ inspired `Iterable` over type `T`."""

    __slots__ = ("_iterable",)

    # region base

    def __init__(self, iterable: Iterable[T] = tuple[T]()) -> None:
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

    @classmethod
    def empty(cls) -> "Enumerable[T]":
        """Creates an empty enumerable."""
        return cls()

    # endregion

    def select(self, selector: Callable[[T], U]) -> "Enumerable[U]":
        """Projects each value of the enterable into a new form of `U`."""
        return Enumerable(selector(x) for x in self)

    def select_many(self, selector: Callable[[T], Iterable[U]]) -> "Enumerable[U]":
        """Projects each value of the enterable into a new form of `U` and flattens the result."""
        return Enumerable(chain.from_iterable(selector(x) for x in self))

    def where(self, predicate: Predicate[T]) -> "Enumerable[T]":
        """Filters the enumerable based on `predicate`."""
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

    @preserve
    def contains(self, element: T, *, comparer: Literal["eq", "is"] = "eq", preserve: bool = False) -> bool:
        """Returns `True` if the enumerable contains the given element.

        Dispatches `Enumerable.any` using a predicate that either checks for equality `eq` or identity `is`.
        """
        match comparer:
            case "eq":
                return self.any(lambda x: operator.eq(x, element), preserve=preserve)
            case "is":
                return self.any(lambda x: operator.is_(x, element), preserve=preserve)

    def reverse(self) -> "Enumerable[T]":
        """Reverses the enumerable.

        Simply reverses the underlying iterable if possible, otherwise unwraps it and than reverses it.
        """
        match self._iterable:
            case Reversible() as reversible:
                return Enumerable(reversed(reversible))
            case _:
                container = self.to_tuple()
                return Enumerable(reversed(container))

    def first(self, predicate: Predicate[T] | None = None) -> T:
        """Returns the first element (matching `predicate`) of the enterable."""
        match predicate:
            case Callable():
                return next(x for x in self if predicate(x))
            case _:
                return next(x for x in self)

    def first_or_default(self, default: T, predicate: Predicate[T] | None = None) -> T:
        """Returns the first element (matching `predicate`) of the enterable, or `default`."""
        match predicate:
            case Callable():
                return next((x for x in self if predicate(x)), default)
            case _:
                return next(x for x in self)

    def element_at(self, index: int) -> T:
        """Returns the element at `index`."""
        with contextlib.suppress(StopIteration, IndexError):
            match self._iterable:
                case Sequence():
                    return self._iterable[index]
                case _:
                    return next(x for i, x in (enumerate(self)) if i == index)
        error = f"Enumerable has no element at index, {index=}."
        raise IndexError(error)

    def element_at_or_default(self, index: int, default: T) -> T:
        """Returns the element at `index` or `default`."""
        try:
            return self.element_at(index)
        except IndexError:
            return default

    def single(self, predicate: Predicate[T] | None) -> T:
        """Returns the only element of the enumerable."""
        container = self.where(predicate) if isinstance(predicate, Callable) else self
        match (container.count(preserve=True), predicate):
            case (1, _):
                return container.first()
            case (0, None):
                error = "Enumerable is empty."
            case (0, _):
                error = f"Enumerable does not contain any one element matching predicate, {predicate=}."
            case (_, None):
                error = "Enumerable contains more than a one element."
            case _:
                error = f"Enumerable contains more than one element matching predicate, {predicate=}."
        raise ValueError(error)

    def single_or_default(self, default: T, predicate: Predicate[T]) -> T:
        """Returns the only element of the enumerable or `default`."""
        try:
            return self.single(predicate)
        except ValueError:
            return default

    def last(self, predicate: Predicate[T] | None = None) -> T:
        """Returns the last element (matching `predicate`) of the enterable.

        Internally reverses the enumerable, may consume any underlying iterators.
        """
        return self.reverse().first(predicate)

    def last_or_default(self, default: T, predicate: Predicate[T] | None = None) -> T:
        """Returns the last element (matching `predicate`) of the enterable, or `default`.

        Internally reverses the enumerable, may consume any underlying iterators.
        """
        return self.reverse().first_or_default(default, predicate)

    def take(self, count: int) -> "Enumerable[T]":
        """Returns up to `count` elements from the enterable."""
        return Enumerable(islice(self, count))

    def take_last(self, count: int) -> "Enumerable[T]":
        """Returns up to `count` elements from the back of the enterable.

        Internally reverses the enumerable, may consume any underlying iterators.
        """
        return self.reverse().take(count)

    def take_while(self, predicate: Predicate[T]) -> "Enumerable[T]":
        """Returns elements from the enterable as long as `predicate` evaluates to `True`."""
        return Enumerable(takewhile(predicate, self))

    def skip(self, count: int) -> "Enumerable[T]":
        """Returns an enumerable without the first `count` elements."""
        return Enumerable(islice(self, count, None))

    def skip_last(self, count: int) -> "Enumerable[T]":
        """Returns an enumerable without the last `count` elements."""

        if isinstance(self._iterable, Sequence):
            return Enumerable(self._iterable[:-count])

        # shift and count and iterator of self
        container = deque[T](maxlen=count)
        iterator = iter(self)
        for _ in range(count):
            try:
                container.append(next(iterator))
            except StopIteration:
                return Enumerable[T].empty()

        # yield form the shifted iterator
        def shifted() -> Iterator[T]:
            for element in iterator:
                yield container.popleft()
                container.append(element)

        return Enumerable(shifted())

    def skip_while(self, predicate: Predicate[T]) -> "Enumerable[T]":
        """Returns an enumerable skipping elements until `predicate` equals `False` for the first time."""
        take = False
        return Enumerable(x for x in self if (take := take or not predicate(x)))

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

    @preserve
    def count(self, *, preserve: bool = False) -> int:  # noqa: ARG002
        """Returns the number of elements in the enumerable."""
        match self._iterable:
            case Collection():
                return len(self._iterable)
            case _:
                return sum(1 for _ in self)

    @overload
    def sum(self: "Enumerable[int]", *, preserve: bool = False) -> int: ...

    @overload
    def sum(self: "Enumerable[int | float]", *, preserve: bool = False) -> float: ...

    @preserve
    def sum(self: "Enumerable[int] | Enumerable[int | float]", *, preserve: bool = False) -> int | float:  # noqa: ARG002
        """Returns the sum of all elements."""
        return sum(self)

    @preserve
    def sum_by(self, key: Callable[[T], int] | Callable[[T], int | float], *, preserve: bool = False) -> int | float:  # noqa: ARG002
        """Returns the sum of all elements by `key`."""
        return sum(key(x) for x in self)

    @preserve
    def average(self: "Enumerable[int | float]", *, preserve: bool = False) -> float:  # noqa: ARG002
        """Returns the average of all elements."""
        length = 0
        return Enumerable(x for count, x in enumerate(self, 1) if (length := count)).sum() / length

    @preserve
    def average_by(self, key: Callable[[T], int | float], *, preserve: bool = False) -> float:  # noqa: ARG002
        """Returns the average of all elements by `key`."""
        length = 0
        return Enumerable(key(x) for count, x in enumerate(self, 1) if (length := count)).sum() / length

    @overload
    def aggregate(self, reducer: Callable[[T, T], T]) -> T: ...

    @overload
    def aggregate(self, reducer: Callable[[T, T], T], initial: None) -> T: ...

    @overload
    def aggregate(self, reducer: Callable[[U, T], U], initial: U) -> U: ...

    def aggregate(self, reducer: Callable[[U, T], U] | Callable[[T, T], T], initial: U | None = None) -> U | T:
        """Aggregates the elements of this enumerable using the given `reducer`."""
        match initial:
            case None:
                reducer = cast(Callable[[T, T], T], reducer)
                return reduce(reducer, self)
            case _:
                reducer = cast(Callable[[U, T], U], reducer)
                return reduce(reducer, self, initial)

    def group(
        self: "Enumerable[ComparableT]",
        *,
        reverse: bool = False,
    ) -> "Enumerable[GroupedEnumerable[ComparableT, ComparableT]]":
        """Groups the elements of the enumerable, returning a new enumerable of keys and groups."""
        container = self.order(reverse=reverse)
        return Enumerable(GroupedEnumerable(group, key=key) for key, group in groupby(container))

    def group_by(
        self,
        key: Callable[[T], "ComparableT"],
        *,
        reverse: bool = False,
    ) -> "Enumerable[GroupedEnumerable[ComparableT, T]]":
        """Groups the elements of the enumerable by a given key, returning a new enumerable of keys and groups."""
        container = self.order_by(key, reverse=reverse)
        return Enumerable(GroupedEnumerable(group, key=key) for key, group in groupby(container, key=key))

    def order(self: "Enumerable[ComparableT]", *, reverse: bool = False) -> "Enumerable[ComparableT]":
        """Orders the elements of this enumerable in ascending order."""
        return Enumerable(sorted(self, reverse=reverse))

    def order_by(self, key: Callable[[T], "ComparableT"], *, reverse: bool = False) -> "OrderedEnumerable[T]":
        """Orders the elements of this enumerable in ascending order using the given primary key."""
        return OrderedEnumerable(self._iterable, key, reverse=reverse)

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

    def chunk(self, size: int) -> "Enumerable[Enumerable[T]]":
        """Returns an enumerable over subsets up to length `size` of this enumerable."""
        return Enumerable(Enumerable(x) for x in batched(self, size))

    def window(self, size: int, *, padding: T | None = None) -> "Enumerable[Enumerable[T]]":
        """Returns an enumerable of sliding windows over each element.

        If `padding is None` the window starts with a size of `1` and grows until `size` is reached
        before elements are dropped, e.g. `window((0, 1, 2, 3), 3)` => `(0,), (0, 1,), (0, 1, 2), (1, 2, 3)`.
        Otherwise the enumerable is padded (font & back) with `padding` so that each element is always centered
        within its window, e.g. `window((0, 1, 2, 3), 3, padding=-1)` => `(-1, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, -1)`
        """

        if not size & 1 or size < 0:
            error = f"Size must be and uneven integer greater than zero, {size=}."
            raise ValueError(error)

        window = deque[T](maxlen=size)

        if padding is None:
            return Enumerable(Enumerable(window) for x in self if not window.append(x))

        width = size // 2
        dummies = [padding] * width
        window.extend(dummies)
        window.extend(self.take(width))
        enumerable = self if isinstance(self._iterable, Iterator) else self.skip(width)
        enumerable = enumerable.concat(dummies)
        return Enumerable(Enumerable(window) for x in enumerable if not window.append(x))

    # region set-operations

    def difference(self, iterable: Iterable[T]) -> "Enumerable[T]":
        """Returns an enumerable containing the set difference of this and the given iterable."""
        container = set(iterable)
        return Enumerable(x for x in self if x not in container and not container.add(x))

    def difference_by(self, iterable: Iterable[T], key: Callable[[T], U]) -> "Enumerable[T]":
        """Returns an enumerable containing the set difference of this and the given iterable, determined by `key`."""
        container = {key(x) for x in iterable}
        return Enumerable(x for x in self if (y := key(x)) not in container and not container.add(y))

    def intersection(self, iterable: Iterable[T]) -> "Enumerable[T]":
        """Returns an enumerable containing the set intersection of this and the given iterable."""
        container = set(iterable)
        return Enumerable(x for x in self if x in container)

    def intersection_by(self, iterable: Iterable[T], key: Callable[[T], U]) -> "Enumerable[T]":
        """Returns an enumerable containing the set intersection of this and the given iterable, determined by `key`."""
        container = {key(x) for x in iterable}
        return Enumerable(x for x in self if key(x) in container)

    def union(self, iterable: Iterable[T]) -> "Enumerable[T]":
        """Returns an enumerable containing the set union of this and the given iterable."""
        return self.concat(iterable).distinct()

    def union_by(self, iterable: Iterable[T], key: Callable[[T], U]) -> "Enumerable[T]":
        """Returns an enumerable containing the set union of this and the given iterable, determined by `key`."""
        return self.concat(iterable).distinct_by(key)

    # endregion

    def equals(self, iterable: Iterable[U], *, preserve: bool = False) -> bool:
        """Returns `True` if both iterables iterate over the same element."""
        try:
            return self.zip(iterable, strict=True).all(lambda x: x[0] == x[1], preserve=preserve)
        except ValueError:
            return False

    # region type-conversion

    def cast(self, dtype: type[U]) -> "Enumerable[U]":  # noqa: ARG002
        """Proxies `typing.cast`, should be used with caution."""
        return cast("Enumerable[U]", self)

    def of_type(self, dtype: type[U]) -> "Enumerable[U]":
        """Filters the enumerable a given type."""
        return self.where(lambda x: isinstance(x, dtype)).cast(dtype)

    def to_deque(self) -> deque[T]:
        """Creates a new `deque[T]` by consuming the enumerable."""
        return deque(self)

    def to_list(self) -> list[T]:
        """Creates a new `list[T]` by consuming the enumerable."""
        return list(self)

    def to_set(self) -> set[T]:
        """Creates a new `set[T]` by consuming the enumerable."""
        return set(self)

    def to_tuple(self) -> tuple[T, ...]:
        """Creates a new `tuple[T]` by consuming the enumerable."""
        return tuple(self)

    @overload
    def to_dict(self, key: Callable[[T], K]) -> dict[K, T]: ...

    @overload
    def to_dict(self, key: Callable[[T], K], func: None) -> dict[K, T]: ...

    @overload
    def to_dict(self, key: Callable[[T], K], func: Callable[[T], U]) -> dict[K, U]: ...

    def to_dict(self, key: Callable[[T], K], func: Callable[[T], U] | None = None) -> dict[K, U] | dict[K, T]:
        """Creates a new `dict[K, U]` or `dict[K, T]` by consuming the enumerable."""
        match func:
            case Callable():
                return {key(x): func(x) for x in self}
            case _:
                return {key(x): x for x in self}

    # endregion

    def freeze(self) -> "Enumerable[T]":
        """Returns a new enumerable with its elements unwrapped into a collection type.

        Converts an inner iterator into a static collection type (using `Enumerable.to_list`) which allows
        multiple iterations over its contents without consuming any of its elements.

        The effect of this method is similar to the `preserve` decorator and therefore any method that uses the
        preserve keyword. But unlike those which rely on `itertools.tee` to only cache consumed elements, this
        method unwraps the whole graph at once which may consume more memory.
        """
        match self._iterable:
            case Iterator():
                return Enumerable(self.to_tuple())
            case _:
                return self


class OrderedEnumerable(Enumerable[T]):
    """Wraps an `order_by` operation to allow ordering by secondary `key`s."""

    __slots__ = ("_order",)

    def __init__(self, iterable: Iterable[T], key: Callable[[T], "ComparableT"], *, reverse: bool) -> None:
        super().__init__(iterable)
        self._order = deque[tuple[Callable[[T], "Comparable"], bool]]()
        self._order.append((key, reverse))

    def order_by(self, key: Callable[[T], "ComparableT"], *, reverse: bool = False) -> Self:
        """Orders the elements of this enumerable in ascending order using the given primary key."""
        self._order.append((key, reverse))
        return self

    def then_by(self, key: Callable[[T], "ComparableT"], *, reverse: bool = False) -> Self:
        """Orders the elements of this enumerable in ascending order using the given secondary key."""
        self._order.appendleft((key, reverse))
        return self

    def __iter__(self) -> Iterator[T]:
        for key, reverse in self._order:
            self._iterable = sorted(self._iterable, key=key, reverse=reverse)
        return super().__iter__()


class GroupedEnumerable(Enumerable[T], Generic[U, T]):
    """Wraps a key and associated values as an Enumerable, result of `group` or `group_by` operations."""

    __slots__ = ("_key",)

    def __init__(self, iterable: Iterable[T], *, key: U) -> None:
        super().__init__(iterable)
        self._key = key

    @property
    def key(self) -> U:
        """Gets the key of the group."""
        return self._key


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
