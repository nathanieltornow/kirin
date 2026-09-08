from typing import Any, Literal

import pytest

from kirin import ir, types
from kirin.prelude import structural_no_opt
from kirin.rewrite import Walk
from kirin.dialects import py, scf, func, debug, ilist
from kirin.dialects.ilist.rewrite.to_loop import MapToForLoop


def lower(method: ir.Method[..., Any]) -> None:
    result = Walk(MapToForLoop()).rewrite(method.code)
    assert result.has_done_something
    assert not result.exceeded_max_iter
    method.verify()
    assert not any(isinstance(stmt, ilist.Map) for stmt in method.code.walk())
    repeated = Walk(MapToForLoop()).rewrite(method.code)
    assert not repeated.has_done_something
    assert not repeated.exceeded_max_iter


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], []), ([3], [7]), ([1, 2, 3], [3, 5, 7]), ([-2, 0, 4], [-3, 1, 9])],
)
def test_map_to_loop(values: list[int], expected: list[int]) -> None:
    @structural_no_opt
    def transform(x: int) -> int:
        return 2 * x + 1

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[int, Any]:
        return ilist.map(transform, xs)

    xs = ilist.IList(values.copy(), elem=types.Int)
    assert list(mapped(xs)) == expected
    lower(mapped)
    assert list(mapped(xs)) == expected
    assert list(xs) == values
    assert sum(isinstance(stmt, scf.For) for stmt in mapped.code.walk()) == 1


def test_map_preserves_element_and_length_types() -> None:
    @structural_no_opt
    def positive(x: int) -> bool:
        return x > 0

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Literal[3]]) -> ilist.IList[bool, Literal[3]]:
        return ilist.map(positive, xs)

    xs = ilist.IList([-1, 0, 1], elem=types.Int)
    assert list(mapped(xs)) == [False, False, True]
    lower(mapped)
    assert list(mapped(xs)) == [False, False, True]
    loop = next(stmt for stmt in mapped.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == ilist.IListType[types.Bool, types.Literal(3)]
    assert loop.body.blocks[0].args[1].type == ilist.IListType[types.Bool, types.Any]
    call = next(stmt for stmt in loop.body.walk() if isinstance(stmt, func.Call))
    assert call.result.type == types.Bool
    empty = loop.initializers[0]
    assert empty.type == ilist.IListType[types.Bool, types.Literal(0)]


def test_map_unknown_callback_return_type() -> None:
    @structural_no_opt
    def identity(x: int) -> Any:
        return x

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[Any, Any]:
        return ilist.map(identity, xs)

    xs = ilist.IList([2, 5], elem=types.Int)
    assert list(mapped(xs)) == [2, 5]
    lower(mapped)
    assert list(mapped(xs)) == [2, 5]
    loop = next(stmt for stmt in mapped.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == ilist.IListType[types.Any, types.Any]


@pytest.mark.parametrize("enabled", [False, True])
def test_map_inside_condition(enabled: bool) -> None:
    @structural_no_opt
    def increment(x: int) -> int:
        return x + 1

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any], enabled: bool) -> ilist.IList[int, Any]:
        if enabled:
            ys = ilist.map(increment, xs)
        else:
            ys = xs
        return ys

    xs = ilist.IList([1, 4], elem=types.Int)
    expected = [2, 5] if enabled else [1, 4]
    assert list(mapped(xs, enabled)) == expected
    lower(mapped)
    assert list(mapped(xs, enabled)) == expected


@pytest.mark.parametrize("count", [0, 1, 3])
def test_map_inside_loop(count: int) -> None:
    @structural_no_opt
    def increment(x: int) -> int:
        return x + 1

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any], count: int) -> ilist.IList[int, Any]:
        for _ in range(count):
            xs = ilist.map(increment, xs)
        return xs

    xs = ilist.IList([1, 4], elem=types.Int)
    expected = [1 + count, 4 + count]
    assert list(mapped(xs, count)) == expected
    lower(mapped)
    assert list(mapped(xs, count)) == expected


def test_map_captured_value() -> None:
    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any], offset: int) -> ilist.IList[int, Any]:
        def shift(x: int) -> int:
            return x + offset

        return ilist.map(shift, xs)

    xs = ilist.IList([1, 4], elem=types.Int)
    assert list(mapped(xs, 7)) == [8, 11]
    lower(mapped)
    assert list(mapped(xs, 7)) == [8, 11]


@pytest.mark.parametrize("values", [[], [3, 1, 3]])
def test_map_callback_effects(
    values: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    dialects = structural_no_opt.add(debug)

    @dialects
    def emit(x: int) -> int:
        debug.info("map", x)
        return x + 1

    @dialects
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[int, Any]:
        return ilist.map(emit, xs)

    xs = ilist.IList(values, elem=types.Int)
    assert list(mapped(xs)) == [x + 1 for x in values]
    output = capsys.readouterr().out
    observed = [
        int(line.rsplit(" = ", 1)[1]) for line in output.splitlines() if " = " in line
    ]
    assert observed == values
    lower(mapped)
    assert list(mapped(xs)) == [x + 1 for x in values]
    output = capsys.readouterr().out
    observed = [
        int(line.rsplit(" = ", 1)[1]) for line in output.splitlines() if " = " in line
    ]
    assert observed == values


def test_consecutive_maps() -> None:
    @structural_no_opt
    def increment(x: int) -> int:
        return x + 1

    @structural_no_opt
    def double(x: int) -> int:
        return 2 * x

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[int, Any]:
        return ilist.map(double, ilist.map(increment, xs))

    xs = ilist.IList([1, 3], elem=types.Int)
    assert list(mapped(xs)) == [4, 8]
    lower(mapped)
    assert list(mapped(xs)) == [4, 8]
    assert sum(isinstance(stmt, scf.For) for stmt in mapped.code.walk()) == 2


def test_non_map_unchanged() -> None:
    result = MapToForLoop().rewrite(py.Constant(1))
    assert not result.has_done_something
