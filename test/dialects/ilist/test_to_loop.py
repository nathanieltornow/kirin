import re
from typing import Any, Literal

import pytest

from kirin import ir, types
from kirin.passes import TypeInfer
from kirin.prelude import structural_no_opt
from kirin.rewrite import Walk, Chain
from kirin.dialects import py, scf, func, debug, ilist
from kirin.dialects.ilist.rewrite.to_loop import (
    AllToForLoop,
    AnyToForLoop,
    MapToForLoop,
    FoldToForLoop,
    ScanToForLoop,
    ForEachToForLoop,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_DEBUG_VALUE = re.compile(r" = (-?\d+)")


def emitted_values(output: str) -> list[int]:
    """The integers `debug.info` printed, ignoring the printer's color escapes."""
    return [int(value) for value in _DEBUG_VALUE.findall(_ANSI_ESCAPE.sub("", output))]


def lower(method: ir.Method[..., Any]) -> tuple[types.TypeAttribute, ...]:
    TypeInfer(method.dialects, no_raise=False)(method)
    original_types = tuple(
        stmt.result.type for stmt in method.code.walk() if isinstance(stmt, ilist.Map)
    )
    result = Walk(MapToForLoop()).rewrite(method.code)
    assert result.has_done_something
    assert not result.exceeded_max_iter
    method.verify()
    assert not any(isinstance(stmt, ilist.Map) for stmt in method.code.walk())
    repeated = Walk(MapToForLoop()).rewrite(method.code)
    assert not repeated.has_done_something
    assert not repeated.exceeded_max_iter
    return original_types


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


def test_map_preserves_inferred_callback_return_type() -> None:
    @structural_no_opt
    def identity(x: int) -> Any:
        return x

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[Any, Any]:
        return ilist.map(identity, xs)

    xs = ilist.IList([2, 5], elem=types.Int)
    assert list(mapped(xs)) == [2, 5]
    (inferred_type,) = lower(mapped)
    assert list(mapped(xs)) == [2, 5]
    loop = next(stmt for stmt in mapped.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == inferred_type


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
    observed = emitted_values(output)
    assert observed == values
    lower(mapped)
    assert list(mapped(xs)) == [x + 1 for x in values]
    output = capsys.readouterr().out
    observed = emitted_values(output)
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


def lower_fold(method: ir.Method[..., Any]) -> None:
    TypeInfer(method.dialects, no_raise=False)(method)
    result = Walk(FoldToForLoop()).rewrite(method.code)
    assert result.has_done_something
    assert not result.exceeded_max_iter
    method.verify()
    assert not any(
        isinstance(stmt, (ilist.Foldl, ilist.Foldr)) for stmt in method.code.walk()
    )
    repeated = Walk(FoldToForLoop()).rewrite(method.code)
    assert not repeated.has_done_something
    assert not repeated.exceeded_max_iter


@pytest.mark.parametrize(
    ("values", "initial", "expected"),
    [([], 7, 7), ([3], 2, 23), ([1, 2, 3], 0, 123), ([1, 2, 3], 4, 4123)],
)
def test_foldl_to_loop(values: list[int], initial: int, expected: int) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> int:
        return 10 * acc + x

    @structural_no_opt
    def folded(xs: ilist.IList[int, Any], initial: int) -> int:
        return ilist.foldl(step, xs, initial)

    xs = ilist.IList(values.copy(), elem=types.Int)
    assert folded(xs, initial) == expected
    lower_fold(folded)
    assert folded(xs, initial) == expected
    assert list(xs) == values
    loops = [stmt for stmt in folded.code.walk() if isinstance(stmt, scf.For)]
    assert len(loops) == 1
    assert loops[0].results[0].type == types.Int
    assert loops[0].body.blocks[0].args[1].type == types.Int


def test_foldl_distinct_accumulator_type() -> None:
    @structural_no_opt
    def step(acc: tuple[int, int], x: int) -> tuple[int, int]:
        return (acc[0] + x, acc[1] + 1)

    @structural_no_opt
    def folded(xs: ilist.IList[int, Any]) -> tuple[int, int]:
        return ilist.foldl(step, xs, (10, 0))

    xs = ilist.IList([2, 3, 4], elem=types.Int)
    assert folded(xs) == (19, 3)
    lower_fold(folded)
    assert folded(xs) == (19, 3)
    loop = next(stmt for stmt in folded.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == types.Tuple[types.Int, types.Int]
    call = next(stmt for stmt in loop.body.walk() if isinstance(stmt, func.Call))
    assert call.inputs[0].type == types.Tuple[types.Int, types.Int]
    assert call.inputs[1].type == types.Int


@pytest.mark.parametrize("enabled", [False, True])
def test_foldl_inside_condition(enabled: bool) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> int:
        return 10 * acc + x

    @structural_no_opt
    def folded(xs: ilist.IList[int, Any], enabled: bool) -> int:
        if enabled:
            result = ilist.foldl(step, xs, 0)
        else:
            result = 7
        return result

    xs = ilist.IList([1, 2, 3], elem=types.Int)
    expected = 123 if enabled else 7
    assert folded(xs, enabled) == expected
    lower_fold(folded)
    assert folded(xs, enabled) == expected


@pytest.mark.parametrize("count", [0, 1, 3])
def test_foldl_inside_loop(count: int) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> int:
        return 10 * acc + x

    @structural_no_opt
    def folded(xs: ilist.IList[int, Any], count: int) -> int:
        result = 0
        for _ in range(count):
            result = ilist.foldl(step, xs, result)
        return result

    xs = ilist.IList([1, 2], elem=types.Int)
    expected = {0: 0, 1: 12, 3: 121212}[count]
    assert folded(xs, count) == expected
    lower_fold(folded)
    assert folded(xs, count) == expected


def test_foldl_captured_value() -> None:
    @structural_no_opt
    def folded(xs: ilist.IList[int, Any], offset: int) -> int:
        def step(acc: int, x: int) -> int:
            return acc + x + offset

        return ilist.foldl(step, xs, 0)

    xs = ilist.IList([1, 2], elem=types.Int)
    assert folded(xs, 5) == 13
    lower_fold(folded)
    assert folded(xs, 5) == 13


@pytest.mark.parametrize("values", [[], [3, 1, 3]])
def test_foldl_callback_effects(
    values: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    dialects = structural_no_opt.add(debug)

    @dialects
    def step(acc: int, x: int) -> int:
        debug.info("fold", x)
        return acc + x

    @dialects
    def folded(xs: ilist.IList[int, Any]) -> int:
        return ilist.foldl(step, xs, 7)

    xs = ilist.IList(values, elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_fold(folded)
        assert folded(xs) == 7 + sum(values)
        output = capsys.readouterr().out
        observed = emitted_values(output)
        assert observed == values


def test_foldl_non_match_unchanged() -> None:
    assert not FoldToForLoop().rewrite(py.Constant(1)).has_done_something


@pytest.mark.parametrize(
    ("values", "initial", "expected"),
    [([], 7, 7), ([3], 2, 23), ([1, 2, 3], 0, 321), ([1, 2, 3], 4, 4321)],
)
def test_foldr_to_loop(values: list[int], initial: int, expected: int) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> int:
        return 10 * acc + x

    @structural_no_opt
    def folded(xs: ilist.IList[int, Any], initial: int) -> int:
        return ilist.foldr(step, xs, initial)

    xs = ilist.IList(values.copy(), elem=types.Int)
    assert folded(xs, initial) == expected
    lower_fold(folded)
    assert folded(xs, initial) == expected
    assert list(xs) == values
    loop = next(stmt for stmt in folded.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == types.Int


def test_foldr_callback_order(capsys: pytest.CaptureFixture[str]) -> None:
    dialects = structural_no_opt.add(debug)

    @dialects
    def step(acc: int, x: int) -> int:
        debug.info("fold", x)
        return acc + x

    @dialects
    def folded(xs: ilist.IList[int, Any]) -> int:
        return ilist.foldr(step, xs, 0)

    xs = ilist.IList([1, 2, 3], elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_fold(folded)
        assert folded(xs) == 6
        output = capsys.readouterr().out
        observed = emitted_values(output)
        assert observed == [3, 2, 1]


def lower_scan(method: ir.Method[..., Any]) -> None:
    TypeInfer(method.dialects, no_raise=False)(method)
    result = Walk(ScanToForLoop()).rewrite(method.code)
    assert result.has_done_something
    assert not result.exceeded_max_iter
    method.verify()
    assert not any(isinstance(stmt, ilist.Scan) for stmt in method.code.walk())
    assert not Walk(ScanToForLoop()).rewrite(method.code).has_done_something


@pytest.mark.parametrize(
    ("values", "initial", "expected_state", "expected_outputs"),
    [([], 7, 7, []), ([3], 2, 23, [23]), ([1, 2, 3], 0, 123, [1, 12, 123])],
)
def test_scan_to_loop(
    values: list[int], initial: int, expected_state: int, expected_outputs: list[int]
) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> tuple[int, int]:
        next_acc = 10 * acc + x
        return next_acc, next_acc

    @structural_no_opt
    def scanned(
        xs: ilist.IList[int, Any], initial: int
    ) -> tuple[int, ilist.IList[int, Any]]:
        return ilist.scan(step, xs, initial)

    xs = ilist.IList(values.copy(), elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_scan(scanned)
        state, outputs = scanned(xs, initial)
        assert state == expected_state
        assert list(outputs) == expected_outputs
        assert list(xs) == values


def test_scan_distinct_output_type_and_length() -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> tuple[int, bool]:
        next_acc = acc + x
        return next_acc, next_acc > 3

    @structural_no_opt
    def scanned(
        xs: ilist.IList[int, Literal[3]],
    ) -> tuple[int, ilist.IList[bool, Literal[3]]]:
        return ilist.scan(step, xs, 0)

    xs = ilist.IList([1, 2, 3], elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_scan(scanned)
        state, outputs = scanned(xs)
        assert state == 6
        assert list(outputs) == [False, False, True]
    loop = next(stmt for stmt in scanned.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type == types.Int
    assert loop.results[1].type == ilist.IListType[types.Bool, types.Literal(3)]
    assert loop.body.blocks[0].args[2].type == ilist.IListType[types.Bool, types.Any]


@pytest.mark.parametrize("enabled", [False, True])
def test_scan_captured_value_inside_condition(enabled: bool) -> None:
    @structural_no_opt
    def scanned(
        xs: ilist.IList[int, Any], offset: int, enabled: bool
    ) -> tuple[int, ilist.IList[int, Any]]:
        def step(acc: int, x: int) -> tuple[int, int]:
            return acc + x, x + offset

        if enabled:
            result = ilist.scan(step, xs, 0)
        else:
            result = (7, xs)
        return result

    xs = ilist.IList([1, 2], elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_scan(scanned)
        state, outputs = scanned(xs, 5, enabled)
        assert state == (3 if enabled else 7)
        assert list(outputs) == ([6, 7] if enabled else [1, 2])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_scan_inside_loop(count: int) -> None:
    @structural_no_opt
    def step(acc: int, x: int) -> tuple[int, int]:
        return acc + x, x + 1

    @structural_no_opt
    def scanned(
        xs: ilist.IList[int, Any], count: int
    ) -> tuple[int, ilist.IList[int, Any]]:
        acc = 0
        for _ in range(count):
            pair = ilist.scan(step, xs, acc)
            acc = pair[0]
            xs = pair[1]
        return acc, xs

    xs = ilist.IList([1, 2], elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_scan(scanned)
        state, outputs = scanned(xs, count)
        assert state == {0: 0, 1: 3, 3: 15}[count]
        assert list(outputs) == [1 + count, 2 + count]


@pytest.mark.parametrize("values", [[], [3, 1, 3]])
def test_scan_callback_effects(
    values: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    dialects = structural_no_opt.add(debug)

    @dialects
    def step(acc: int, x: int) -> tuple[int, int]:
        debug.info("scan", x)
        return acc + x, x

    @dialects
    def scanned(xs: ilist.IList[int, Any]) -> tuple[int, ilist.IList[int, Any]]:
        return ilist.scan(step, xs, 7)

    xs = ilist.IList(values, elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            lower_scan(scanned)
        state, outputs = scanned(xs)
        assert state == 7 + sum(values)
        assert list(outputs) == values
        output = capsys.readouterr().out
        observed = emitted_values(output)
        assert observed == values


def test_scan_non_match_unchanged() -> None:
    assert not ScanToForLoop().rewrite(py.Constant(1)).has_done_something


@pytest.mark.parametrize("values", [[], [2], [3, 1, 3]])
@pytest.mark.parametrize("enabled", [False, True])
def test_for_each_effects_and_captures(
    values: list[int], enabled: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    dialects = structural_no_opt.add(debug)

    @dialects
    def visit(xs: ilist.IList[int, Any], offset: int, enabled: bool) -> int:
        def emit(x: int) -> None:
            debug.info("value", x + offset)

        if enabled:
            ilist.for_each(emit, xs)
        return len(xs)

    xs = ilist.IList(values.copy(), elem=types.Int)
    for rewritten in (False, True):
        if rewritten:
            TypeInfer(visit.dialects, no_raise=False)(visit)
            assert Walk(ForEachToForLoop()).rewrite(visit.code).has_done_something
            visit.verify()
            assert not any(
                isinstance(stmt, ilist.ForEach) for stmt in visit.code.walk()
            )
            loop = next(stmt for stmt in visit.code.walk() if isinstance(stmt, scf.For))
            assert len(loop.results) == 0
            assert not Walk(ForEachToForLoop()).rewrite(visit.code).has_done_something
        assert visit(xs, 5, enabled) == len(values)
        output = capsys.readouterr().out
        observed = emitted_values(output)
        assert observed == ([x + 5 for x in values] if enabled else [])
        assert list(xs) == values


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], (False, True)),
        ([True], (True, True)),
        ([False], (False, False)),
        ([True, True], (True, True)),
        ([False, False], (False, False)),
        ([False, True, False], (True, False)),
        ([True, False, True], (True, False)),
    ],
)
def test_boolean_reductions(values: list[bool], expected: tuple[bool, bool]) -> None:
    @structural_no_opt
    def reduce(xs: ilist.IList[bool, Any]) -> tuple[bool, bool]:
        return ilist.any(xs), ilist.all(xs)

    xs = ilist.IList(values, elem=types.Bool)
    assert reduce(xs) == expected
    TypeInfer(reduce.dialects, no_raise=False)(reduce)
    rule = Walk(Chain(AnyToForLoop(), AllToForLoop()))
    assert rule.rewrite(reduce.code).has_done_something
    reduce.verify()
    assert reduce(xs) == expected
    assert not any(
        isinstance(stmt, (ilist.stmts.Any, ilist.stmts.All))
        for stmt in reduce.code.walk()
    )
    loops = [stmt for stmt in reduce.code.walk() if isinstance(stmt, scf.For)]
    assert len(loops) == 2
    assert all(loop.results[0].type == types.Bool for loop in loops)
    assert not rule.rewrite(reduce.code).has_done_something


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("count", [0, 3])
def test_boolean_reductions_nested(enabled: bool, count: int) -> None:
    @structural_no_opt
    def reduce(xs: ilist.IList[bool, Any], enabled: bool, count: int) -> int:
        total = 0
        for _ in range(count):
            if enabled:
                if ilist.any(xs):
                    total = total + 1
                if ilist.all(xs):
                    total = total + 10
        return total

    xs = ilist.IList([True, False], elem=types.Bool)
    expected = count if enabled else 0
    assert reduce(xs, enabled, count) == expected
    TypeInfer(reduce.dialects, no_raise=False)(reduce)
    assert (
        Walk(Chain(AnyToForLoop(), AllToForLoop()))
        .rewrite(reduce.code)
        .has_done_something
    )
    reduce.verify()
    assert reduce(xs, enabled, count) == expected


def test_iteration_rules_non_match() -> None:
    for rule in (ForEachToForLoop(), AnyToForLoop(), AllToForLoop()):
        assert not rule.rewrite(py.Constant(1)).has_done_something


def test_pass_rewrites_every_higher_order_statement() -> None:
    @structural_no_opt
    def double(x: int) -> int:
        return 2 * x

    @structural_no_opt
    def add(acc: int, x: int) -> int:
        return acc + x

    @structural_no_opt
    def running(acc: int, x: int) -> tuple[int, int]:
        return acc + x, acc + x

    @structural_no_opt
    def positive(x: int) -> bool:
        return x > 0

    @structural_no_opt
    def kernel(xs: ilist.IList[int, Any]) -> tuple[int, int, int, bool, bool]:
        doubled = ilist.map(double, xs)
        ilist.for_each(double, doubled)
        signs = ilist.map(positive, doubled)
        scanned = ilist.scan(running, doubled, 0)
        return (
            ilist.foldl(add, doubled, 0),
            ilist.foldr(add, doubled, 0),
            ilist.foldl(add, scanned[1], 0),
            ilist.any(signs),
            ilist.all(signs),
        )

    xs = ilist.IList([1, 2, 3], elem=types.Int)
    expected = kernel(xs)
    assert expected == (12, 12, 20, True, True)

    pass_ = ilist.IListToLoop(kernel.dialects, no_raise=False)
    assert pass_(kernel).has_done_something
    assert not any(
        isinstance(
            stmt,
            (
                ilist.Map,
                ilist.Foldl,
                ilist.Foldr,
                ilist.Scan,
                ilist.ForEach,
                ilist.stmts.Any,
                ilist.stmts.All,
            ),
        )
        for stmt in kernel.code.walk()
    )
    assert sum(isinstance(stmt, scf.For) for stmt in kernel.code.walk()) == 9
    assert kernel(xs) == expected
    assert list(xs) == [1, 2, 3]
    assert not pass_(kernel).has_done_something


def test_pass_infers_types_before_rewriting() -> None:
    @structural_no_opt
    def double(x: int) -> int:
        return 2 * x

    @structural_no_opt
    def mapped(xs: ilist.IList[int, Any]) -> ilist.IList[int, Any]:
        return ilist.map(double, xs)

    assert not mapped.inferred
    assert ilist.IListToLoop(mapped.dialects, no_raise=False)(mapped).has_done_something
    assert mapped.inferred
    loop = next(stmt for stmt in mapped.code.walk() if isinstance(stmt, scf.For))
    assert loop.results[0].type.is_subseteq(ilist.IListType[types.Int])
    assert list(mapped(ilist.IList([1, 2], elem=types.Int))) == [2, 4]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("count", [0, 2])
def test_pass_rewrites_nested_statements(enabled: bool, count: int) -> None:
    @structural_no_opt
    def double(x: int) -> int:
        return 2 * x

    @structural_no_opt
    def add(acc: int, x: int) -> int:
        return acc + x

    @structural_no_opt
    def kernel(xs: ilist.IList[int, Any], enabled: bool, count: int) -> int:
        total = 0
        for _ in range(count):
            if enabled:
                total = total + ilist.foldl(add, ilist.map(double, xs), 0)
        return total

    xs = ilist.IList([1, 2, 3], elem=types.Int)
    expected = kernel(xs, enabled, count)
    assert expected == (12 * count if enabled else 0)
    assert ilist.IListToLoop(kernel.dialects, no_raise=False)(kernel).has_done_something
    assert not any(
        isinstance(stmt, (ilist.Map, ilist.Foldl)) for stmt in kernel.code.walk()
    )
    assert kernel(xs, enabled, count) == expected


def test_pass_leaves_unrelated_methods_unchanged() -> None:
    @structural_no_opt
    def kernel(xs: ilist.IList[int, Any]) -> int:
        return len(xs)

    TypeInfer(kernel.dialects, no_raise=False)(kernel)
    assert not ilist.IListToLoop(kernel.dialects, no_raise=False)(
        kernel
    ).has_done_something
    assert not any(isinstance(stmt, scf.For) for stmt in kernel.code.walk())
