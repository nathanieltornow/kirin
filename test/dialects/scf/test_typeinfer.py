from pytest import mark

from kirin import types
from kirin.prelude import structural_no_opt
from kirin.analysis import TypeInference
from kirin.dialects.ilist import IListType

type_infer = TypeInference(structural_no_opt)


@mark.xfail(reason="for with early return not supported in scf lowering")
def test_inside_return_loop():
    @structural_no_opt
    def simple_loop(x: float):
        for i in range(0, 3):
            return i
        return x

    frame, ret = type_infer.run(simple_loop)
    assert ret.is_subseteq(types.Int | types.Float)


@mark.xfail(reason="if with early return not supported in scf lowering")
def test_simple_ifelse():
    @structural_no_opt
    def simple_ifelse(x: int):
        cond = x > 0
        if cond:
            return cond
        else:
            return 0

    frame, ret = type_infer.run(simple_ifelse)
    assert ret.is_subseteq(types.Bool | types.Int | types.NoneType)


def test_loop_carried_ilist_growth():
    """A list that grows every iteration has an unknown length, not a union of
    the lengths reached after zero and one iterations."""

    @structural_no_opt
    def grow(n: int):
        y = [1]
        for i in range(n):
            y = y + [i]
        return y

    frame, ret = type_infer.run(grow)
    assert ret.is_structurally_equal(IListType[types.Int, types.Any])


def test_loop_carried_union_is_not_widened():
    """Widening only extrapolates infinite chains; unrelated types still union."""

    @structural_no_opt
    def alternating(n: int):
        y = 1
        for i in range(n):
            if i > 2:
                y = "a"
        return y

    frame, ret = type_infer.run(alternating)
    assert ret.is_structurally_equal(types.Int | types.String)
