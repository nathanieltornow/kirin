from pytest import mark

from kirin import types
from kirin.prelude import structural, structural_no_opt
from kirin.analysis import TypeInference

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


def test_the_dead_branch_of_a_constant_condition_is_typed():
    @structural(typeinfer=True)
    def main(x: float):
        flag = 1 > 0
        if flag:
            y = x * 0.5
        else:
            y = x * 0.25
        return y

    products = [s for s in main.callable_region.walk() if s.name == "mult"]
    assert len(products) == 2
    assert all(p.results[0].type == types.Float for p in products)
    assert main.return_type == types.Float
