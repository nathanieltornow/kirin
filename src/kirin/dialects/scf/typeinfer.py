from kirin import types, interp
from kirin.analysis import ForwardFrame, TypeInference, const
from kirin.dialects import func
from kirin.dialects.eltype import ElType

from . import absint
from .stmts import For, IfElse
from ._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(absint.Methods):
    def infer_if_else(
        self,
        interp_: interp.AbstractInterpreter,
        frame: interp.AbstractFrame,
        stmt: IfElse,
    ):
        frame.set(
            stmt.cond, frame.get(stmt.cond).meet(types.Bool)
        )  # set cond backwards
        if isinstance(hint := stmt.cond.hints.get("const"), const.Value):
            # The dead branch still gets its types, since `verify_type` checks
            # every statement. Its results do not reach the join.
            dead = stmt.else_body if hint.data else stmt.then_body
            self._infer_if_else_cond(interp_, frame, stmt, dead)
        return super().infer_if_else(interp_, frame, stmt)

    @interp.impl(For)
    def for_loop(
        self,
        interp_: TypeInference,
        frame: ForwardFrame[types.TypeAttribute],
        stmt: For,
    ):
        loop_vars = frame.get_values(stmt.initializers)
        body_block = stmt.body.blocks[0]
        block_args = body_block.args

        eltype_stmt = ElType(stmt.iterable)
        eltype = interp_.frame_eval(frame, eltype_stmt)
        eltype_stmt.drop_all_references()

        if not isinstance(eltype, tuple):  # error
            return
        item = eltype[0]
        frame.set_values(block_args, (item,) + loop_vars)

        if isinstance(body_block.last_stmt, func.Return):
            frame.worklist.append(interp.Successor(body_block, item, *loop_vars))
            return  # if terminate is Return, there is no result

        loop_vars_ = interp_.frame_call_region(frame, stmt, stmt.body, item, *loop_vars)
        if isinstance(loop_vars_, interp.ReturnValue):
            return loop_vars_
        elif isinstance(loop_vars_, tuple):
            return interp_.join_results(loop_vars, loop_vars_)
        else:  # None, loop has no result
            return
