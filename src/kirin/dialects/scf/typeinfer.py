from kirin import types, interp
from kirin.analysis import ForwardFrame, TypeInference
from kirin.dialects import func
from kirin.dialects.eltype import ElType

from . import absint
from .stmts import For, IfElse
from ._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(absint.Methods):
    @interp.impl(IfElse)
    def if_else_(
        self,
        interp_: TypeInference,
        frame: ForwardFrame[types.TypeAttribute],
        stmt: IfElse,
    ):
        frame.set(
            stmt.cond, frame.get(stmt.cond).meet(types.Bool)
        )  # set cond backwards
        return super().if_else(interp_, frame, stmt)

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

        for _ in range(8):
            loop_vars_ = interp_.frame_call_region(
                frame, stmt, stmt.body, item, *loop_vars
            )
            if isinstance(loop_vars_, interp.ReturnValue):
                return loop_vars_
            elif not isinstance(loop_vars_, tuple):  # None, loop has no result
                return

            next_vars = tuple(
                old.widen(new) for old, new in zip(loop_vars, loop_vars_)
            )
            if all(new.is_subseteq(old) for old, new in zip(loop_vars, next_vars)):
                return loop_vars  # fixpoint reached
            loop_vars = next_vars
            for arg, value in zip(block_args[1:], loop_vars):
                frame.set(arg, value)
        return tuple(types.Any for _ in loop_vars)
