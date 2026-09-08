from kirin import ir, types
from kirin.dialects import py, scf, ilist, func
from kirin.rewrite.abc import RewriteRule, RewriteResult
from kirin.ir.nodes.stmt import Statement


class MapToForLoop(RewriteRule):
    """Rewrite `ilist.map` into an indexed loop that accumulates callback results.

    For example, rewrites:
    ```python
    ys = ilist.map(fn, xs)
    ```

    to:
    ```python
    ys = []
    for i in range(len(xs)):
        ys = ys + [fn(xs[i])]
    ```
    """

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if not isinstance(node, ilist.Map):
            return RewriteResult()

        collection_type = node.collection.type
        result_type = node.result.type
        if not (
            isinstance(collection_type, types.Generic)
            and collection_type.is_subseteq(ilist.IListType)
            and isinstance(result_type, types.Generic)
            and result_type.is_subseteq(ilist.IListType)
        ):
            return RewriteResult()

        input_element_type = collection_type.vars[0]
        output_element_type = result_type.vars[0]
        accumulator_type = ilist.IListType[output_element_type, types.Any]

        length = py.Len(node.collection)
        zero = py.Constant(0)
        one = py.Constant(1)
        indices = ilist.stmts.Range(zero.result, length.result, one.result)
        empty = ilist.New((), elem_type=output_element_type)

        body = ir.Block(argtypes=(types.Int, accumulator_type))
        index, accumulated = body.args

        element = py.GetItem(node.collection, index)
        element.result.type = input_element_type

        call = func.Call(
            callee=node.fn,
            inputs=(element.result,),
            kwargs=(),
        )
        call.result.type = output_element_type

        singleton = ilist.New((call.result,), elem_type=output_element_type)
        appended = py.Add(accumulated, singleton.result)
        appended.result.type = accumulator_type

        for stmt in (element, call, singleton, appended, scf.Yield(appended.result)):
            body.stmts.append(stmt)

        loop = scf.For(indices.result, ir.Region(body), empty.result)
        loop.results[0].type = result_type

        for stmt in (length, zero, one, indices, empty, loop):
            stmt.insert_before(node)

        node.result.replace_by(loop.results[0])
        node.delete()
        return RewriteResult(has_done_something=True)
