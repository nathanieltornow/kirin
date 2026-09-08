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
        if not isinstance(node, ilist.stmts.Map):
            return RewriteResult()

        length = py.Len(node.collection)
        zero = py.Constant(0)
        one = py.Constant(1)
        indices = ilist.stmts.Range(zero.result, length.result, one.result)

        element_type: types.TypeAttribute = types.Any
        callback_type = node.fn.type
        if isinstance(callback_type, types.FunctionType):
            if callback_type.return_type is not None:
                element_type = callback_type.return_type

        output_type = ilist.IListType[element_type, types.Any]
        res = ilist.New((), elem_type=element_type)

        body = ir.Block(argtypes=(types.Int, output_type))
        index, accumulated = body.args
        element = py.GetItem(node.collection, index)
        collection_type = node.collection.type
        if isinstance(collection_type, types.Generic) and collection_type.is_subseteq(
            ilist.IListType
        ):
            element.result.type = collection_type.vars[0]

        call = func.Call(callee=node.fn, inputs=(element.result,), kwargs=())

        call.result.type = element_type
        singleton = ilist.New((call.result,), elem_type=element_type)

        appended = py.Add(accumulated, singleton.result)
        appended.result.type = output_type

        yield_ = scf.Yield(appended.result)

        for stmt in [element, call, singleton, appended, yield_]:
            body.stmts.append(stmt)

        loop = scf.For(indices.result, ir.Region(body), res.result)
        if (
            isinstance(collection_type, types.Generic)
            and collection_type.is_subseteq(ilist.IListType)
            and len(collection_type.vars) == 2
        ):
            loop.results[0].type = ilist.IListType[
                element_type, collection_type.vars[1]
            ]

        for stmt in (length, zero, one, indices, res, loop):
            stmt.insert_before(node)
        node.result.replace_by(loop.results[0])
        node.delete()

        return RewriteResult(has_done_something=True)
