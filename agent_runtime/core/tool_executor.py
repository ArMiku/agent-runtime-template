from collections.abc import AsyncGenerator
from typing import Generic

from .run_context import ContextWrapper, TContext
from .tool import FunctionTool, ToolExecResult


class BaseFunctionToolExecutor(Generic[TContext]):
    def execute(
        self,
        tool: FunctionTool,
        run_context: ContextWrapper[TContext],
        **tool_args,
    ) -> AsyncGenerator[ToolExecResult, None]:
        raise NotImplementedError
