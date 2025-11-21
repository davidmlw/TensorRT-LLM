from typing import Any, Optional

from .llm import LLM


# We extend with the AsyncLLM class rather than modifying the LLM class directly
# since AsyncLLM operations depend on Ray, while the LLM class should be generic
# and independent of the orchestrator.
# NOTE: This class is for internal use only, not for external use. It will be
# changed frequently and may not be stable.
class AsyncLLM(LLM):
    """AsyncLLM is a subclass of LLM that supports asynchronous setup, release and
    resume operations that are necessary for RL or agentic scenarios.
    """

    def __init__(self, *args, **kwargs):
        # NOTE: This should be light, the heavy initialization should be done in the async_init_phase.
        super().__init__(*args, **kwargs)

    # It will inherit the LLM.generate_async method.

    # I doubt if we need to support both sync and async modes for each method.
    # Maybe only the async mode is enough considering RL developers are more
    # likely to use async mode.

    async def setup_async(self):
        """Setup the LLM asynchronously."""
        pass

    def release_async(self):
        """Release the GPU memory used by the LLM asynchronously."""
        pass

    def resume_async(self):
        """Resume the LLM asynchronously."""

    def collective_rpc_call(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        unique_reply_rank: Optional[int] = None,
    ) -> list[Any]:
        """Execute an RPC call on all GPU workers. Currently, this is only supported for RayExecutor.

        Args:
            method (str): The name of the worker method to execute.
            args (tuple[Any, ...]): Positional arguments to pass to the worker method. Defaults to ().
            kwargs (dict, optional): Keyword arguments to pass to the worker method. Defaults to None.
            non_block (bool): Whether to block until all workers have completed the RPC call. Defaults to False.
            unique_reply_rank (int, optional): The rank of the worker that will be used to send the reply. Defaults to None.

        Returns:
            list[Any]: A list of results from each worker.
        """
        pass
