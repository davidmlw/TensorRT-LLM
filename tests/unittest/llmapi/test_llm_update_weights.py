import argparse
import torch
import torch.distributed as dist
import atexit
import os
from typing import Any
from tensorrt_llm import SamplingParams
from tensorrt_llm import LLM
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
    ShardedStateDictConfig,
    FullStateDictConfig
)
#from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer

import contextlib
from typing import Generator
import pynvml


def init_distributed():
    """Initialize distributed training"""
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, torch.device("cuda:0")

    # Set default environment variables if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return world_size, rank, torch.device(f"cuda:{rank}")

def exit_distributed():
    """Exit distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

@contextlib.contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Context manager for NVML initialization and shutdown.

    Raises:
        RuntimeError: If NVML initialization fails
    """
    try:
        pynvml.nvmlInit()
        yield
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def device_id_to_physical_device_id(device_id: int) -> int:
    """Convert a logical device ID to a physical device ID considering CUDA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        try:
            physical_device_id = int(device_ids[device_id])
            return physical_device_id
        except ValueError:
            raise RuntimeError(
                f"Failed to convert logical device ID {device_id} to physical device ID. Available devices are: {device_ids}."
            )
    else:
        return device_id

def get_device_uuid(device_idx: int) -> str:
    """Get the UUID of a CUDA device using NVML."""
    # Convert logical device index to physical device index
    global_device_idx = device_id_to_physical_device_id(device_idx)

    # Get the device handle and UUID
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            # Ensure the UUID is returned as a string, not bytes
            if isinstance(uuid, bytes):
                return uuid.decode("utf-8")
            elif isinstance(uuid, str):
                return uuid
            else:
                raise RuntimeError(
                    f"Unexpected UUID type: {type(uuid)} for device {device_idx} (global index: {global_device_idx})"
                )
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f"Failed to get device UUID for device {device_idx} (global index: {global_device_idx}): {e}"
            )

class fsdp_interface:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = self.load_fsdp_model(model_dir)

    def load_fsdp_model(self, model_dir):
        """Load and initialize FSDP model"""
        # Initialize distributed setup
        print(f"World size: {self.world_size}, Rank: {self.rank}, Device: {self.device}")

        # Setup mixed precision policy for FSDP
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

        if self.rank == 0:
            print(f"Loading FSDP model from {model_dir}")

        # Initialize FSDP model
        fsdp_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # Print model info
        if self.rank == 0:
            total_params = sum(p.numel() for p in fsdp_model.parameters())
            trainable_params = sum(p.numel() for p in fsdp_model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model device: {next(fsdp_model.parameters()).device}")

        # Wrap model with FSDP
        fsdp_model = FSDP(
            fsdp_model,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True
        )

        if self.rank == 0:
            print("FSDP model initialized successfully")

        self._held_streamed_param_reference = None
        self._held_sharded_state_dict_reference = None

        return fsdp_model


    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    @torch.no_grad()
    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        # If the model is not FSDP, then we need to manually move it to the GPU
        # For an FSDP model, model.state_dict() will move the params to the GPU
        if not isinstance(self.model, FSDP):
            self.model = self.manual_load_to_gpu(self.model)
            self._held_sharded_state_dict_reference = self.model.state_dict()
        else:
            # Get sharded state dict instead of full state dict for FSDP1
            with FSDP.state_dict_type(
                self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig()
            ):
                self._held_sharded_state_dict_reference = self.model.state_dict()

        # Collect info for streaming multiple tensors
        state_dict_info = []
        for name, tensor in self._held_sharded_state_dict_reference.items():
            # dtensor's numel will return complete tensor instead of only local tensor
            size_in_bytes = tensor.element_size() * tensor.numel()
            state_dict_info.append((name, size_in_bytes))

        #print(f"State dict info: {state_dict_info}")

        return state_dict_info

    @torch.no_grad()
    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        from torch.distributed.tensor import DTensor
        from torch.multiprocessing.reductions import reduce_tensor

        assert self._held_sharded_state_dict_reference is not None, (
            "prepare_weights_for_ipc must be called before get_weights_ipc_handles"
        )

        # Clean up the held tensors to reduce peak memory
        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        converted_params = {}
        for key in keys:
            # Get full_tensor for dtensor (GPU > 1)
            print(f"key: {key}")
            tensor = self._held_sharded_state_dict_reference[key]
            if isinstance(tensor, DTensor):
                full_tensor = tensor.full_tensor()
            else:
                full_tensor = tensor
            # Convert parameters to the configured dtype
            #print(f"FSDP rank {self.rank} name: {key}, shape: {full_tensor.shape}, {full_tensor[0]}")
            converted_params[key] = full_tensor

        # Temporary record the full tensor for cleanup
        # It is needed for cleanup the last full_tensor in the refit process
        self._held_streamed_param_reference = converted_params

        # Get device UUID for IPC
        device_uuid = self.report_device_id()
        # Create handles for the tensors
        all_handles = []
        for key, p in converted_params.items():
            handle = reduce_tensor(p.detach())
            all_handles.append((key, handle))

        #print(f"device_uuid: {device_uuid}, All handles keys: {[key for key, _ in all_handles]}")
        print(f"device_uuid: {device_uuid}")
        return {device_uuid: all_handles}

class trtllm_interface:
    def __init__(self, model_dir, tensor_parallel_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
        self.model_dir = model_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = self.load_trtllm_model(model_dir, tensor_parallel_size)

    def load_trtllm_model(self, model_dir, tensor_parallel_size):
        if self.rank == 0:
            print("Loading TensorRT-LLM model")
            return LLM(
                model=model_dir,
                tensor_parallel_size=tensor_parallel_size,
                #disable_overlap_scheduler=True,
                #load_format='auto'
                #load_format='dummy'
            )
        else:
            return None

def cleanup():
    """Cleanup function to destroy process group"""
    if dist.is_initialized():
        print(f"Cleaning up process group on rank {dist.get_rank()}")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")

    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        default='/model/Qwen2.5-0.5B-Instruct',
                        help="Model checkpoint directory.")

    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=2,
                        help="Tensor parallel size (number of GPUs to use)")

    parser.add_argument('--use_fsdp',
                        action='store_true',
                        help="Use FSDP model loading instead of direct TensorRT-LLM loading")

    args = parser.parse_args()

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    world_size, rank, device = init_distributed()

    sampling_params = SamplingParams(max_tokens=32)

    # Load FSDP model
    fsdp = fsdp_interface(args.model_dir)
    trtllm = trtllm_interface(args.model_dir, args.tensor_parallel_size)

    grouped_param_keys = [key for key,size in fsdp.prepare_weights_for_ipc()]
    handles = fsdp.get_weights_ipc_handles(grouped_param_keys)
    #print(f"handles: {handles}")

    # Collect handles from all ranks
    all_handles = [None for _ in range(world_size)]
    dist.all_gather_object(all_handles, handles)
    all_handles = {k: v for d in all_handles for k, v in d.items()}
    print(f"all_handles: {all_handles.keys()}")

    if rank == 0:
        print(f"Collected handles from all {world_size} ranks:")

    # Now all_handles contains the handles from each rank
    # all_handles[0] = handles from rank 0
    # all_handles[1] = handles from rank 1, etc.

    # For FSDP mode, we would need additional logic to integrate withTensorRT-LLM
    # This is a placeholder for now
    if rank == 0:

        outputs = trtllm.llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

        ## load the model from fsdp
        ## then generate the output again
        result = trtllm.llm.sleep(1)
        print(f"sleep result: {result}")

        result = trtllm.llm.wakeup()
        print(f"wakeup result: {result}")

        result = trtllm.llm.update_weights_from_ipc_handles(all_handles)
        print(f"update weights result: {result}")

        outputs = trtllm.llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

    exit_distributed()
if __name__ == '__main__':
    main()

# torchrun --nproc_per_node=2 generate.py --model_dir /model/Qwen2.5-0.5B-Instruct --tensor_parallel_size 2