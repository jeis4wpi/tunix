import os
import jax.numpy as jnp
import pathwaysutils

from vllm import LLM
pathwaysutils.initialize()


# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "1"


MODEL = "meta-llama/Llama-3.1-8B-Instruct"

golden_llm = LLM(
    MODEL,
    max_model_len=128,
    tensor_parallel_size=16,
    gpu_memory_utilization=0.3,
)

print("vLLM model loaded successfully")

import functools
import os
from pprint import pprint
import re
import sys

from datetime import datetime
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import grain
import humanize


import pathwaysutils
pathwaysutils.initialize()

import jax
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger

from transformers import AutoTokenizer

from flax import linen as nn
from tunix.models.llama3 import model as llama3_lib
import numpy as np
from etils import epath

from tunix.rl.rollout.base_rollout import RolloutConfig

from MaxText.globals import MAXTEXT_ASSETS_ROOT

from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard


# ~/HOME/maxtext/MaxText/examples

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to get the project root
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter

print(f"JAX devices: {jax.devices()}")

DEBUG = True  # set to True to run in debug mode, for more print statements
HOME = os.path.expanduser("~") + "/"
print(f"Home directory (from Python): {HOME}")



# Look for base.yml in two possible locations.
path1 = os.path.join(HOME, "maxtext/src/MaxText/configs/base.yml")
path2 = "src/MaxText/configs/base.yml"
if os.path.exists(path1):
  BASE_YAML_PATH = path1
elif os.path.exists(path2):
  BASE_YAML_PATH = path2
else:
  raise FileNotFoundError(
      "Could not find base.yml in the expected locations: "
      f"{path1} or {path2}"
  )
def get_ref_maxtext_model(config):

  model, mesh = model_creation_utils.create_nnx_model(config)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model,)

    model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = model_config

  return tunix_model, mesh



model_config = llama3_lib.ModelConfig.llama3_1_8b()

# Load the reference model
# Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
# To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
config_ref = pyconfig.initialize(
    [
        "",
        BASE_YAML_PATH,
    ],
    base_output_directory="dummy",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    tokenizer_type="tiktoken",
    tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer_llama3.tiktoken"),
    load_parameters_path="gs://mazumdera-test-bucket-europe-west4/llama3.1-8b-Instruct/scanned-pathways/0/items",
    # load_parameters_path="path/to/scanned/checkpoint",
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=1024,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
)

# breakpoint()



llama3_1_8b, mesh = get_ref_maxtext_model(config_ref)
llama3_1_8b.config = model_config
print("Maxtext model loaded successfully")


utils.transfer_state_with_mappings(src_state=nnx.state(llama3_1_8b),dst_state=golden_llm.llm_engine.model_executor.driver_worker.model_runner.state,key_mappings=llama3_1_8b.to_hf_mappings(),key_mapping_hook_fns=llama3_1_8b.to_hf_hook_fns(),transpose_keys=llama3_1_8b.to_hf_transpose_keys(), reshard_fn=reshard.reshard_pytree,)

print("after weight transfer")

print(golden_llm.generate("what is the capital of France?"))
