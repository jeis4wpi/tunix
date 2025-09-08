# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -x # Enable xtrace


# GCS BUCKET
# # Pretrained
# GEMMA3_1B_PT = 'gs://gemma-data/checkpoints/gemma3-1b-pt'
# GEMMA3_4B_PT = 'gs://gemma-data/checkpoints/gemma3-4b-pt'
# GEMMA3_12B_PT = 'gs://gemma-data/checkpoints/gemma3-12b-pt'
# GEMMA3_27B_PT = 'gs://gemma-data/checkpoints/gemma3-27b-pt'
# # Instruction Tuned
# GEMMA3_1B_IT = 'gs://gemma-data/checkpoints/gemma3-1b-it'
# GEMMA3_4B_IT = 'gs://gemma-data/checkpoints/gemma3-4b-it'
# GEMMA3_12B_IT = 'gs://gemma-data/checkpoints/gemma3-12b-it'
# GEMMA3_27B_IT = 'gs://gemma-data/checkpoints/gemma3-27b-it'
# # Tokenizer
# GEMMA3_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'


# not tested, OOM on v5e when do gemma conversion
python3 -m tunix.sft.peft_main \
  base_config.yaml \
  model_name="gemma3-4b" \
  ckpt_dir="gs://gemma-data/checkpoints/gemma3-4b-pt" \
  ckpt_source="gcs" \
  tokenizer_path="gs://gemma-data/tokenizers/tokenizer_gemma3.model" \
  dataset_name="mtnt/en-fr" \
  optimizer="adamw" \
  learning_rate=1e-5 \
  training_config.eval_every_n_steps=20 \
  training_config.max_steps=100 \
  training_config.metrics_logging_options.log_dir="/tmp/tensorboard/full" \
  training_config.metrics_logging_options.flush_every_n_steps=20 \
  lora_config={} \
  mesh.shape="(2,2)" \
  mesh.axis_names="('fsdp','tp')"