#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

export NCCL_P2P_DISABLE=1

# DeepSpeed Team
OUTPUT=/home/public/span/test_dpo
ZERO_STAGE=2

mkdir -p $OUTPUT


deepspeed --master_port 12346 --include localhost:6,7 DPO/main.py \
   --data_path sarahpann/PRM800K_simplified \
   --data_split 2,4,4 \
   --actor_model_name_or_path meta-math/MetaMath-Mistral-7B \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-5 \
   --actor_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --enable_hybrid_engine \
   --actor_lora_dim 64 \
   --actor_lora_module_name "layers." \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log