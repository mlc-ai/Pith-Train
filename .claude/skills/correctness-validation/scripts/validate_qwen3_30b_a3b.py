"""
Run 15 training steps with Qwen3-30B-A3B for correctness validation.

Uses a constant tiny learning rate (1e-6) and loads from a released HuggingFace
checkpoint converted to DCP. The goal is to verify that loss computation and
gradient flow are correct, not to actually train.

Launch with:
    bash .claude/skills/correctness-validation/scripts/launch_validate.sh qwen3-30b-a3b
"""

from pathlib import Path

from pithtrain.tasks.pretrain_language_model import PretrainLanguageModelCfg, launch

cfg = PretrainLanguageModelCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_language_model/qwen3-30b-a3b/config.json")
training.optimizer = "Adam"
training.scheduler = "Constant"
training.max_lr = 1e-6
training.min_lr = 1e-6
training.warmup_steps = 0
training.max_steps = 15
training.micro_batch_size = 1
training.global_batch_size = 512
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = "disabled"
training.save_location = Path("workspace/checkpoints/qwen3-30b-a3b")

if __name__ == "__main__":
    launch(cfg)
