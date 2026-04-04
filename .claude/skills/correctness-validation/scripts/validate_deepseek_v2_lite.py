"""
Run 15 training steps with DeepSeek-V2-Lite for correctness validation.

Uses a constant tiny learning rate (1e-6) and loads from a released HuggingFace
checkpoint converted to DCP. The goal is to verify that loss computation and
gradient flow are correct, not to actually train.

Launch with:
    bash .claude/skills/correctness-validation/scripts/launch_validate.sh deepseek-v2-lite
"""

from pathlib import Path

from pithtrain.tasks.pretrain_language_model import PretrainLanguageModelCfg, launch

cfg = PretrainLanguageModelCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 2

training = cfg.training
training.model = Path("examples/pretrain_language_model/deepseek-v2-lite/config.json")
training.optimizer = "Adam"
training.scheduler = "Constant"
training.max_lr = 1e-6
training.min_lr = 1e-6
training.warmup_steps = 0
training.max_steps = 15
training.micro_batch_size = 1
training.global_batch_size = 512
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
training.moe_load_balance_type = "sequence"
training.moe_load_balance_coef = 3e-3
training.fp8_training = "disabled"
training.save_location = Path("workspace/checkpoints/deepseek-v2-lite")

if __name__ == "__main__":
    launch(cfg)
