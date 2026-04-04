"""
Setup shared data for DeepSeek-V2-Lite correctness validation.

Downloads a minimal DCLM corpus shard, tokenizes it with the DeepSeek-V2 tokenizer,
and converts the HuggingFace checkpoint to DCP format.

Idempotent: skips steps whose output already exists.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.build_tokenized_corpus import BuildTokenizedCorpusCfg
from pithtrain.tasks.build_tokenized_corpus import launch as step2_launch
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg
from pithtrain.tasks.convert_checkpoint import launch as step4_launch

# Step 1: Download minimal DCLM shard

RAWTXT = Path("workspace/datasets/dclm-baseline/rawtxt")
SHARD = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"

if not (Path(RAWTXT, SHARD)).exists():
    print(f"Downloading DCLM shard: {SHARD}")
    snapshot_download(
        "mlfoundations/dclm-baseline-1.0",
        repo_type="dataset",
        local_dir=str(RAWTXT),
        allow_patterns=SHARD,
    )
else:
    print(f"DCLM shard already exists: {Path(RAWTXT, SHARD)}")

# Step 2: Tokenize with DeepSeek-V2 tokenizer

TOKTXT = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")

if not TOKTXT.exists() or not any(TOKTXT.glob("*.bin")):
    print("Tokenizing corpus with DeepSeek-V2 tokenizer")
    cfg = BuildTokenizedCorpusCfg()
    cfg.tokenizer_name = "deepseek-ai/DeepSeek-V2-Lite"
    cfg.source_path = RAWTXT
    cfg.output_path = TOKTXT
    step2_launch(cfg)
else:
    print(f"Tokenized corpus already exists: {TOKTXT}")

# Step 3: Download HuggingFace checkpoint

HF_IMPORT = Path("workspace/checkpoints/deepseek-v2-lite/hf-import")

if not HF_IMPORT.exists() or not any(HF_IMPORT.glob("*.safetensors")):
    print("Downloading DeepSeek-V2-Lite HuggingFace checkpoint")
    snapshot_download(repo_id="deepseek-ai/DeepSeek-V2-Lite", local_dir=str(HF_IMPORT))
else:
    print(f"HuggingFace checkpoint already exists: {HF_IMPORT}")

# Step 4: Convert to DCP format

TORCH_DCP = Path("workspace/checkpoints/deepseek-v2-lite/torch-dcp/step-00000000")

if not TORCH_DCP.exists() or not any(TORCH_DCP.iterdir()):
    print("Converting HuggingFace checkpoint to DCP format")
    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = HF_IMPORT
    cfg.save_path = TORCH_DCP
    step4_launch(cfg)
else:
    print(f"DCP checkpoint already exists: {TORCH_DCP}")

print("Setup complete for DeepSeek-V2-Lite correctness validation.")
