"""
Setup shared data for Qwen3-30B-A3B correctness validation.

Downloads a minimal DCLM corpus shard, tokenizes it with the Qwen3 tokenizer,
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

# Step 2: Tokenize with Qwen3 tokenizer

TOKTXT = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")

if not TOKTXT.exists() or not any(TOKTXT.glob("*.bin")):
    print("Tokenizing corpus with Qwen3 tokenizer")
    cfg = BuildTokenizedCorpusCfg()
    cfg.tokenizer_name = "Qwen/Qwen3-30B-A3B"
    cfg.source_path = RAWTXT
    cfg.output_path = TOKTXT
    step2_launch(cfg)
else:
    print(f"Tokenized corpus already exists: {TOKTXT}")

# Step 3: Download HuggingFace checkpoint

HF_IMPORT = Path("workspace/checkpoints/qwen3-30b-a3b/hf-import")

if not HF_IMPORT.exists() or not any(HF_IMPORT.glob("*.safetensors")):
    print("Downloading Qwen3-30B-A3B HuggingFace checkpoint")
    snapshot_download(repo_id="Qwen/Qwen3-30B-A3B", local_dir=str(HF_IMPORT))
else:
    print(f"HuggingFace checkpoint already exists: {HF_IMPORT}")

# Step 4: Convert to DCP format

TORCH_DCP = Path("workspace/checkpoints/qwen3-30b-a3b/torch-dcp/step-00000000")

if not TORCH_DCP.exists() or not any(TORCH_DCP.iterdir()):
    print("Converting HuggingFace checkpoint to DCP format")
    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = HF_IMPORT
    cfg.save_path = TORCH_DCP
    step4_launch(cfg)
else:
    print(f"DCP checkpoint already exists: {TORCH_DCP}")

print("Setup complete for Qwen3-30B-A3B correctness validation.")
