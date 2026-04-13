"""
Setup shared data for DeepSeek-V2-Lite correctness validation.
"""

import os
from pathlib import Path


def run_step1():
    """
    Download one shard of the DCLM corpus.
    """
    rawtxt = Path("workspace/datasets/dclm-baseline/rawtxt")
    rawzst = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"
    if Path(rawtxt, rawzst).exists():
        print(f"DCLM shard already exists: {Path(rawtxt, rawzst)}")
        return
    from huggingface_hub import hf_hub_download

    repo_id = "mlfoundations/dclm-baseline-1.0"
    hf_hub_download(repo_id, rawzst, repo_type="dataset", local_dir=rawtxt)


def run_step2():
    """
    Tokenize the DCLM corpus with DeepSeek-V2-Lite tokenizer.
    """
    toktxt = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
    if toktxt.exists() and any(toktxt.glob("*.bin")):
        print(f"Tokenized corpus already exists: {toktxt}")
        return
    from pithtrain.tasks.build_tokenized_corpus import BuildTokenizedCorpusCfg, launch

    cfg = BuildTokenizedCorpusCfg()
    cfg.tokenizer_name = "deepseek-ai/DeepSeek-V2-Lite"
    cfg.source_path = Path("workspace/datasets/dclm-baseline/rawtxt")
    cfg.output_path = toktxt
    cfg.num_workers = min(os.cpu_count() or 1, 24)  # single file; avoid spawning idle pools
    launch(cfg)


def run_step3():
    """
    Download the DeepSeek-V2-Lite HuggingFace checkpoint.
    """
    hf_import = Path("workspace/checkpoints/deepseek-v2-lite/hf-import")
    if hf_import.exists() and any(hf_import.glob("*.safetensors")):
        print(f"HuggingFace checkpoint already exists: {hf_import}")
        return
    from huggingface_hub import snapshot_download

    snapshot_download("deepseek-ai/DeepSeek-V2-Lite", local_dir=hf_import)


def run_step4():
    """
    Convert the checkpoint into DCP format for training.
    """
    torch_dcp = Path("workspace/checkpoints/deepseek-v2-lite/torch-dcp/step-00000000")
    if Path(torch_dcp, ".metadata").exists():
        print(f"DCP checkpoint already exists: {torch_dcp}")
        return
    from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path = Path("workspace/checkpoints/deepseek-v2-lite/hf-import")
    cfg.save_path = torch_dcp
    launch(cfg)


if __name__ == "__main__":
    run_step1()
    run_step2()
    run_step3()
    run_step4()
