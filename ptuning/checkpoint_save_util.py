# Shared by CLASSIFY_PTUNING / KEYWORDS_PTUNING / NL2SQL_PTUNING main.py
import json
import logging
import os
import re
import time

import torch
from transformers import TrainerCallback
from transformers.modeling_utils import unwrap_model

logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """
    Writes output_dir/training_progress.json whenever Trainer logs (e.g. every logging_steps).
    Open or refresh this file in the editor to see global_step / loss without the original terminal.
    """

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs") or {}
        data = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if getattr(state, "epoch", None) is not None else None,
            "max_steps": int(args.max_steps) if args.max_steps is not None and args.max_steps >= 0 else None,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "logs": logs,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = os.path.join(args.output_dir, "training_progress.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("TrainingProgressCallback: could not write %s: %s", path, e)


class ForcePrefixSaveCallback(TrainerCallback):
    """After each HF checkpoint save, overwrite with reliable pytorch_model.bin (HF may leave folder empty)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        model = unwrap_model(model)
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            save_ptuning_artifacts(model, self.tokenizer, ckpt_dir, state.global_step)
        except Exception as e:
            logger.warning("ForcePrefixSaveCallback: %s", e)


def save_ptuning_artifacts(model, tokenizer, output_dir: str, global_step: int) -> str:
    """
    Always write P-tuning weights + tokenizer + manifest. Does not rely on HF Trainer.save_model / should_save.
    Inference (04-chatglm_ptuning.py) loads pytorch_model.bin with keys transformer.prefix_encoder.* .
    """
    os.makedirs(output_dir, exist_ok=True)
    sd = model.state_dict()
    filtered = {
        k: v.detach().cpu().clone()
        for k, v in sd.items()
        if k.startswith("transformer.prefix_encoder")
    }
    if not filtered:
        raise RuntimeError(
            "No transformer.prefix_encoder.* tensors in model.state_dict(); P-tuning not saved."
        )
    bin_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(filtered, bin_path)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    manifest = {
        "global_step": int(global_step),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pytorch_model_bin_bytes": os.path.getsize(bin_path),
        "num_prefix_tensors": len(filtered),
        "output_dir": os.path.abspath(output_dir),
    }
    # 与 HF 全量权重常见的 .safetensors 命名对齐；仅含 prefix 张量。推理仍用 pytorch_model.bin（torch.load）。
    try:
        from safetensors.torch import save_file

        st_path = os.path.join(output_dir, "model.safetensors")
        save_file(filtered, st_path)
        manifest["model_safetensors"] = os.path.basename(st_path)
        manifest["model_safetensors_bytes"] = os.path.getsize(st_path)
    except ImportError:
        logger.info(
            "未安装 safetensors，已跳过 model.safetensors；pip install safetensors 后下次保存会同时写出。"
        )
    except Exception as e:
        logger.warning("写入 model.safetensors 失败（不影响 pytorch_model.bin）：%s", e)
    man_path = os.path.join(output_dir, "ptuning_save_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "P-tuning artifacts saved: %s (%d bytes, %d tensors). Manifest: %s",
        bin_path,
        manifest["pytorch_model_bin_bytes"],
        manifest["num_prefix_tensors"],
        man_path,
    )
    return bin_path


def resave_safetensors_from_pytorch_bin(ckpt_dir: str) -> str:
    """
    仅补写：从已有 pytorch_model.bin 生成 model.safetensors，并更新/合并 ptuning_save_manifest.json。
    不加载 6B 底座，适合对历史 checkpoint-10/20/30 补跑保存逻辑。
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(bin_path)
    filtered = torch.load(bin_path, map_location="cpu")
    if not isinstance(filtered, dict) or not filtered:
        raise RuntimeError("pytorch_model.bin 内容无效或为空")
    from safetensors.torch import save_file

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    save_file(filtered, st_path)

    manifest_path = os.path.join(ckpt_dir, "ptuning_save_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}
    m = re.match(r"checkpoint-(\d+)$", os.path.basename(ckpt_dir))
    if m and manifest.get("global_step") is None:
        manifest["global_step"] = int(m.group(1))
    manifest["pytorch_model_bin_bytes"] = os.path.getsize(bin_path)
    manifest["num_prefix_tensors"] = len(filtered)
    manifest["output_dir"] = ckpt_dir
    manifest["model_safetensors"] = os.path.basename(st_path)
    manifest["model_safetensors_bytes"] = os.path.getsize(st_path)
    manifest["resaved_safetensors_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("resave_safetensors_from_pytorch_bin: %s", st_path)
    return st_path
