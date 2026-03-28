# [04] 阅读顺序编号文件，对应原文件: chatglm_ptuning.py
import os
import re
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Optional
import gc
import inspect
import torch

# 兼容不同版本的 LangChain 回调类型；若不可用则降级为占位类型。
try:
    from langchain.callbacks.manager import CallbackManagerForLLMRun
except Exception:
    try:
        from langchain_core.callbacks import CallbackManagerForLLMRun
    except Exception:
        class CallbackManagerForLLMRun:  # type: ignore
            pass
from transformers import (AutoModel, AutoConfig, AutoTokenizer)
from config import cfg
from prompt_util import nl2sql_prompt_prefix


def _resolve_llm_model_path() -> Path:
    """
    cfg.LLM_MODEL_DIR 的绝对路径。目录不存在时提前报错，避免新版 huggingface_hub
    将 Windows 路径误当作 Hub repo_id 触发 HFValidationError。
    """
    p = Path(cfg.LLM_MODEL_DIR).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            "LLM 模型目录不存在或不是目录: {}\n"
            "请设置环境变量 LLM_MODEL_DIR 指向本地 chatglm2-6b（含 config.json、tokenizer 等），"
            "Windows 默认 G:\\Models\\chatglm2-6b（见 config/cfg.py）。".format(p)
        )
    return p


def _patch_chatglm_tied_weights_compat() -> None:
    """
    transformers 4.4x 在 finalize 阶段调用 mark_tied_weights_as_initialized，访问 all_tied_weights_keys；
    本地 trust_remote_code 的旧版 ChatGLM modeling 仅有 _tied_weights_keys，会触发 AttributeError。
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if getattr(PreTrainedModel, "_chatglm_tied_weights_patched", False):
        return
    # transformers 4.4x+ 曾提供该方法；较新版本已移除，无需再打补丁。
    if not hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
        PreTrainedModel._chatglm_tied_weights_patched = True
        return
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _mark_tied_weights_patched(self, *args, **kwargs):
        # 旧版 ChatGLM 只有 _tied_weights_keys；新版 transformers 读 all_tied_weights_keys.keys()。
        # 若实例上显式挂了 all_tied_weights_keys=None，hasattr 为真仍会崩，须把 None 视作缺失。
        atk = getattr(self, "all_tied_weights_keys", None)
        if atk is None:
            twk = getattr(self, "_tied_weights_keys", None)
            atk = twk if twk is not None else {}
            try:
                object.__setattr__(self, "all_tied_weights_keys", atk)
            except Exception:
                try:
                    self.__dict__["all_tied_weights_keys"] = atk
                except Exception:
                    pass
        return _orig(self, *args, **kwargs)

    PreTrainedModel.mark_tied_weights_as_initialized = _mark_tied_weights_patched
    PreTrainedModel._chatglm_tied_weights_patched = True


def _patch_quantizers_get_keys_to_not_convert() -> None:
    """
    bitsandbytes 量化 preprocess 调用 get_keys_to_not_convert(model)，
    直接访问 model.all_tied_weights_keys；trust_remote_code 旧版 ChatGLM 仅有 _tied_weights_keys。
    旧版在 quantizers.base；新版在 integrations.bitsandbytes。
    """
    quant_base = None
    try:
        from transformers.quantizers import base as quant_base
    except Exception:
        pass
    try:
        from transformers.integrations import bitsandbytes as bnb_int
    except Exception:
        bnb_int = None

    target = None
    if quant_base is not None and hasattr(quant_base, "get_keys_to_not_convert"):
        target = quant_base
    elif bnb_int is not None and hasattr(bnb_int, "get_keys_to_not_convert"):
        target = bnb_int
    if target is None:
        return
    if getattr(target, "_chatglm_get_keys_to_not_convert_patched", False):
        return
    _orig = target.get_keys_to_not_convert

    def _ensure_all_tied_weights_keys(model) -> None:
        d = getattr(model, "__dict__", {})
        if d.get("all_tied_weights_keys", None) is not None:
            return
        twk = d.get("_tied_weights_keys")
        if twk is None:
            twk = getattr(model, "_tied_weights_keys", None)
        if twk is None:
            twk = getattr(model.__class__, "_tied_weights_keys", None)
        if twk is not None and not isinstance(twk, dict):
            twk = {}
        elif twk is None:
            twk = {}
        d["all_tied_weights_keys"] = twk

    def _wrapped(model):
        _ensure_all_tied_weights_keys(model)
        return _orig(model)

    target.get_keys_to_not_convert = _wrapped
    target._chatglm_get_keys_to_not_convert_patched = True


def _patch_generation_mixin_extract_past_compat() -> None:
    """
    兼容 transformers 4.30 / 4.4x：子类向 super() 传 standardize_cache_format，
    或 generate() 向仅接受 (self,outputs) 的重写传入该关键字时，用 try/except 回退到两参调用。
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        return
    if getattr(GenerationMixin, "_hf_extract_past_compat_patched", False):
        return
    # transformers 4.4x+ 可能已移除此钩子，跳过补丁即可
    raw = getattr(GenerationMixin, "_extract_past_from_model_output", None)
    if raw is None:
        return

    def _wrapped(self, outputs, *args, **kwargs):
        try:
            return raw(self, outputs, *args, **kwargs)
        except TypeError:
            return raw(self, outputs)

    GenerationMixin._extract_past_from_model_output = _wrapped
    GenerationMixin._hf_extract_past_compat_patched = True


def _patch_model_extract_past_compat(model) -> None:
    """
    对「首个在 __dict__ 中定义 _extract_past_from_model_output 的类」打补丁（多为 ChatGLM 重写）。
    若无 __dict__ 项但解析出的方法仍不接受 standardize_cache_format，则对 model.__class__ 兜底。
    """
    mro = model.__class__.__mro__
    candidates: list = []
    for cls in mro:
        if cls is object:
            break
        if "_extract_past_from_model_output" in cls.__dict__:
            candidates.append(cls)
    if not candidates:
        candidates = [model.__class__]

    for cls in candidates:
        if getattr(cls, "_hf_extract_past_compat_patched", False):
            continue
        raw = cls.__dict__.get("_extract_past_from_model_output")
        if raw is None:
            fn = getattr(cls, "_extract_past_from_model_output", None)
            if fn is None:
                continue
            raw = getattr(fn, "__func__", fn)
        try:
            sig = inspect.signature(raw)
            if "standardize_cache_format" in sig.parameters:
                continue
        except Exception:
            pass

        def _make(orig):
            def _wrapped(self, outputs, *args, **kwargs):
                try:
                    return orig(self, outputs, *args, **kwargs)
                except TypeError:
                    return orig(self, outputs)

            return _wrapped

        setattr(cls, "_extract_past_from_model_output", _make(raw))
        cls._hf_extract_past_compat_patched = True
        return

    # 兜底：直接包一层 __class__ 上解析到的方法
    cls = model.__class__
    if getattr(cls, "_hf_extract_past_compat_patched", False):
        return
    fn = getattr(cls, "_extract_past_from_model_output", None)
    if fn is None:
        # 新版 transformers 可能已从 GenerationMixin 移除该方法，ChatGLM 类上也不存在，generate 仍会调用
        def _fallback_extract_past(self, outputs, *args, **kwargs):
            if hasattr(outputs, "past_key_values"):
                return outputs.past_key_values
            if isinstance(outputs, tuple) and len(outputs) > 1:
                return outputs[1]
            return None

        setattr(cls, "_extract_past_from_model_output", _fallback_extract_past)
        cls._hf_extract_past_compat_patched = True
        return
    try:
        if "standardize_cache_format" in inspect.signature(fn).parameters:
            return
    except Exception:
        pass
    orig = getattr(fn, "__func__", fn)

    def _make2(o):
        def _w(self, outputs, *args, **kwargs):
            try:
                return o(self, outputs, *args, **kwargs)
            except TypeError:
                return o(self, outputs)

        return _w

    setattr(cls, "_extract_past_from_model_output", _make2(orig))
    cls._hf_extract_past_compat_patched = True


_patch_generation_mixin_extract_past_compat()
_patch_chatglm_tied_weights_compat()
_patch_quantizers_get_keys_to_not_convert()


def _llm_load_kwargs_base() -> dict:
    """trust_remote_code；low_cpu_mem_usage 恒为 False（避免 meta 占位 + 8bit/accelerate 时加载/推理报错）。

    旧 cfg.CHATGLM_LOW_CPU_MEM_USAGE 已忽略；若需省内存请用 8bit/4bit 或减小模型，勿开 low_cpu_mem。
    """
    return {
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
    }


def _model_is_quantized(model) -> bool:
    return bool(
        getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    )


def _get_llm_load_kwargs(*, for_nothing: bool = False, no_quant: bool = False) -> dict:
    """合并 bitsandbytes 量化参数；无 CUDA 时不启用量化。

    for_nothing=True：加载第二套无 P-Tuning 底座时，默认不量化（见 cfg.CHATGLM_LOAD_IN_8BIT_FOR_NOTHING）。
    no_quant=True：强制非量化（整模 FP16/FP32）。P-Tuning（含 prefix）加载**必须** no_quant：8bit+device_map 与后续
    prefix_encoder.load_state_dict 组合易在推理时触发「Cannot copy out of meta tensor」。
    """
    kwargs = _llm_load_kwargs_base()
    if no_quant:
        return kwargs
    if for_nothing and not getattr(cfg, "CHATGLM_LOAD_IN_8BIT_FOR_NOTHING", False):
        try:
            from loguru import logger

            logger.info(
                "PtuningType.Nothing：CHATGLM_LOAD_IN_8BIT_FOR_NOTHING=0，第二套基座使用 FP16/FP32（"
                "避免双 8bit 下 bitsandbytes Int8Params 与 accelerate dispatch 不兼容）"
            )
        except Exception:
            pass
        return kwargs
    use_4bit = getattr(cfg, "CHATGLM_LOAD_IN_4BIT", False)
    use_8bit = getattr(cfg, "CHATGLM_LOAD_IN_8BIT", False)
    if use_4bit and use_8bit:
        use_8bit = False
    if not (use_4bit or use_8bit):
        return kwargs
    if not torch.cuda.is_available():
        try:
            from loguru import logger

            logger.warning("CHATGLM_LOAD_IN_4BIT/8BIT 需要 CUDA，已使用非量化加载")
        except Exception:
            pass
        return kwargs
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        try:
            from loguru import logger

            logger.warning(
                "已请求 4bit/8bit 量化但 bitsandbytes 未安装，回退为 FP16/FP32 基座加载。请执行: pip install bitsandbytes"
            )
        except Exception:
            print(
                "WARNING: bitsandbytes not installed; loading base without quantization. pip install bitsandbytes",
                flush=True,
            )
        return kwargs
    # transformers 在 device_map 非空时要求 accelerate>=1.1.0（is_accelerate_available）
    try:
        from transformers.utils.import_utils import is_accelerate_available

        if not is_accelerate_available():
            try:
                from loguru import logger

                logger.warning(
                    "4bit/8bit 需 accelerate>=1.1.0 才能使用 device_map，当前环境不满足，"
                    "回退为 FP16/FP32 基座加载。请执行: pip install \"accelerate>=1.1.0\""
                )
            except Exception:
                print(
                    'WARNING: pip install "accelerate>=1.1.0" for 8bit; fallback to FP16.',
                    flush=True,
                )
            return kwargs
    except Exception:
        pass
    # trust_remote_code 的 ChatGLM 不接受 load_in_8bit 传入 __init__；须用 quantization_config（transformers 5.x）
    from transformers import BitsAndBytesConfig

    kwargs["device_map"] = "auto"
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=bool(
                getattr(cfg, "CHATGLM_BNB_INT8_CPU_OFFLOAD", True)
            ),
        )
    return kwargs


def _move_model_to_device(model, device: torch.device) -> None:
    """量化模型由 device_map 放置，禁止再 .to() 整模。"""
    if _model_is_quantized(model):
        model.eval()
        return
    model.to(device).eval()


def _patch_chatglm_config_max_length_alias(config) -> None:
    """
    兼容旧 modeling 与新版 transformers（量化 / device_map 等）：
    - num_hidden_layers：HF 常用名；ChatGLMConfig 仅有 num_layers
    - max_length：旧 modeling_chatglm.__init__；新版 config 用 seq_length
    加载后 max_length 若与 seq_length 重复，须再调用 _strip_chatglm_config_generation_aliases。
    """
    if config is None:
        return
    if getattr(config, "num_hidden_layers", None) is None:
        nl = getattr(config, "num_layers", None)
        if nl is not None:
            try:
                setattr(config, "num_hidden_layers", int(nl))
            except Exception:
                pass
    if getattr(config, "max_length", None) is not None:
        return
    seq = getattr(config, "seq_length", None)
    if seq is None:
        return
    try:
        setattr(config, "max_length", seq)
    except Exception:
        pass


def _strip_chatglm_config_generation_aliases(model) -> None:
    """
    transformers 5.x：config 上若存在 max_length 等，会视为「用 config 控制生成」并在 generate 时抛错。
    旧 ChatGLM 已在 __init__ 用 max_length 填过 max_sequence_length，此处删除与 seq_length 重复的别名。
    """
    if model is None:
        return
    config = getattr(model, "config", None)
    if config is None:
        return
    seq = getattr(config, "seq_length", None)
    ml = getattr(config, "max_length", None)
    if ml is not None:
        try:
            if seq is not None and int(ml) == int(seq):
                delattr(config, "max_length")
        except Exception:
            try:
                delattr(config, "max_length")
            except Exception:
                pass
    # 将默认生成长度放到 generation_config，避免依赖 config
    try:
        g = getattr(model, "generation_config", None)
        if g is not None and seq is not None and getattr(g, "max_length", None) is None:
            g.max_length = int(seq)
    except Exception:
        pass
    _patch_chatglm_disable_dynamic_cache(model)


def _patch_chatglm_disable_dynamic_cache(model) -> None:
    """
    transformers 5.x 的 generate() 默认注入 DynamicCache；旧 ChatGLM 仅支持 tuple 形式的 past_key_values，
    否则会报 DynamicCache is not subscriptable（如 get_masks 中 past_key_values[0][0]）。
    通过声明不支持默认 DynamicCache，让 generate 走旧式缓存路径。
    """
    if model is None:
        return
    cls = model.__class__
    if getattr(cls, "_hf_chatglm_no_dynamic_cache", False):
        return
    if "ChatGLM" not in cls.__name__:
        return

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    cls._supports_default_dynamic_cache = _supports_default_dynamic_cache
    cls._hf_chatglm_no_dynamic_cache = True


# 创建一个枚举类型, 不同ptuning分类
class PtuningType(Enum):
    Nothing = 0
    Classify = 1
    Keywords = 2
    NL2SQL = 3


#  ChatGLM_Ptuning类：通过传入不同PtuningType初始化可以达成单模型多训练权重的使用方式
def _apply_fp16_if_enabled(
    model, ptuning_type: PtuningType, *, classify_force_fp32: bool = False
) -> None:
    """12GB 显存等场景：FP16 降显存；P-Tuning 时 prefix 保持 float32。bitsandbytes 量化模型不可再 .half()。"""
    if _model_is_quantized(model):
        return
    if not getattr(cfg, "QA_CHATGLM_FP16", True):
        return
    if ptuning_type is PtuningType.NL2SQL and getattr(cfg, "QA_NL2SQL_FP32", False):
        return
    # 分类 P-Tuning 在 CPU 上整模 FP16 时，ChatGLM forward 易报 Cannot copy out of meta tensor；整模 FP32 可规避。
    if classify_force_fp32 and ptuning_type is PtuningType.Classify:
        model.float()
        model.transformer.prefix_encoder.float()
        return
    if ptuning_type in (PtuningType.Classify, PtuningType.Keywords, PtuningType.NL2SQL):
        # 若 from_pretrained 已用 torch_dtype=float16，则不再 .half()，避免重复转换与异常
        first = next(model.parameters(), None)
        if first is not None and first.dtype != torch.float16:
            model.half()
        model.transformer.prefix_encoder.float()
    elif ptuning_type is PtuningType.Nothing:
        model.half()


def _materialize_chatglm_prefix_tokens(model: torch.nn.Module) -> None:
    """HF from_pretrained 常在 meta 上构建再灌权重；ChatGLMModel.prefix_tokens 不在 checkpoint
    内，会一直保持 meta，get_prompt 时 .to(device) 报 Cannot copy out of meta tensor。"""
    tr = getattr(model, "transformer", None)
    if tr is None:
        return
    ps = getattr(tr, "pre_seq_len", None)
    if ps is None:
        return
    pt = getattr(tr, "prefix_tokens", None)
    if pt is None or not getattr(pt, "is_meta", False):
        return
    dev = next(model.parameters()).device
    tr.prefix_tokens = torch.arange(int(ps), device=dev, dtype=torch.long)


def _assert_no_meta_tensors(model, label: str = "model") -> None:
    """meta 设备上的参数/缓冲区无法参与计算，会在 forward 中报 Cannot copy out of meta tensor。"""
    bad = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    bad_b = [n for n, b in model.named_buffers() if getattr(b, "is_meta", False)]
    if bad_b:
        bad = (bad or []) + [f"[buffer]{n}" for n in bad_b[:8]]
    if bad:
        raise RuntimeError(
            "{} 仍含 meta 张量（示例）: {} — 请确认 P-Tuning 加载时 CHATGLM_LOAD_IN_8BIT=0；"
            "或设 QA_NL2SQL_FP32=1（仅 NL2SQL）；并升级 transformers/accelerate。".format(
                label, bad[:8]
            )
        )


def _ptuning_load_extra_kwargs() -> dict:
    """P-Tuning 整模加载：与 no_quant 配合，FP16；新版 transformers 用 dtype，避免 torch_dtype 弃用告警。"""
    extra: dict = {}
    if not getattr(cfg, "QA_CHATGLM_FP16", True):
        return extra
    try:
        sig = inspect.signature(AutoModel.from_pretrained)
        if "dtype" in sig.parameters:
            extra["dtype"] = torch.float16
            return extra
    except Exception:
        pass
    extra["torch_dtype"] = torch.float16
    return extra


def _strip_prefix_ckpt_to_state_dict(prefix_state_dict: dict) -> dict:
    """训练保存的 pytorch_model.bin 可能带 transformer.prefix_encoder. / module. / prefix_encoder. 前缀。"""
    out: dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            out[k[len("transformer.prefix_encoder.") :]] = v
        elif k.startswith("module.transformer.prefix_encoder."):
            out[k[len("module.transformer.prefix_encoder.") :]] = v
        elif k.startswith("prefix_encoder."):
            out[k[len("prefix_encoder.") :]] = v
    return out


def _load_prefix_weights_into_model(model, check_point_path: str) -> None:
    """从 checkpoint 加载 prefix_encoder；键不匹配或仍为 meta 时尽早报错。"""
    try:
        from loguru import logger
    except Exception:
        logger = None  # type: ignore

    if not os.path.isfile(check_point_path):
        raise FileNotFoundError(
            "P-Tuning 检查点不存在: {}（需含 pytorch_model.bin）".format(check_point_path)
        )
    raw = torch.load(check_point_path, map_location="cpu")
    new_sd = _strip_prefix_ckpt_to_state_dict(raw)
    if not new_sd:
        sample = list(raw.keys())[:24]
        raise RuntimeError(
            "检查点内未解析到 prefix 权重（需键名 transformer.prefix_encoder.* 或 prefix_encoder.*）。"
            "path={} 示例键: {}".format(check_point_path, sample)
        )
    inc = model.transformer.prefix_encoder.load_state_dict(new_sd, strict=False)
    if getattr(inc, "missing_keys", None) and logger:
        logger.warning("prefix_encoder load_state_dict missing_keys: {}", list(inc.missing_keys)[:20])
    if getattr(inc, "unexpected_keys", None) and logger:
        logger.warning("prefix_encoder load_state_dict unexpected_keys: {}", list(inc.unexpected_keys)[:12])
    pe = model.transformer.prefix_encoder
    for n, p in pe.named_parameters():
        if getattr(p, "is_meta", False):
            raise RuntimeError(
                "prefix_encoder.{} 仍为 meta：检查点未写入或与当前 pre_seq_len 不匹配。path={}".format(
                    n, check_point_path
                )
            )
    # 与训练脚本一致：prefix 保持 float32，再由 _apply_fp16_if_enabled 与主干协调
    pe.float()


def _extract_nl2sql_block(response: str) -> str:
    """
    从 ChatGLM 输出中取出 SQL。旧逻辑用 find('```sql')+7 切片，在找不到围栏时
    会从错误下标截断，整段像乱码；且未兼容 ```SQL / 首尾空白。
    """
    if not response:
        return ""
    text = response.strip()
    m = re.search(r"```\s*sql\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if re.match(r"(?is)^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b", text):
        return text.rstrip("`").strip()
    m = re.search(r"```\s*([\s\S]*?)```", text)
    if m:
        inner = m.group(1).strip()
        if re.match(r"(?is)^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b", inner):
            return inner
    return text


class ChatGLM_Ptuning(ABC):
    model_name = cfg.LLM_MODEL_DIR

    tokenizer: AutoTokenizer = None
    model: AutoModel = None
    config: AutoConfig = None
    isClassify = False
    isKeywords = False
    isNL2SQL = False

    # 通过传入微调权重类型来加载不同的权重进行工作
    # device: None 表示自动 cuda（若可用）；可传 "cpu" / "cuda" 以拆分多模型到 CPU+GPU，缓解显存不足
    def __init__(self, ptuning_type: PtuningType, device: Optional[str] = None):
        check_point_path = ""
        # 载入Tokenizer：Path + local_files_only，避免 Windows 绝对路径被误判为 Hub repo id
        model_path = _resolve_llm_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self._patch_tokenizer_pad_compat()

        if device is None:
            _planned_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _planned_dev = torch.device(device)
            if _planned_dev.type == "cuda" and not torch.cuda.is_available():
                _planned_dev = torch.device("cpu")
        # P-Tuning 带 prefix：禁止 8bit/device_map，避免 prefix 权重注入后 forward 出现 meta tensor
        _pt_kw = {**_get_llm_load_kwargs(no_quant=True), **_ptuning_load_extra_kwargs()}
        if ptuning_type is PtuningType.NL2SQL and getattr(cfg, "QA_NL2SQL_FP32", False):
            _pt_kw.pop("torch_dtype", None)
            _pt_kw.pop("dtype", None)

        classify_force_fp32 = False
        if ptuning_type is PtuningType.Classify or ptuning_type is PtuningType.NL2SQL or ptuning_type is PtuningType.Keywords:
            # 分类 P-Tuning（pre_seq_len 与 NL2SQL 一致时可共享基座）
            if ptuning_type is PtuningType.Classify:
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                                         pre_seq_len=cfg.CLASSIFY_PTUNING_PRE_SEQ_LEN, local_files_only=True)
                _patch_chatglm_config_max_length_alias(self.config)
                check_point_path = os.path.join(cfg.CLASSIFY_CHECKPOINT_PATH, "pytorch_model.bin")
                # CPU 或显式 QA_CLASSIFY_FP32：分类整模 FP32 加载，避免 FP16+CPU 推理 meta tensor
                classify_force_fp32 = _planned_dev.type == "cpu" or getattr(
                    cfg, "QA_CLASSIFY_FP32", False
                )
                _load_kw = dict(_pt_kw)
                if classify_force_fp32:
                    _load_kw.pop("torch_dtype", None)
                    _load_kw.pop("dtype", None)
                try:
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_load_kw
                    )
                except TypeError:
                    _load_kw.pop("torch_dtype", None)
                    _load_kw.pop("dtype", None)
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_load_kw
                    )
                self.isClassify = True
            # 关键词 P-Tuning（pre_seq_len 与 NL2SQL 一致时可共享基座）
            elif ptuning_type is PtuningType.Keywords:
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                                         pre_seq_len=cfg.KEYWORDS_PTUNING_PRE_SEQ_LEN, local_files_only=True)
                _patch_chatglm_config_max_length_alias(self.config)
                check_point_path = os.path.join(cfg.KEYWORDS_CHECKPOINT_PATH, "pytorch_model.bin")
                try:
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_pt_kw
                    )
                except TypeError:
                    _pt_kw.pop("torch_dtype", None)
                    _pt_kw.pop("dtype", None)
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_pt_kw
                    )
                self.isKeywords = True
            # NL2SQL P-Tuning
            elif ptuning_type is PtuningType.NL2SQL:
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                                         pre_seq_len=cfg.NL2SQL_PTUNING_PRE_SEQ_LEN, local_files_only=True)
                _patch_chatglm_config_max_length_alias(self.config)
                check_point_path = os.path.join(cfg.NL2SQL_CHECKPOINT_PATH, "pytorch_model.bin")
                try:
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_pt_kw
                    )
                except TypeError:
                    _pt_kw.pop("torch_dtype", None)
                    _pt_kw.pop("dtype", None)
                    self.model = AutoModel.from_pretrained(
                        model_path, config=self.config, **_pt_kw
                    )
                self.isNL2SQL = True
            # 装载 prefix 权重（键名兼容多种训练导出格式）
            _load_prefix_weights_into_model(self.model, check_point_path)
        else:
            # 未识别到微调
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            _patch_chatglm_config_max_length_alias(self.config)
            self.model = AutoModel.from_pretrained(
                model_path, config=self.config, **_get_llm_load_kwargs(for_nothing=True, no_quant=_planned_dev.type == "cpu")
            )
            self.isClassify = self.isNL2SQL = False

        _strip_chatglm_config_generation_aliases(self.model)

        self._torch_device = _planned_dev
        _apply_fp16_if_enabled(
            self.model, ptuning_type, classify_force_fp32=classify_force_fp32
        )
        try:
            _move_model_to_device(self.model, self._torch_device)
        except RuntimeError as e:
            # 12GB 级显存：分类/NL2SQL 占满内存后，底座 FP16 整模上 GPU 仍可能 OOM
            if _model_is_quantized(self.model):
                raise
            if self._torch_device.type != "cuda" or "out of memory" not in str(e).lower():
                raise
            self.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._torch_device = torch.device("cpu")
            try:
                from loguru import logger

                logger.warning(
                    "GPU 显存不足，已将 {} 回退到 CPU（开放问答会变慢）。"
                    "也可在启动前设置环境变量 QA_CHATGLM_DEVICE=cpu 避免本提示。",
                    ptuning_type,
                )
            except Exception:
                print(
                    "WARNING: GPU OOM, model moved to CPU. Set QA_CHATGLM_DEVICE=cpu to skip.",
                    flush=True,
                )
            _move_model_to_device(self.model, self._torch_device)
        _materialize_chatglm_prefix_tokens(self.model)
        _patch_model_extract_past_compat(self.model)
        if ptuning_type in (PtuningType.Classify, PtuningType.Keywords, PtuningType.NL2SQL):
            _assert_no_meta_tensors(self.model, str(ptuning_type))

    def _patch_tokenizer_pad_compat(self) -> None:
        """
        兼容高版本 transformers 传入 padding_side 参数，
        避免旧版 ChatGLM tokenizer._pad 签名不匹配导致推理报错。
        """
        pad_fn = getattr(self.tokenizer, "_pad", None)
        if pad_fn is None:
            return
        try:
            sig = inspect.signature(pad_fn)
        except Exception:
            return
        if "padding_side" in sig.parameters:
            return
        original_pad = pad_fn

        def _pad_with_compat(*args, padding_side=None, **kwargs):
            return original_pad(*args, **kwargs)

        self.tokenizer._pad = _pad_with_compat

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if len(prompt) > 5120:
            prompt = prompt[:5120]
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_length=8192,
                top_p=1, do_sample=False,
                temperature=0.001)
        except Exception as e:
            try:
                from loguru import logger

                logger.warning("ChatGLM_Ptuning.chat 失败: {}", e)
            except Exception:
                print(e)
            # 不把整段 RAG/长 prompt 回显给用户，避免指令与参考材料出现在「助手」气泡中
            response = "（模型推理失败，请稍后重试。）"
        return response

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt=prompt, stop=stop, run_manager=None)

    def _get_classify_prompt(self, question) -> str:
        classify_prompt = '''
        请问“{}”是属于下面哪个类别的问题?
        A: 招标/采购基础信息查询。
           例如项目编号、公告标题、发布日期、采购人、采购人联系方式、代理机构、代理机构联系方式等。
        B: 中标结果明细查询。
           例如中标人、中标金额、代理服务收费金额、中标人联系方式、项目地点、招标文件等。
        C: 条件过滤下的明细查询（仍返回具体记录）。
           例如“2026年3月浦东发布的项目里，中标人是谁”“某代理机构经手的项目标题有哪些”。
        D: 计算题（公式推导类）。
           无法直接从单条原始字段得到，需要计算，如增长率、占比、比率、均值变化等。
        E: 统计/排序/聚合检索题（SQL检索类）。
           例如数量统计、求和、TopN、最大最小、分组统计、去重计数等。
        F: 开放性问题。
           例如介绍、分析、建议、原因解释、政策解读、概念问答（“什么是XXX”）。

        示例：
        - “项目编号310115131251226162488-15301828谁中标了？” -> B
        - “2026年哪家代理机构代理项目最多？” -> E
        - “2026年3月发布项目里中标金额前3的是哪些？” -> E
        - “该项目中标金额同比增长率是多少？” -> D
        - “请简要分析近期浦东招标趋势” -> F

        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.
        '''.format(question)
        return classify_prompt

    # 加载Classify训练权重后，来强化问题的分类能力，返回问题的类型字母编号
    def classify(self, question: str):
        if self.isClassify:
            classify_prompt = self._get_classify_prompt(question)
            response, _ = self.model.chat(
                self.tokenizer,
                classify_prompt,
                history=[],
                max_length=cfg.CLASSIFY_GEN_MAX_LENGTH,
                top_p=1, do_sample=False,
                temperature=0.001,
                use_cache=False,
            )
            return response
        else:
            print("Error: 未装载Classify训练权重，无法继续任务")

    def _get_keywords_prompt(self, question) -> str:
        question_prompt = '''
                请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词，你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.
                用户输入：
                '''
        keywords_prompt = f"{question_prompt} {question}"
        return keywords_prompt

    # 加载Keywords训练权重后，来强化问题的提取关键词能力，返回问题的关键词
    # 查询题和计算题返回计算核心词，统计题返回符合数据库检索的字段，开放题正常返回关键词
    def keywords(self, question: str):
        if self.isKeywords:
            keywords_prompt = self._get_keywords_prompt(question)
            response, _ = self.model.chat(
                self.tokenizer,
                keywords_prompt,
                history=[],
                max_length=cfg.KEYWORDS_GEN_MAX_LENGTH,
                top_p=1, do_sample=False,
                temperature=0.001,
                use_cache=False,
            )
            return response
        else:
            print("Error: 未装载Keywords训练权重，无法继续任务")

    @property
    def _get_nl2sql_prompt(self) -> str:
        # 与 prompt_util.nl2sql_prompt_prefix 及 ptuning/generate_procurement_ptuning_data 训练样本一致
        return nl2sql_prompt_prefix()

    # 加载NL2SQL训练权重后，来强化问题自然语言对SQL语句的转换
    def nl2sql(self, question: str):
        if self.isNL2SQL:
            question_prompt = f"{self._get_nl2sql_prompt}\"{question}\""
            self.model.eval()
            with torch.inference_mode():
                response, _ = self.model.chat(
                    self.tokenizer,
                    question_prompt,
                    history=[],
                    max_length=cfg.NL2SQL_PTUNING_MAX_LENGTH,
                    top_p=1, do_sample=False,
                    temperature=0.001,
                    use_cache=False,
                )
            return _extract_nl2sql_block(response)
        else:
            print("Error: 未装载NL2SQL训练权重，无法继续任务")



    # 卸载掉已经装在权重的模型
    def unload_model(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SharedPrefixChatGLM(ChatGLM_Ptuning):
    """
    一份 6B 基座 + 仅 NL2SQL 的 prefix_encoder。
    仅当分类与 NL2SQL 的 pre_seq_len 一致时才可共用；当前 512≠128，完整问答默认不加载本类。
    """

    def __init__(self, device: Optional[str] = None):
        model_path = _resolve_llm_model_path()
        self.model_name = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self._patch_tokenizer_pad_compat()
        if device is None:
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._torch_device = torch.device(device)
            if self._torch_device.type == "cuda" and not torch.cuda.is_available():
                self._torch_device = torch.device("cpu")
        pre = cfg.NL2SQL_PTUNING_PRE_SEQ_LEN
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, pre_seq_len=pre, local_files_only=True
        )
        _patch_chatglm_config_max_length_alias(self.config)
        _sp_kw = {**_get_llm_load_kwargs(no_quant=True), **_ptuning_load_extra_kwargs()}
        if getattr(cfg, "QA_NL2SQL_FP32", False):
            _sp_kw.pop("torch_dtype", None)
            _sp_kw.pop("dtype", None)
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                config=self.config,
                **_sp_kw,
            )
        except TypeError:
            _sp_kw.pop("torch_dtype", None)
            _sp_kw.pop("dtype", None)
            self.model = AutoModel.from_pretrained(
                model_path,
                config=self.config,
                **_sp_kw,
            )
        _strip_chatglm_config_generation_aliases(self.model)
        self.isClassify = False
        self.isKeywords = False
        self.isNL2SQL = True
        self._active: Optional[PtuningType] = None

        path = os.path.join(cfg.NL2SQL_CHECKPOINT_PATH, "pytorch_model.bin")
        _load_prefix_weights_into_model(self.model, path)
        self._active = PtuningType.NL2SQL

        _apply_fp16_if_enabled(self.model, PtuningType.NL2SQL)
        _move_model_to_device(self.model, self._torch_device)
        _materialize_chatglm_prefix_tokens(self.model)
        _patch_model_extract_past_compat(self.model)
        _assert_no_meta_tensors(self.model, "SharedPrefixChatGLM")

    def classify(self, question: str):
        raise RuntimeError("SharedPrefix 未加载分类 P-Tuning，请使用 ChatGLM_Ptuning(PtuningType.Classify)")

    def keywords(self, question: str):
        raise RuntimeError("已取消关键词 P-Tuning，批量脚本仅使用规则关键词")

    def nl2sql(self, question: str):
        return ChatGLM_Ptuning.nl2sql(self, question)


class _LiteStubChat:
    """精简模式占位，不加载底座 6B；SQL 纠错/生成类调用会受限。"""

    def __call__(self, prompt: str, stop=None, **kwargs) -> str:
        return "（精简模式未加载底座，无法模型纠错）"

    def set_history_len(self, history_len: int = 10) -> None:
        pass

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _history_len(self) -> int:
        return 10

    def unload_model(self) -> None:
        pass


def _default_device(name: str) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_qa_models():
    """
    返回 (cls_model, sql_model, chat_model)。
    完整模式：分类 P-Tuning（A–F）+ NL2SQL P-Tuning + 底座（F 类检索/生成）；分类与 NL2SQL pre_seq_len 不同时各加载一套 6B。
    精简模式 QA_LITE_NL2SQL_ONLY：不加载分类 P-Tuning，cls_model 为 None。
    RAG 探针 QA_RAG_PROBE：仅加载底座（Nothing），cls/sql 均为 None，只测 F 类检索+RAG。
    """
    from loguru import logger

    if getattr(cfg, "QA_RAG_PROBE", False):
        if getattr(cfg, "QA_LITE_NL2SQL_ONLY", False):
            logger.warning("已设 QA_LITE_NL2SQL_ONLY，但 QA_RAG_PROBE 优先：本次不加载 NL2SQL")
        _, _, d_chat = cfg.qa_chatglm_device_split()
        d_chat = d_chat if d_chat is not None else _default_device("chat")
        logger.warning(
            "QA_RAG_PROBE（RAG 探针）：仅加载底座 device={}，跳过分类/NL2SQL；用于测试向量检索+RAG",
            d_chat,
        )
        chat_model = ChatGLM_Ptuning(PtuningType.Nothing, device=d_chat)
        return None, None, chat_model

    if getattr(cfg, "QA_LITE_NL2SQL_ONLY", False):
        dev = getattr(cfg, "QA_LITE_NL2SQL_DEVICE", None)
        if dev is None:
            _, d_sql, _ = cfg.qa_chatglm_device_split()
        else:
            d_sql = dev
        logger.warning(
            "QA_LITE_NL2SQL_ONLY：仅加载 NL2SQL（device={}）；"
            "若 QA_LITE_FORCE_CLASS=F 则额外加载底座用于开放问答，否则底座为占位",
            d_sql,
        )
        sql_model = ChatGLM_Ptuning(PtuningType.NL2SQL, device=d_sql)
        if getattr(cfg, "QA_LITE_FORCE_CLASS", "E") == "F":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _, _, d_chat = cfg.qa_chatglm_device_split()
            d_chat = d_chat if d_chat is not None else _default_device("chat")
            chat_model = ChatGLM_Ptuning(PtuningType.Nothing, device=d_chat)
            logger.info(
                "QA_LITE_FORCE_CLASS=F：已加载底座（device={}）用于精简模式下的 F 类/闲聊",
                getattr(chat_model, "_torch_device", d_chat),
            )
            return None, sql_model, chat_model
        return None, sql_model, _LiteStubChat()

    d_cls, d_sql, d_chat = cfg.qa_chatglm_device_split()
    d_cls = d_cls if d_cls is not None else _default_device("cls")
    d_sql = d_sql if d_sql is not None else _default_device("sql")
    d_chat = d_chat if d_chat is not None else _default_device("chat")

    logger.info(
        "分类 P-Tuning（pre_seq_len={}）+ NL2SQL（pre_seq_len={}）+ 底座：device {}, {}, {}",
        cfg.CLASSIFY_PTUNING_PRE_SEQ_LEN,
        cfg.NL2SQL_PTUNING_PRE_SEQ_LEN,
        d_cls,
        d_sql,
        d_chat,
    )
    cls_model = ChatGLM_Ptuning(PtuningType.Classify, device=d_cls)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sql_model = ChatGLM_Ptuning(PtuningType.NL2SQL, device=d_sql)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    chat = ChatGLM_Ptuning(PtuningType.Nothing, device=d_chat)
    return cls_model, sql_model, chat


if __name__ == '__main__':

    model = ChatGLM_Ptuning(PtuningType.NL2SQL)
    print(model.nl2sql("2026年哪家代理机构的代理项目最多？"))

    model.unload_model()

    model = ChatGLM_Ptuning(PtuningType.Nothing)
    print(model("你好啊！"))

    model.unload_model()
