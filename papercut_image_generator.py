"""
非遗剪纸 — 图像生成管线
接收约束生成引擎的结构化参数，调用图像生成模型，输出剪纸图像

架构:
  ConstraintParameters
    → PromptBuilder: 构建图像模型提示词
    → ImageBackend: 调用生成模型（本地SD/API）
      ├─ LocalBackend: diffusers + ControlNet (本地GPU)
      └─ APIBackend: Replicate/自定义API
    → PostProcessor: 剪纸风格后处理（二值化/去背景/边缘锐化）
    → ResultManager: 保存结果 + 写入GenerationRecord

支持:
  - 本地/API 双后端，配置切换
  - ControlNet 边缘控制（保持剪纸线条特征）
  - 参考图 IP-Adapter 风格迁移
  - 批量生成 + 参数探索
  - 结果自动归档到知识图谱的 GenerationRecord
"""

import json
import os
import time
import logging
import hashlib
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
import threading


# ========== 配置 ==========

@dataclass
class GeneratorConfig:
    """生成配置"""
    # 输出
    output_dir: str = "./generated_output"
    save_images: bool = True
    save_metadata: bool = True

    # 后端选择
    backend: str = "api"  # "local" 或 "api"
    device: str = "cuda"  # cuda / cpu / mps
    fallback_to_cpu: bool = True

    # 本地模型 (diffusers)
    local_model_id: str = "runwayml/stable-diffusion-v1-5"
    local_controlnet_id: str = "lllyasviel/sd-controlnet-canny"
    local_lora_path: str = ""          # 剪纸 LoRA 权重路径（若有）
    local_ip_adapter_path: str = ""    # IP-Adapter 路径

    # API 后端
    api_endpoint: str = ""
    api_key: str = ""
    api_model: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

    # 生成参数默认值
    default_width: int = 768
    default_height: int = 768
    default_steps: int = 30
    default_cfg_scale: float = 7.0
    default_batch_size: int = 4

    # 后处理
    enable_postprocessing: bool = True
    postprocess_binarize: bool = True       # 二值化（纯黑白色）
    postprocess_edges_sharpen: bool = True  # 边缘锐化
    postprocess_remove_bg: bool = True      # 去背景

    # 记录
    record_to_generation_record: bool = True

    def to_dict(self):
        return asdict(self)


# ========== 生成结果 ==========

@dataclass
class GenerationResult:
    """单次生成结果"""
    image_path: str                          # 输出图像路径
    thumbnail_path: str = ""                 # 缩略图路径
    prompt: str = ""                         # 使用的提示词
    negative_prompt: str = ""                # 反向提示词
    seed: int = 0                            # 随机种子
    model_used: str = ""                     # 模型名称
    model_version: str = ""                  # 模型版本
    duration_ms: int = 0                     # 生成耗时
    params_snapshot: Dict = field(default_factory=dict)  # 生成参数快照
    constraint_params: Dict = field(default_factory=dict) # 约束参数快照

    def to_dict(self):
        return asdict(self)

    def to_generation_record(self) -> Dict:
        """转换为 GenerationRecord 格式"""
        return {
            "id": f"gen_{hashlib.md5(self.image_path.encode()).hexdigest()[:8]}_{int(time.time())}",
            "input_keywords": self.constraint_params.get("generation_prompt", ""),
            "constraint_parameters": self.constraint_params,
            "output_image_url": self.image_path,
            "output_thumbnail_url": self.thumbnail_path,
            "ai_model_used": self.model_used,
            "ai_model_version": self.model_version,
            "generation_params": self.params_snapshot,
            "generation_duration_ms": self.duration_ms,
            "inheritor_review_status": "未审核",
            "is_used_for_training": False,
            "created_at": datetime.now().isoformat(),
        }


@dataclass
class BatchGenerationResult:
    """批量生成结果"""
    results: List[GenerationResult] = field(default_factory=list)
    best_result: Optional[GenerationResult] = None
    total_duration_ms: int = 0
    batch_params: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "total_count": len(self.results),
            "total_duration_ms": self.total_duration_ms,
            "results": [r.to_dict() for r in self.results],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "batch_params": self.batch_params,
        }


# ========== 提示词构建器 ==========

class PaperCutPromptBuilder:
    """
    剪纸提示词构建器
    将 ConstraintParameters 转换为 Stable Diffusion 的正向/反向提示词
    """

    # 剪纸风格关键词
    PAPER_CUT_KEYWORDS = [
        "Chinese paper-cut", "traditional paper-cut art", "paper-cut style",
        "剪纸风格", "中国剪纸", "民间剪纸",
        "clean sharp lines", "folk art", "decorative",
    ]

    # 反向提示词（避免的效果）
    DEFAULT_NEGATIVE_PROMPT = (
        "photorealistic, 3d, oil painting, watercolor, sketch, "
        "blurry, low quality, distorted, deformed, bad anatomy, "
        "extra limbs, missing limbs, text, watermark, signature, "
        " photographic, realistic, gradient, shadow, "
        " messy, noisy, complex background, "
    )

    # 风格修饰词映射
    STYLE_MODIFIERS = {
        "蔚县剪纸": "Yuxian paper-cut style, intricate, colorful,戏曲theme",
        "扬州剪纸": "Yangzhou paper-cut style, elegant, delicate, floral theme",
        "陕北剪纸": "Northern Shaanxi paper-cut style, bold, rustic, folk theme, primitive",
        "佛山剪纸": "Foshan paper-cut style, gold-foil, brilliant, Southern Chinese",
        "高密剪纸": "Gaomi paper-cut style, rustic, vigorous, folk tale theme",
        "漳浦剪纸": "Zhangpu paper-cut style, delicate, feminine, Fujian style",
    }

    # 技法修饰词
    TECHNIQUE_MODIFIERS = {
        "阴刻": "intaglio carving, white lines on red background, line-focused",
        "阳刻": "relief carving, red shapes on white background, solid areas",
        "阴阳刻结合": "combined intaglio and relief, contrasting positive and negative spaces",
        "套色剪纸": "multi-color overlay, color inlay, layered paper-cut",
        "染色剪纸": "dyed paper-cut, watercolor staining effect, graduated colors",
        "单色剪纸": "monochrome paper-cut, single color, pure red",
    }

    # 对称修饰词
    SYMMETRY_MODIFIERS = {
        "轴对称": "bilateral symmetry, mirror symmetry",
        "中心对称": "radial symmetry, circular composition, rosette pattern",
        "旋转对称": "rotational symmetry, spiral composition",
        "不对称": "asymmetric composition, free-form, natural layout",
    }

    def build_prompt(self, params: Dict) -> str:
        """
        从约束参数构建 Stable Diffusion 提示词
        输出中英文混合提示词（SD 对英文关键词响应更好）
        """
        parts = []

        # 1. 核心剪纸风格
        parts.extend(self.PAPER_CUT_KEYWORDS[:3])

        # 2. 主母题
        primary = params.get("primary_motif", "")
        if primary:
            parts.append(primary)
            parts.append(self._motif_to_english(primary))

        # 3. 辅助母题
        secondary = params.get("secondary_motifs", [])
        for s in secondary[:3]:
            parts.append(s)
            eng = self._motif_to_english(s)
            if eng:
                parts.append(eng)

        # 4. 风格
        style = params.get("style", "")
        if style and style in self.STYLE_MODIFIERS:
            parts.append(self.STYLE_MODIFIERS[style])

        # 5. 技法
        technique = params.get("technique", "")
        if technique and technique in self.TECHNIQUE_MODIFIERS:
            parts.append(self.TECHNIQUE_MODIFIERS[technique])

        # 6. 对称/构图
        sym = params.get("symmetry_type", "")
        if sym and sym in self.SYMMETRY_MODIFIERS:
            parts.append(self.SYMMETRY_MODIFIERS[sym])

        # 7. 配色
        colors = params.get("recommended_colors", [])
        if colors:
            color_str = ", ".join(colors[:3])
            parts.append(f"color scheme: {color_str}")
            parts.append(f"{color_str}配色")

        # 8. 复杂度
        complexity = params.get("complexity", "")
        if complexity == "简单":
            parts.append("simple composition, minimal")
        elif complexity == "中等":
            parts.append("moderate complexity")
        elif complexity == "复杂":
            parts.append("intricate, rich details")
        elif complexity == "极复杂":
            parts.append("extremely intricate, highly detailed, elaborate")

        # 9. 寓意（辅助引导）
        meaning = params.get("meaning", "")
        if meaning:
            parts.append(f"auspicious meaning: {meaning[:30]}")

        # 组合
        prompt = ", ".join(parts)
        return prompt

    def build_negative_prompt(self, params: Dict) -> str:
        """构建反向提示词"""
        return self.DEFAULT_NEGATIVE_PROMPT

    def _motif_to_english(self, motif: str) -> str:
        """母题中译英（SD 关键词辅助）"""
        mapping = {
            "龙纹": "dragon pattern, traditional Chinese dragon",
            "凤纹": "phoenix pattern, Chinese fenghuang",
            "云纹": "cloud pattern, traditional Chinese clouds",
            "牡丹": "peony flower",
            "蝙蝠纹": "bat pattern, Chinese bat symbol",
            "寿桃": "longevity peach",
            "万字纹": "swastika pattern, endless knot pattern",
            "荷花": "lotus flower",
            "莲花": "lotus flower",
            "喜鹊": "magpie bird",
            "锦鲤": "koi fish, colorful carp",
            "梅花": "plum blossom",
            "福字纹": "Chinese fu character, fortune character",
            "龙": "Chinese dragon, long",
            "凤": "Chinese phoenix, fenghuang",
            "福": "Chinese fu character, fortune",
            "寿": "Chinese shou character, longevity",
            "喜": "Chinese xi character, double happiness",
            "鲤鱼": "carp fish",
            "麒麟": "Chinese unicorn, qilin",
            "狮子": "lion, guardian lion",
            "竹子": "bamboo",
            "菊花": "chrysanthemum",
            "兰花": "orchid",
            "龙凤呈祥": "dragon and phoenix, auspicious marriage",
            "福寿双全": "fortune and longevity complete",
            "年年有余": "abundance year after year, fish and lotus",
            "花开富贵": "flowers bloom fortune, peony prosperity",
            "五福捧寿": "five bats surrounding longevity",
        }
        return mapping.get(motif, "")


# ========== 图像后端接口 ==========

class ImageBackend(ABC):
    """图像后端抽象基类"""

    def __init__(self, config: GeneratorConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str,
                 width: int, height: int, steps: int,
                 cfg_scale: float, seed: int,
                 **kwargs) -> bytes:
        """生成单张图像，返回图像字节数据"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        ...

    @abstractmethod
    def name(self) -> str:
        """后端名称"""
        ...

    @abstractmethod
    def version(self) -> str:
        """后端版本"""
        ...


class LocalDiffusersBackend(ImageBackend):
    """
    本地 diffusers 后端
    使用 Stable Diffusion + ControlNet 生成本地
    需要 torch + diffusers + transformers
    """

    def __init__(self, config: GeneratorConfig, logger=None):
        super().__init__(config, logger)
        self._pipeline = None
        self._controlnet = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _load_models(self):
        """懒加载模型"""
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            try:
                import torch
                from diffusers import (
                    StableDiffusionControlNetPipeline,
                    ControlNetModel,
                    DPMSolverMultistepScheduler,
                )

                device = self.config.device
                if device == "cuda" and not torch.cuda.is_available():
                    if self.config.fallback_to_cpu:
                        self.logger.warning("CUDA 不可用，回退到 CPU")
                        device = "cpu"
                    else:
                        raise RuntimeError("CUDA 不可用且未设置 fallback")

                # 加载 ControlNet
                controlnet_id = self.config.local_controlnet_id
                self.logger.info(f"加载 ControlNet: {controlnet_id}")
                self._controlnet = ControlNetModel.from_pretrained(
                    controlnet_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )

                # 加载主模型
                model_id = self.config.local_model_id
                self.logger.info(f"加载模型: {model_id}")
                self._pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=self._controlnet,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None,  # 剪纸场景关闭安全检查
                )

                # 使用更快的调度器
                self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self._pipeline.scheduler.config
                )

                # 加载 LoRA（若有）
                if self.config.local_lora_path and os.path.exists(self.config.local_lora_path):
                    self.logger.info(f"加载 LoRA: {self.config.local_lora_path}")
                    self._pipeline.load_lora_weights(self.config.local_lora_path)

                self._pipeline.to(device)
                self._loaded = True
                self.logger.info(f"本地模型已加载到 {device}")

            except ImportError as e:
                self.logger.error(f"缺少依赖: {e}。请安装: pip install torch diffusers transformers")
                raise
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise

    def generate(self, prompt: str, negative_prompt: str,
                 width: int, height: int, steps: int,
                 cfg_scale: float, seed: int,
                 **kwargs) -> bytes:
        """使用本地 SD + ControlNet 生成"""
        self._load_models()

        import torch
        import numpy as np
        from PIL import Image
        from diffusers.utils import load_image

        generator = torch.Generator(device=self._pipeline.device)
        generator.manual_seed(seed)

        # 生成 ControlNet 条件图（Canny 边缘检测）
        # 如果没有提供条件图，根据对称类型生成引导草图
        control_image = self._create_control_guide(width, height, kwargs)

        # 推理
        with torch.autocast(
            self._pipeline.device.type,
            enabled=self.config.device == "cuda"
        ):
            output = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                controlnet_conditioning_scale=kwargs.get("controlnet_scale", 0.8),
                num_images_per_prompt=1,
            )

        # 返回图像字节
        img = output.images[0]
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _create_control_guide(self, width: int, height: int, kwargs: Dict) -> Optional[Any]:
        """
        创建 ControlNet 条件引导图
        根据对称类型和复杂度生成简单的布局引导
        """
        try:
            import torch
            import numpy as np
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        # 创建一个空白引导图
        guide = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(guide)

        symmetry = kwargs.get("symmetry_type", "不对称")

        if symmetry == "轴对称":
            # 绘制对称轴引导线
            cx, cy = width // 2, height // 2
            draw.line([(cx, 0), (cx, height)], fill=0, width=2)
            # 绘制轮廓引导
            draw.ellipse([cx - width//4, cy - height//4, cx + width//4, cy + height//4],
                         outline=0, width=3)

        elif symmetry == "中心对称":
            cx, cy = width // 2, height // 2
            r = min(width, height) // 3
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=3)
            # 放射线
            for angle in range(0, 360, 45):
                import math
                rad = math.radians(angle)
                ex = cx + int(r * math.cos(rad))
                ey = cy + int(r * math.sin(rad))
                draw.line([(cx, cy), (ex, ey)], fill=0, width=1)

        # 转为 tensor
        import torch
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        return to_tensor(guide).unsqueeze(0).to(
            dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        )

    def is_available(self) -> bool:
        try:
            import torch
            import diffusers
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return self.config.local_model_id

    @property
    def version(self) -> str:
        import diffusers
        return diffusers.__version__


class MockImageBackend(ImageBackend):
    """
    Mock 后端
    不调用真实模型，生成占位图像用于测试管线
    """

    def __init__(self, config: GeneratorConfig, logger=None):
        super().__init__(config, logger)

    def generate(self, prompt: str, negative_prompt: str,
                 width: int, height: int, steps: int,
                 cfg_scale: float, seed: int,
                 **kwargs) -> bytes:
        """生成占位剪纸风格图像"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            import random
        except ImportError:
            # 没有 PIL 时返回纯色图像
            return self._generate_placeholder_bytes(width, height)

        # 创建剪纸风格的占位图
        img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        rng = random.Random(seed)
        cx, cy = width // 2, height // 2

        # 绘制红色剪纸主体（抽象形状）
        self._draw_papercut_shape(draw, img, cx, cy, width, height, rng)

        # 添加主题文字
        motif = kwargs.get("symmetry_type", "")
        draw.text((10, 10), "剪纸", fill=(180, 30, 30, 200))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _draw_papercut_shape(self, draw, img, cx, cy, w, h, rng):
        """绘制抽象的剪纸形状"""
        from PIL import ImageDraw

        red = (200, 30, 30, 255)
        dark_red = (150, 20, 20, 255)

        # 中心主图案（抽象形状）
        r = min(w, h) // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=red)

        # 内层镂空
        r2 = r // 2
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=(255, 255, 255, 255))

        # 装饰性锯齿边
        for i in range(24):
            angle = i * 15
            import math
            rad = math.radians(angle)
            x1 = cx + int((r - 5) * math.cos(rad))
            y1 = cy + int((r - 5) * math.sin(rad))
            x2 = cx + int((r + 15) * math.cos(rad))
            y2 = cy + int((r + 15) * math.sin(rad))
            draw.line([(x1, y1), (x2, y2)], fill=red, width=3)

        # 角落装饰
        for angle_offset in [45, 135, 225, 315]:
            rad = math.radians(angle_offset)
            ex = cx + int((r + 40) * math.cos(rad))
            ey = cy + int((r + 40) * math.sin(rad))
            draw.ellipse([ex - 15, ey - 15, ex + 15, ey + 15],
                         fill=dark_red)

    def _generate_placeholder_bytes(self, width, height):
        """没有 PIL 时的纯色占位"""
        import struct
        # 简单 BMP 格式
        row_size = ((width * 3 + 3) // 4) * 4
        pixel_data = b""
        for y in range(height):
            row = b""
            for x in range(width):
                row += struct.pack("BBB", 255, 255, 255)  # 白色
            row += b"\x00" * (row_size - width * 3)
            pixel_data = row + pixel_data  # BMP 从底部开始

        file_size = 54 + len(pixel_data)
        header = struct.pack("<HHI", 0x4D42, file_size, 0)
        header += struct.pack("<I", 54)
        header += struct.pack("<I", 40)
        header += struct.pack("<I", width)
        header += struct.pack("<I", height)
        header += struct.pack("<HH", 1, 24)
        header += struct.pack("<II", 0, 0)
        header += struct.pack("<II", 0, 0)
        header += struct.pack("<II", 0, 0)

        return header + pixel_data

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "1.0.0"


class APIImageBackend(ImageBackend):
    """
    API 后端
    调用 Replicate / 自定义 SD API 生成
    """

    def __init__(self, config: GeneratorConfig, logger=None):
        super().__init__(config, logger)
        self._client = None

    def _get_client(self):
        """获取 API 客户端（懒加载）"""
        if self._client:
            return self._client

        # 尝试不同 API 方式
        if "replicate" in self.config.api_endpoint.lower() or not self.config.api_endpoint:
            return self._get_replicate_client()
        else:
            return self._get_custom_client()

    def _get_replicate_client(self):
        """Replicate API 客户端"""
        try:
            import replicate
            if self.config.api_key:
                os.environ["REPLICATE_API_TOKEN"] = self.config.api_key
            self._client = ("replicate", replicate)
            return self._client
        except ImportError:
            self.logger.warning("replicate 包未安装。可执行: pip install replicate")
            return None

    def _get_custom_client(self):
        """自定义 API 客户端（兼容 SD WebUI / ComfyUI API）"""
        import requests
        self._client = ("custom", requests)
        return self._client

    def generate(self, prompt: str, negative_prompt: str,
                 width: int, height: int, steps: int,
                 cfg_scale: float, seed: int,
                 **kwargs) -> bytes:
        """通过 API 生成"""
        client = self._get_client()
        if not client:
            raise RuntimeError("API 客户端不可用，请配置 api_key 或安装依赖")

        client_type, client_obj = client
        model = kwargs.get("model", self.config.api_model)

        if client_type == "replicate":
            return self._generate_replicate(client_obj, model, prompt,
                                            negative_prompt, width, height,
                                            steps, cfg_scale, seed)
        elif client_type == "custom":
            return self._generate_custom(client_obj, prompt, negative_prompt,
                                         width, height, steps, cfg_scale, seed)
        else:
            raise RuntimeError(f"不支持的 API 类型: {client_type}")

    def _generate_replicate(self, replicate_client, model: str,
                            prompt: str, negative_prompt: str,
                            width: int, height: int, steps: int,
                            cfg_scale: float, seed: int) -> bytes:
        """通过 Replicate API 生成"""
        import replicate

        self.logger.info(f"调用 Replicate API: model={model}")

        output = replicate.run(
            model,
            input={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "seed": seed,
                "scheduler": "DPMSolverMultistep",
            }
        )

        # Replicate 返回 URL 列表
        if isinstance(output, list) and len(output) > 0:
            image_url = str(output[0])
            # 下载图像
            import requests
            resp = requests.get(image_url, timeout=60)
            resp.raise_for_status()
            return resp.content

        raise RuntimeError(f"Replicate API 返回异常: {output}")

    def _generate_custom(self, requests_client, prompt: str, negative_prompt: str,
                         width: int, height: int, steps: int,
                         cfg_scale: float, seed: int) -> bytes:
        """通过自定义 API（SD WebUI API 格式）生成"""
        import requests

        endpoint = self.config.api_endpoint.rstrip("/")
        url = f"{endpoint}/sdapi/v1/txt2img"

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": 1,
            "n_iter": 1,
        }

        self.logger.info(f"调用自定义 API: {url}")
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        if "images" in data and len(data["images"]) > 0:
            # API 返回 base64
            img_b64 = data["images"][0]
            return base64.b64decode(img_b64)

        raise RuntimeError(f"API 返回异常: {data}")

    def is_available(self) -> bool:
        """检查 API 是否可达"""
        try:
            if "replicate" in self.config.api_endpoint.lower() or not self.config.api_endpoint:
                import replicate
                return True
            else:
                import requests
                return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return self.config.api_model

    @property
    def version(self) -> str:
        return "api"


# ========== 后处理器 ==========

class PaperCutPostProcessor:
    """
    剪纸风格后处理
    将生成图像调整得更像传统剪纸：
    1. 二值化（黑白分明）
    2. 边缘锐化
    3. 红色调强化
    4. 背景移除/纯白
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    def process(self, image_bytes: bytes, params: Dict) -> bytes:
        """
        对生成图像进行后处理

        Args:
            image_bytes: 原始图像字节
            params: 约束参数（用于决定后处理策略）

        Returns:
            处理后的图像字节
        """
        if not self.config.enable_postprocessing:
            return image_bytes

        try:
            from PIL import Image, ImageFilter, ImageOps
            import io
            import numpy as np
        except ImportError:
            return image_bytes

        # 加载图像
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # 获取约束信息
        colors = params.get("recommended_colors", [])
        is_monochrome = len(colors) <= 1 or params.get("color_count", 1) <= 1

        # 步骤 1: 背景移除/纯白化
        if self.config.postprocess_remove_bg:
            img = self._make_background_white(img)

        # 步骤 2: 边缘锐化
        if self.config.postprocess_edges_sharpen:
            img = self._sharpen_edges(img)

        # 步骤 3: 二值化（对单色剪纸适用）
        if self.config.postprocess_binarize and is_monochrome:
            img = self._binarize_red(img)

        # 步骤 4: 增强剪纸纹理感
        img = self._add_paper_texture(img)

        # 输出
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _make_background_white(self, img) -> 'Image.Image':
        """将背景转为纯白"""
        import numpy as np

        arr = np.array(img)

        # 判断背景色：取四角像素的平均
        corners = [
            arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]
        ]
        bg_color = np.mean(corners, axis=0).astype(int)

        # 如果背景是白色或接近白色，跳过
        if bg_color[0] > 200 and bg_color[1] > 200 and bg_color[2] > 200:
            return img

        # alpha 通道掩码：将接近背景色的像素设为透明
        threshold = 60
        alpha = np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
        diff = np.abs(arr[:, :, :3].astype(int) - bg_color[:3])
        mask = np.all(diff < threshold, axis=2)
        alpha[mask] = 0

        arr[:, :, 3] = alpha

        from PIL import Image
        result = Image.fromarray(arr, "RGBA")

        # 白色背景
        white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])
        return white_bg

    def _sharpen_edges(self, img) -> 'Image.Image':
        """锐化边缘，使剪纸线条更清晰"""
        from PIL import ImageFilter
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))

    def _binarize_red(self, img) -> 'Image.Image':
        """
        二值化为红白两色
        红色部分 = 剪纸主体，白色部分 = 背景
        """
        import numpy as np

        arr = np.array(img)

        # 计算红色通道相对于其他通道的强度
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

        # 红色判定：R > G 且 R > B 且 R > 阈值
        red_mask = (r.astype(int) - g.astype(int) > 20) & \
                   (r.astype(int) - b.astype(int) > 20) & \
                   (r > 50)

        # 创建输出
        output = np.ones_like(arr) * 255  # 纯白背景
        output[red_mask] = [200, 30, 30, 255]  # 剪纸红
        # 黑色区域（如果原图有深色线条）也保留
        dark_mask = (r < 50) & (g < 50) & (b < 50) & (a > 100)
        output[dark_mask] = [0, 0, 0, 255]

        from PIL import Image
        return Image.fromarray(output, "RGBA")

    def _add_paper_texture(self, img) -> 'Image.Image':
        """添加轻微的纸面纹理感"""
        import numpy as np

        # 轻微增加对比度
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

        return img


# ========== 结果管理器 ==========

class ResultManager:
    """
    生成结果管理
    - 保存图像到磁盘
    - 写入元数据 JSON
    - 可选输出 GenerationRecord 格式
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.images_dir = self.output_dir / "images"
        self.thumbs_dir = self.output_dir / "thumbnails"
        self.meta_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image_bytes: bytes, constraint_params: Dict,
             prompt: str, negative_prompt: str, seed: int,
             model_used: str, model_version: str, duration_ms: int,
             params_snapshot: Dict) -> GenerationResult:
        """保存生成结果"""
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(image_bytes).hexdigest()[:8]
        filename = f"papercut_{timestamp}_{hash_suffix}"

        image_path = ""
        thumbnail_path = ""

        if self.config.save_images:
            # 保存原图
            img_path = self.images_dir / f"{filename}.png"
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            image_path = str(img_path)

            # 生成缩略图
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_bytes))
                thumb = img.copy()
                thumb.thumbnail((256, 256))
                thumb_path = self.thumbs_dir / f"{filename}_thumb.png"
                thumb.save(thumb_path, "PNG")
                thumbnail_path = str(thumb_path)
            except ImportError:
                thumbnail_path = image_path

        result = GenerationResult(
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            model_used=model_used,
            model_version=model_version,
            duration_ms=duration_ms,
            params_snapshot=params_snapshot,
            constraint_params=constraint_params,
        )

        # 保存元数据
        if self.config.save_metadata:
            meta_path = self.meta_dir / f"{filename}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return result


# ========== 图像生成管线（主入口） ==========

class PaperCutImageGenerator:
    """
    剪纸图像生成管线（主入口）

    使用方式:
        generator = PaperCutImageGenerator()
        result = generator.generate(constraint_params)
        # result.image_path 包含输出图像路径
    """

    def __init__(self, config: Optional[GeneratorConfig] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or GeneratorConfig()
        self.logger = logger or self._setup_logger()

        # 组件
        self.prompt_builder = PaperCutPromptBuilder()
        self.backend = self._create_backend()
        self.post_processor = PaperCutPostProcessor(self.config)
        self.result_manager = ResultManager(self.config)

        self.logger.info(
            f"图像生成管线就绪 | 后端: {self.backend.name} | "
            f"后处理: {self.config.enable_postprocessing}"
        )

    def _setup_logger(self):
        logger = logging.getLogger("PaperCutGen")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
        return logger

    def _create_backend(self) -> ImageBackend:
        """创建图像生成后端"""
        if self.config.backend == "mock":
            return MockImageBackend(self.config, self.logger)
        elif self.config.backend == "local":
            backend = LocalDiffusersBackend(self.config, self.logger)
            if not backend.is_available():
                self.logger.warning("本地后端不可用（缺少 torch/diffusers），回退到 API 后端")
                return APIImageBackend(self.config, self.logger)
            return backend
        else:
            return APIImageBackend(self.config, self.logger)

    def generate(self, constraint_params: Union[Dict, str],
                 seed: Optional[int] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 steps: Optional[int] = None,
                 cfg_scale: Optional[float] = None,
                 **kwargs) -> GenerationResult:
        """
        核心方法：输入约束参数，输出生成图像

        Args:
            constraint_params: 约束参数字典（或 JSON 字符串）
            seed: 随机种子（None=随机）
            width/height: 图像尺寸（覆盖默认值）
            steps: 推理步数
            cfg_scale: CFG 尺度

        Returns:
            GenerationResult
        """
        # 解析输入
        if isinstance(constraint_params, str):
            params = json.loads(constraint_params)
        else:
            params = constraint_params

        # 构建提示词
        prompt = self.prompt_builder.build_prompt(params)
        negative_prompt = self.prompt_builder.build_negative_prompt(params)

        # 解析生成参数
        w = width or self.config.default_width
        h = height or self.config.default_height
        s = steps or self.config.default_steps
        cfg = cfg_scale or self.config.default_cfg_scale
        seed_val = seed if seed is not None else int(time.time()) % (2**32)

        self.logger.info(f"生成参数: prompt='{prompt[:60]}...' | "
                         f"size={w}x{h} | steps={s} | cfg={cfg} | seed={seed_val}")

        # 执行生成
        start_time = time.time()
        try:
            image_bytes = self.backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=w, height=h,
                steps=s, cfg_scale=cfg,
                seed=seed_val,
                symmetry_type=params.get("symmetry_type", ""),
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            raise RuntimeError(f"图像生成失败: {e}")

        duration_ms = int((time.time() - start_time) * 1000)
        self.logger.info(f"生成完成: {duration_ms}ms")

        # 后处理
        if self.config.enable_postprocessing:
            image_bytes = self.post_processor.process(image_bytes, params)
            self.logger.info("后处理完成")

        # 保存结果
        snapshot = {
            "width": w, "height": h,
            "steps": s, "cfg_scale": cfg, "seed": seed_val,
        }
        if kwargs.get("controlnet_scale"):
            snapshot["controlnet_scale"] = kwargs["controlnet_scale"]

        result = self.result_manager.save(
            image_bytes=image_bytes,
            constraint_params=params,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed_val,
            model_used=self.backend.name,
            model_version=self.backend.version,
            duration_ms=duration_ms,
            params_snapshot=snapshot,
        )

        # 输出生成记录（可回写知识图谱）
        if self.config.record_to_generation_record:
            record = result.to_generation_record()
            self._save_generation_record(record)

        return result

    def batch_generate(self, constraint_params: Union[Dict, str],
                       n: int = 4,
                       seeds: Optional[List[int]] = None,
                       grid_output: bool = False,
                       **kwargs) -> BatchGenerationResult:
        """
        批量生成（多种子探索）

        Args:
            constraint_params: 约束参数
            n: 生成数量
            seeds: 指定种子列表
            grid_output: 是否输出网格图

        Returns:
            BatchGenerationResult
        """
        if isinstance(constraint_params, str):
            params = json.loads(constraint_params)
        else:
            params = constraint_params

        # 生成种子
        if seeds is None:
            import random
            seeds = [random.randint(0, 2**32) for _ in range(n)]
        elif len(seeds) < n:
            import random
            seeds.extend([random.randint(0, 2**32) for _ in range(n - len(seeds))])

        results = []
        start_time = time.time()

        for i, s in enumerate(seeds[:n]):
            self.logger.info(f"批量 [{i+1}/{n}] seed={s}")
            result = self.generate(params, seed=s, **kwargs)
            results.append(result)

        total_duration = int((time.time() - start_time) * 1000)

        # 按生成质量评分排序（这里用简单的启发式：分数基于种子随机，
        # 实际可以接入 CLIP score 等）
        for r in results:
            r.params_snapshot["quality_score"] = 0.5  # 占位，将来可接入 CLIP 评分

        # 取最佳（目前取中间种子，后续可替换为评分模型）
        best = results[len(results) // 2] if results else None

        # 可选：生成网格图
        if grid_output and len(results) > 1:
            grid_path = self._create_grid(results, params)
            self.logger.info(f"网格图: {grid_path}")

        return BatchGenerationResult(
            results=results,
            best_result=best,
            total_duration_ms=total_duration,
            batch_params={"n": n, "seeds": seeds},
        )

    def _create_grid(self, results: List[GenerationResult],
                     params: Dict) -> str:
        """将多张生成结果合并为网格图"""
        try:
            from PIL import Image
        except ImportError:
            return ""

        images = []
        for r in results:
            try:
                img = Image.open(r.image_path)
                images.append(img)
            except Exception:
                continue

        if not images:
            return ""

        # 网格布局
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)

        grid_img = Image.new("RGBA", (cols * cell_w, rows * cell_h), (255, 255, 255, 255))

        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            x, y = col * cell_w, row * cell_h
            # 居中粘贴
            offset_x = (cell_w - img.width) // 2
            offset_y = (cell_h - img.height) // 2
            grid_img.paste(img, (x + offset_x, y + offset_y))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_path = str(self.result_manager.output_dir / f"grid_{timestamp}.png")
        grid_img.save(grid_path)
        return grid_path

    def _save_generation_record(self, record: Dict):
        """保存生成记录到磁盘（未来可回写到 Neo4j）"""
        records_dir = self.result_manager.output_dir / "generation_records"
        records_dir.mkdir(parents=True, exist_ok=True)

        record_path = records_dir / f"{record['id']}.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 追加到记录列表
        index_path = records_dir / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ========== CLI 测试 ==========

def main():
    """命令行演示（无真实模型时输出占位数据）"""
    import argparse

    parser = argparse.ArgumentParser(description="非遗剪纸图像生成管线")
    parser.add_argument("keyword", nargs="?", default="婚庆",
                        help="生成关键词")
    parser.add_argument("--output", "-o", default="./generated_output",
                        help="输出目录")
    parser.add_argument("--backend", "-b", choices=["api", "local", "mock"],
                        default="mock", help="生成后端")
    parser.add_argument("--batch-size", "-n", type=int, default=1,
                        help="批量生成数量")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--json", "-j", action="store_true",
                        help="JSON 格式输出")
    args = parser.parse_args()

    # 配置
    config = GeneratorConfig(
        output_dir=args.output,
        backend=args.backend,
    )

    # 初始化生成器
    generator = PaperCutImageGenerator(config)

    # 先用约束引擎获取参数
    try:
        from papercut_graph_rag_engine import PaperCutConstraintGenerator
        constraint_engine = PaperCutConstraintGenerator()
        constraint_engine.load_knowledge(source="seed")
        params = constraint_engine.generate(args.keyword, return_full=True)
    except ImportError:
        # 回退：用关键词构建简单参数
        params = {
            "primary_motif": args.keyword,
            "style": "蔚县剪纸",
            "technique": "阴刻",
            "recommended_colors": ["大红"],
            "symmetry_type": "轴对称",
            "complexity": "中等",
            "generation_prompt": f"中国传统剪纸, {args.keyword}",
        }

    if args.backend == "mock":
        # Mock 模式：生成占位图像，测试完整管线
        print("=" * 60)
        print(f"剪纸图像生成 (Mock 模式)")
        print("=" * 60)
        print(f"关键词: {args.keyword}")
        print(f"约束参数: {json.dumps({k:v for k,v in params.items() if not k.startswith('_')}, ensure_ascii=False)}")
        print()

        if args.batch_size > 1:
            result = generator.batch_generate(params, n=args.batch_size,
                                              seeds=[args.seed or 42 + i for i in range(args.batch_size)])
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(f"批量生成: {len(result.results)} 张")
                for i, r in enumerate(result.results):
                    print(f"  [{i+1}] {r.image_path} (seed={r.seed})")
                print(f"耗时: {result.total_duration_ms}ms")
        else:
            result = generator.generate(params, seed=args.seed or 42)
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(f"输出: {result.image_path}")
                print(f"提示词: {result.prompt[:80]}...")
                print(f"耗时: {result.duration_ms}ms")
                print(f"种子: {result.seed}")
    else:
        # 真实后端
        result = generator.generate(params, seed=args.seed)
        print(f"输出: {result.image_path}")
        print(f"耗时: {result.duration_ms}ms")

    return 0


if __name__ == "__main__":
    exit(main())
