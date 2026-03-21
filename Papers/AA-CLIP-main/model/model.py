import collections.abc
import logging
import math
from dataclasses import dataclass
from itertools import repeat
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .modified_resnet import ModifiedResNet
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer


@dataclass
class CLIPVisionCfg:
    # 图像塔配置：描述 Vision backbone 的结构超参数
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.2  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256 # n_queries for attentional pooler
    attn_pooler_heads: int = 8 # n heads for attentional_pooling
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    output_tokens: bool = True


@dataclass
class CLIPTextCfg:
    # 文本塔配置：描述 Text backbone 的结构超参数
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
# 两个配置类提供塔(独立的特征提取器 也就是提取出embed vector的那个神经网络)

def get_cast_dtype(precision: str):
    # 根据训练/推理精度字符串，返回模型内部计算 dtype
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype
# 根具数据的precision 返回不同的类型

def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    """
    :param embed_dim: 映射的维度 没什么好说的
    :param vision_cfg: 配置 可能是Resnet或者Vit 他会根据不同给出不同的输出
    :param quick_gelu: 一个激活函数
    :param cast_dtype:
    :return: 初始化好的特征提取器
    """
    # 允许传 dict 或 dataclass，两者统一成 CLIPVisionCfg
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    # 处理兼容性 简单来说就是如果传过来的是字典就会自动帮你拆开转成CLIPVisionCfg这个配置对象
    act_layer = QuickGELU if quick_gelu else nn.GELU # 激活层 quick gelu 为OpenAi提出的一种快速计算GELU激活函数的近似
    if isinstance(vision_cfg.layers, (tuple, list)):
        # layers 是 tuple/list => 走 ResNet 分支（兼容老结构）

        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    # Resnet 提供各个Stage的层数 但是CLIP测出来transformer性能更好 这里是为了兼容
    else:
        # layers 是 int => 走 ViT 分支（本项目主路径）
        vision_heads = vision_cfg.width // vision_cfg.head_width# 注意力头的数量
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        # 在计算归一化的时候 由于要计算方差 导致要平方 非常容易爆float 所以调用LayerNorm时将其转为float32
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    """
    :param embed_dim: 
    :param text_cfg:
    :param quick_gelu:
    :param cast_dtype:
    :return: 初始化文本特征提取器
    """

    # 统一配置对象类型，便于后续直接读取属性
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    # 返回 TextTransformer 实例（负责 token -> sentence embedding）
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        # CLIP 为 图像塔 + 文本塔 + 一个可学习的温度参数 logit_scale
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        # 创建图像塔
        # 这里先构建完整 text tower，然后把关键子模块挂到 CLIP 顶层
        # 这样 encode_text() 就能直接按 OpenAI 风格实现
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        # 创建文本塔
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        # logit_scale 即为温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, out_layers, normalize: bool = False):
        features = self.visual(image, out_layers)# Image地Encoder 
        return F.normalize(features, dim=-1) if normalize else features
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 经过文本 Transformer，得到每个 token 的上下文特征
        x, tokens = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 取 EOT 位置作为句向量，再映射到共享对齐空间
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
    def forward(self, image, text):
        # 同时编码图像与文本，输出用于相似度计算的向量
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        # 与 CLIP 类似，但把 text 塔作为独立子模块 self.text 暴露出来
        # 某些场景下更方便替换/锁定文本分支
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()
        # 返回embed后的image_features text_features和超参数
def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""
    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)
        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)
    model.apply(_convert_weights)
convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    # 兼容旧 checkpoint：把顶层 text 参数重命名为 text.xxx
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    # 从 OpenAI 风格 state_dict 反推出模型结构超参数，再实例化 CLIP
    # 这是“只给权重文件也能重建模型结构”的关键函数
    vit = "visual.proj" in state_dict

    if vit:
        # ViT 分支：通过 conv/positional embedding 反推 patch_size 与 image_size
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        # ResNet 分支：通过 layerX 的 block 数推回网络深度
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    # 文本塔超参数也从权重形状里推导
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    # OpenAI 发布权重常混合低精度，先对齐再 load
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    # 导出 TorchScript trace 版本（部署/加速场景可用）
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    # 若当前输入分辨率对应的 patch 数没变，则不需要插值
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    # 核心步骤：把 old pos_embed 当作二维网格插值到新网格，再拼回 CLS token
    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed