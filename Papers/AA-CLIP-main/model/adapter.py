import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj

class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        **kwargs,
    ):
        """
        :param clip_model: 已经加载完原本参数且打包好的本地CLIP模型
        :param text_adapt_weight:
        :param image_adapt_weight:
        :param text_adapt_until:
        :param image_adapt_until: 给Vit前几层插入一个Adapter
        :param levels: 从Vit的哪些层导出patch token
        :param relu:
        :param kwargs:
        """
        super().__init__()
        # 这里的 clip_model 是已经加载好预训练权重的 CLIP
        # AdaptedCLIP 的目标不是重写 CLIP，而是“在少量位置加可训练小模块”
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels
        # 默认 [6,12,18,24]，越后层语义越强，前层细节更多

        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        # image layer adapter:
        # 给视觉前 image_adapt_until 层各插一个 Adapter
        # 只改前几层，不大幅破坏原 CLIP 分布
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        # seg_proj:
        # 每个导出层各有一个投影头，把 token 维度投到 768
        # 这些 token 后续用于像素级异常定位（segmentation 分支）
        det_proj = SimpleProj(1024, 768, relu)
        # det_proj:
        # 图像级检测头，输入最后层 token，输出图像级语义向量（detection 分支）
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )
        # 将上面的多个adapter和projector打包
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        # text_adapter 前 text_adapt_until 个用于“层内适配”
        # 最后一个 SimpleProj 用于最终句向量投影
        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 使用xavier初始化参数

    def forward_original(self, x, modality="visual"):
        # 对照用 走原始 CLIP
        # 主要用于调试/比较改造前后的特征
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x):
        # -----------------------------
        # 图像分支主前向（用于 stage2 image adapter 训练）
        # 输入: x [B,3,H,W]
        # 输出:
        #   seg_tokens: list[[B,N_patch,768]]  多尺度 patch 特征（定位）
        #   det_token:  [B,768]                图像级特征（分类）
        # -----------------------------
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # 现在 x 形状为 [B, N_patch, C]

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        # 嵌入位置向量
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)
        # 进入 transformer block 前切成 [L,B,C]

        tokens = [] # 存储要作为输出的patch
        for i in range(24):
            # 逐层过视觉 transformer
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None) # 先求出在CLIP模型中的输出
            if i < self.image_adapt_until:
                # 如果当前层数<要注入的总层数
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                # 把 adapt_out 的范数对齐到原特征范数，稳定训练
                # 避免因 adapter 输出幅值过大直接导致CLIP模型原先的泛用性损失
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                # 残差混合：x_new = i_w * adapter + (1-i_w) * original 也就是论文中提到的超参数
                # i_w 越大，适配强度越高；越小，越保留原始 CLIP 能力
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                # 当到了要输出patch的时候 就去掉 CLS token 保留 patch token 给分割定位分支
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        # tokens 每个元素形状: [B, N_patch, 1024]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        # 对提取到的patch做个投影
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens] # 最后归一化
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        # det_token:
        # 先投影到 768 并归一化，再对 patch 维做平均，得到图像级向量 [B,768]
        return seg_tokens, det_token

    def encode_text(self, text, adapt_text=True):
        # -----------------------------
        # 文本分支主前向（stage1 text adapter 训练 + 推理取文本锚点）
        # text: [B, L]
        # 返回: [B, 768]
        # -----------------------------
        if not adapt_text:
            # 关闭适配时，直接复用原始 CLIP 文本编码器（zero-shot 基线）
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                # 文本端与图像端一样：前几层插 adapter 做轻量语义偏移
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                # t_w 控制文本适配强度
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        # 这里取 EOT token 作为句向量，再过最后一个投影头
        # 输出是“异常感知后的文本锚点向量”，后续会和 patch/image 特征做相似度
        # x = (
            # x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            # @ self.clipmodel.text_projection
        # )
        return x