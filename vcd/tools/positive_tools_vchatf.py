import cv2
import os
import numpy as np
import math
import torch
from transformers import AutoModel, AutoImageProcessor
import json
import timm
import types
from typing import Literal
from PIL import Image


class MotionSaliencyExtractor:
    """
    运动显著性检测器，通过光流法计算视频帧的运动显著性图
    
    主要步骤：
    1. 使用光流算法（TV-L1, Farneback 或 RAFT）计算帧间的运动场
    2. 应用时域带通滤波/邻域一致性分析来抑制背景噪声和抖动
    3. 生成动作热度图
    """

    def __init__(self, method='farneback', spatial_filter_sigma=1.0, temporal_window_size=2, low_freq_threshold=0.1, high_freq_threshold=0.9):
        """
        初始化运动显著性检测器
        
        Args:
            method: 光流计算方法 ('farneback', 'tvl1')
            spatial_filter_sigma: 空间高斯滤波的标准差，用于平滑光流场
            temporal_window_size: 时域滤波窗口大小
            low_freq_threshold: 低频阈值，用于抑制缓慢背景运动
            high_freq_threshold: 高频阈值，用于抑制高频噪声
        """
        self.method = method
        self.spatial_filter_sigma = spatial_filter_sigma
        self.temporal_window_size = temporal_window_size
        self.low_freq_threshold = low_freq_threshold
        self.high_freq_threshold = high_freq_threshold
        
        # 初始化光流计算器
        if method == 'farneback':
            self.optical_flow = None  # Farneback算法不需要特殊初始化
        elif method == 'tvl1':
            self.optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        else:
            raise ValueError(f"不支持的光流方法: {method}")
    
    def _calculate_optical_flow(self, prev_frame, curr_frame):
        """
        计算两帧间的光流场
        
        Args:
            prev_frame: 前一帧 (H, W, 3) RGB格式
            curr_frame: 当前帧 (H, W, 3) RGB格式
            
        Returns:
            flow: 光流场 (H, W, 2)
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        if self.method == 'farneback':
            # 使用Farneback算法计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif self.method == 'tvl1':
            # 使用TV-L1算法计算光流
            flow = self.optical_flow.calc(prev_gray, curr_gray, None)
        return flow
    
    def _compute_motion_magnitude(self, flow):
        """
        计算光流场的幅度图
        
        Args:
            flow: 光流场 (H, W, 2)
            
        Returns:
            magnitude: 运动幅度图 (H, W)
        """
        # 计算光流幅度
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude
    
    def _spatial_filter(self, motion_mag):
        """
        空间滤波，平滑运动幅度图
        
        Args:
            motion_mag: 运动幅度图 (H, W)
            
        Returns:
            filtered_mag: 滤波后的运动幅度图 (H, W)
        """
        # 应用高斯滤波平滑
        if self.spatial_filter_sigma > 0:
            filtered_mag = cv2.GaussianBlur(motion_mag, (0, 0), self.spatial_filter_sigma)
        else:
            filtered_mag = motion_mag
        return filtered_mag
    
    def _temporal_filter(self, motion_mags):
        """
        时域滤波，抑制背景噪声和抖动
        
        Args:
            motion_mags: 运动幅度图序列 list of (H, W)
            
        Returns:
            filtered_mags: 滤波后的运动幅度图序列 list of (H, W)
        """
        if len(motion_mags) < self.temporal_window_size:
            # 如果帧数不足，直接返回原图
            return motion_mags
            
        filtered_mags = []
        half_window = self.temporal_window_size // 2
        
        for i in range(len(motion_mags)):
            # 获取时域窗口内的帧
            start_idx = max(0, i - half_window)
            end_idx = min(len(motion_mags), i + half_window + 1)
            
            # 计算窗口内的统计信息
            window_mags = motion_mags[start_idx:end_idx]
            mean_mag = np.mean(window_mags, axis=0)
            std_mag = np.std(window_mags, axis=0)
            
            # 应用时域滤波：保留显著运动，抑制噪声
            curr_mag = motion_mags[i]
            # 抑制低于均值和低阈值的运动
            threshold = mean_mag + self.low_freq_threshold * std_mag
            filtered_mag = np.maximum(curr_mag - threshold, 0)
            
            # 抑制高频噪声
            high_threshold = mean_mag + self.high_freq_threshold * std_mag
            mask = curr_mag <= high_threshold
            filtered_mag = filtered_mag * mask
            
            filtered_mags.append(filtered_mag)
            
        return filtered_mags
    
    def _normalize_saliency_frame(self, motion_sals):
        """
        归一化显著性图到[0, 1]范围
        
        Args:
            motion_sals: 显著性图序列 list of (H, W)
            
        Returns:
            normalized_sals: 归一化后的显著性图序列 list of (H, W)
        """
        normalized_sals = []
        for sal in motion_sals:
            min_val = sal.min()
            max_val = sal.max()
            if max_val > min_val:
                normalized_sal = (sal - min_val) / (max_val - min_val)
            else:
                normalized_sal = np.zeros_like(sal)
            normalized_sals.append(normalized_sal.astype(np.float32))
        return normalized_sals
    
    def extract_motion_saliency(self, frames: list[np.ndarray], indices=None):
        """
        提取视频帧的运动显著性图
        
        Args:
            frames: 视频帧列表，每个元素是(H, W, 3)形状的NumPy数组 (RGB格式)
            indices: 采样帧的索引列表，如果为None则处理所有帧
            
        Returns:
            motion_sals: 运动显著性图序列 list of (H, W) float32 [0,1]
        """
        if indices is not None:
            # 如果提供了采样索引，则只处理这些索引对应的帧
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            # 否则处理所有帧
            sampled_frames = frames
        
        if len(sampled_frames) < 2:
            # 单帧或无帧情况，返回零显著性图
            return [np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32) for frame in sampled_frames]
        
        # 步骤1: 计算光流场和运动幅度
        motion_mags = []
        # 第一帧没有前一帧，运动幅度设为0
        motion_mags.append(np.zeros((sampled_frames[0].shape[0], sampled_frames[0].shape[1]), dtype=np.float32))
        
        # 计算后续帧的光流和运动幅度
        for i in range(1, len(sampled_frames)):
            flow = self._calculate_optical_flow(sampled_frames[i-1], sampled_frames[i])
            mag = self._compute_motion_magnitude(flow)
            motion_mags.append(mag)
        
        # 步骤2: 空间滤波
        filtered_mags = [self._spatial_filter(mag) for mag in motion_mags]
        
        # 步骤3: 时域滤波
        temporal_filtered_mags = self._temporal_filter(filtered_mags)
        motion_sals = temporal_filtered_mags
        
        return motion_sals
    
    def video_with_overlay(self, frames: list[np.ndarray], motion_sals, alpha=0.5, indices=None):
        """
        将运动显著性图转换为伪彩图并叠加到原帧上
        
        Args:
            frames: 原始视频帧列表
            motion_sals: 运动显著性图序列
            alpha: 显著性图叠加到原帧上的权重
            indices: 采样帧的索引列表，如果为None则处理所有帧
            
        Returns:
            heatmaps: 伪彩显著性图列表
            overlays: 显著性图叠加到原帧上的图像列表
        """
        heatmaps = []
        overlays = []
        motion_sals = self._normalize_saliency_frame(motion_sals)
        
        if indices is not None:
            # 如果提供了采样索引，则只处理这些索引对应的帧
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            # 否则处理所有帧
            sampled_frames = frames
            
        for frame, sal in zip(sampled_frames, motion_sals):
            # 将显著性图转换为伪彩图
            sal_u8 = (np.clip(sal, 0.0, 1.0) * 255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
            # 转换为RGB格式
            heatmap_rgb = heatmap[..., ::-1].copy()
            heatmaps.append(heatmap_rgb)
            
            # 将显著性图叠加到原帧上
            if frame.shape[-1] == 3:
                frame_rgb = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame_rgb = frame
                
            overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            overlays.append(overlay)
            
        return heatmaps, overlays


class DINOv3SaliencyExtractor:
    """
    使用 DINOv3 的 CLS->patch attention 生成每帧的伪彩 RGB 显著图 + 叠加结果

    返回：
        raw_sals: List[np.ndarray]，灰度显著图，float32 [0,1]
        heatmaps: List[np.ndarray]，伪彩 RGB 显著图，uint8
        overlays: List[np.ndarray]，显著图叠加到原帧上的图像，uint8
    """
    def __init__(self, checkpoints: str = "../../checkpoints/DINOv3/dinov3-vitl16-pretrain-lvd1689m", device: str = "cuda:6", batch_size: int = 12, alpha: float = 0.5, attn: Literal["last", "rollout"] = "last"):
        self.device = device 
        self.math = math
                
        config_path = f"{checkpoints}/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_name = config.get("_name_or_path", "vit_large_patch16_dinov3.lvd1689m")
        self.model = timm.create_model(model_name, pretrained=False)
        
        checkpoint_path = f"{checkpoints}/pytorch_model.bin"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_dict = self.model.state_dict()
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
            self.model.load_state_dict(filtered_checkpoint, strict=False)
        
        self.model = self.model.to(self.device).eval()
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.val_transforms = timm.data.create_transform(**self.data_config, is_training=False)
        self.bs = batch_size
        self.alpha = alpha
        assert attn in ("last", "rollout")
        self.attn = attn

        def enable_last_attn_capture_eva(vit_model):
            """
            兼容 timm EVA/EVA02 Attention 的最后一层注意力捕获。
            缓存字段：vit_model.blocks[-1].attn.last_attn  (B, H, N, N)
            """
            last_block = vit_model.blocks[-1]
            attn_mod = last_block.attn

            if getattr(attn_mod, "_capture_enabled", False):
                return

            # EVA Attention 常见字段：qkv, num_heads, head_dim/scale, attn_drop, proj, proj_drop
            # rope 用于对 q/k 做旋转位置编码；如果提供 rope，我们尽量调用其 apply_rotary_emb
            def forward_with_capture(self_attn, x, rope=None, attn_mask=None, **kwargs):
                B, N, C = x.shape

                qkv = self_attn.qkv(x)  # (B, N, 3*C)
                qkv = qkv.reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # 处理 RoPE：timm EVA 的 rope 往往是一个可调用模块或包含 apply_rotary_emb 的对象
                if rope is not None:
                    # 常见：rope(q, k) -> (q, k) 或 rope.apply_rotary_emb(q, k) -> (q, k)
                    if callable(rope):
                        q, k = rope(q, k)
                    elif hasattr(rope, "apply_rotary_emb"):
                        q, k = rope.apply_rotary_emb(q, k)
                    else:
                        # 如果 rope 类型不符合预期，宁可跳过也不要报错
                        pass

                attn = (q @ k.transpose(-2, -1)) * getattr(self_attn, "scale", 1.0)  # (B, H, N, N)

                # 注意：attn_mask 在 EVA 里可能是 additive mask（-inf）或 bool mask
                if attn_mask is not None:
                    # 尝试兼容两类 mask
                    if attn_mask.dtype == torch.bool:
                        attn = attn.masked_fill(attn_mask, float("-inf"))
                    else:
                        attn = attn + attn_mask

                attn = attn.softmax(dim=-1)

                # 缓存 softmax 后注意力
                self_attn.last_attn = attn

                attn = self_attn.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self_attn.proj(x)
                x = self_attn.proj_drop(x)
                return x

            attn_mod.forward = types.MethodType(forward_with_capture, attn_mod)
            attn_mod._capture_enabled = True

        def enable_attn_capture_eva_multi(vit_model, last_k: int = 4):
            """
            给最后 K 个 block 的 attention 打补丁，缓存字段：block.attn.last_attn (B,H,N,N)
            - 兼容 timm EVA/EVA02 Attention
            - 若已经打过补丁则跳过
            """
            assert last_k >= 1
            blocks = vit_model.blocks
            k = min(last_k, len(blocks))
            target_blocks = blocks[-k:]

            for blk in target_blocks:
                attn_mod = blk.attn
                if getattr(attn_mod, "_capture_enabled", False):
                    continue

                def forward_with_capture(self_attn, x, rope=None, attn_mask=None, **kwargs):
                    B, N, C = x.shape

                    qkv = self_attn.qkv(x)  # (B, N, 3*C)
                    qkv = qkv.reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    if rope is not None:
                        if callable(rope):
                            q, k = rope(q, k)
                        elif hasattr(rope, "apply_rotary_emb"):
                            q, k = rope.apply_rotary_emb(q, k)

                    attn = (q @ k.transpose(-2, -1)) * getattr(self_attn, "scale", 1.0)  # (B,H,N,N)

                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            attn = attn.masked_fill(attn_mask, float("-inf"))
                        else:
                            attn = attn + attn_mask

                    attn = attn.softmax(dim=-1)
                    self_attn.last_attn = attn  # 缓存

                    attn = self_attn.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self_attn.proj(x)
                    x = self_attn.proj_drop(x)
                    return x

                attn_mod.forward = types.MethodType(forward_with_capture, attn_mod)
                attn_mod._capture_enabled = True

        if self.attn == "last":
            enable_last_attn_capture_eva(self.model)
        elif self.attn == "rollout":
            enable_attn_capture_eva_multi(self.model)
        else:
            raise ValueError(f"不支持的注意力计算类型: {self.attn}")
        
    @torch.no_grad()
    def _frames_to_inputs(self, frames: list[np.ndarray]):
        pixel_values = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
            pixel_values.append(self.val_transforms(img))
        pixel_values = torch.stack(pixel_values, dim=0).to(self.device)
        return {"pixel_values": pixel_values}

    def _last_cls_attn_maps(self, batch_inputs):
        """
        获取注意力图
        DINOv3 token序列结构: [CLS token] + [register token] + [patch tokens]
        
        Args:
            batch_inputs: 批次输入
            
        Returns:
            cls_attention: CLS token到patch tokens的注意力权重
        """
        # 前向传播，获取注意力
        with torch.amp.autocast('cuda', enabled=True):
            _ = self.model.forward_features(batch_inputs['pixel_values'])
                
            # 使用最后一层的注意力
            last_attention = self.model.blocks[-1].attn.last_attn  # (B, H, N, N)
            
            # 平均所有注意力头
            avg_attention = torch.mean(last_attention, dim=1)  # (B, N, N)
            
            # 提取CLS token到patch tokens的注意力
            # CLS token是第一个token
            cls_attention = avg_attention[:, 0, 5:]  # (B, num_patches)
            return cls_attention
        
    def _aggregate_heads(self, attn: torch.Tensor, mode: str = "entropy", entropy_temp: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        """
        attn: (B, H, N, N) softmax 后注意力
        返回: (B, N, N) 聚合后的注意力

        mode:
            - "mean": 简单均值
            - "entropy": 低熵 head 权重更高（对每个 head 计算平均熵 -> softmax(-H/T) 做权重）
        """

        assert mode in ("mean", "entropy")
        assert attn.dim() == 4, f"expect (B,H,N,N), got {attn.shape}"
        if mode == "mean":
            return attn.mean(dim=1)

        if mode == "entropy":
            # 熵：对 key 维度做 -sum(p log p)，再对 query token 平均，得到每个 head 的标量熵
            p = attn.clamp_min(eps)
            ent = -(p * p.log()).sum(dim=-1)        # (B,H,N)  每个 query 的熵
            ent = ent.mean(dim=-1)                  # (B,H)    对 query 平均

            # 权重：低熵更大，softmax(-ent / T)
            T = max(float(entropy_temp), eps)
            w = torch.softmax((-ent / T), dim=-1)   # (B,H)

            # 加权求和
            w = w[:, :, None, None]                 # (B,H,1,1)
            return (attn * w).sum(dim=1)

        raise ValueError(f"Unsupported head aggregation mode: {mode}")

    @torch.no_grad()
    def _lastk_rollout_cls_attn_maps(self, batch_inputs, last_k: int = 4, head_agg: str = "entropy", entropy_temp: float = 1.0, add_residual: bool = True, eps: float = 1e-8):
        """
        后 K 层 Attention Rollout，只输出 CLS -> patch 的 rollout 向量

        返回：
            cls_rollout: (B, num_patches)  即 rollout 矩阵 R 的 CLS 行取 5: 之后
        """
        assert head_agg in ("mean", "entropy")
        L = len(self.model.blocks)
        k = min(int(last_k), L)
        start = L - k

        with torch.amp.autocast("cuda", enabled=True):
            _ = self.model.forward_features(batch_inputs["pixel_values"])

            # 取 token 数 N（从最后一层缓存拿即可；若 last_k=1 也可用）
            last_attn = self.model.blocks[-1].attn.last_attn  # (B,H,N,N)
            B, _, N, _ = last_attn.shape

            # 初始化 rollout 矩阵 R = I
            R = torch.eye(N, device=last_attn.device, dtype=last_attn.dtype).unsqueeze(0).expand(B, N, N)

            for li in range(start, L):
                attn_l = self.model.blocks[li].attn.last_attn  # (B,H,N,N)

                A = self._aggregate_heads(attn_l, mode=head_agg, entropy_temp=entropy_temp, eps=eps)  # (B,N,N)

                if add_residual:
                    A = A + torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)

                # 行归一化
                A = A / (A.sum(dim=-1, keepdim=True).clamp_min(eps))

                # 递推：R = A @ R（从浅到深）
                R = A @ R

            # 按你的硬编码：CLS=0，patch 从 5: 开始
            cls_rollout = R[:, 0, 5:]  # (B, num_patches)
            return cls_rollout

    def _vector_to_saliency(self, v: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
        v_cpu = v.detach().float().cpu().numpy()
        num_p = v_cpu.size

        g = int(round(self.math.sqrt(num_p)))
        grid_h = g
        grid_w = int(self.math.ceil(num_p / grid_h))
        if grid_h * grid_w < num_p:
            grid_w = int(self.math.ceil(num_p / grid_h))

        pad = grid_h * grid_w - num_p
        if pad > 0:
            vec = np.concatenate([v_cpu, np.zeros(pad, dtype=v_cpu.dtype)], axis=0)
        else:
            vec = v_cpu

        grid = vec.reshape(grid_h, grid_w)
        sal = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        mn = float(sal.min())
        mx = float(sal.max())
        if mx - mn < 1e-8:
            sal_norm = np.zeros_like(sal, dtype=np.float32)
        else:
            sal_norm = ((sal - mn) / (mx - mn)).astype(np.float32)
        return sal_norm

    def extract_dino_video_pixel_rollout(self, frames: list[np.ndarray], indices=None, last_k: int = 4, head_agg: str = "entropy", entropy_temp: float = 1.0, add_residual: bool = True):
        """
        输出每帧的 rollout 向量（patch 级 1D），不做网格/像素映射。
        对齐你原有 extract_dino_video_patch() 的用法。

        Returns:
            raw_sals: List[torch.Tensor]，每个元素 shape=(num_patches,)
        """
        assert isinstance(frames, list)
        raw_sals: list[np.ndarray] = []

        # 如果提供了indices，则只处理采样帧
        if indices is not None:
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            sampled_frames = frames

        # 每帧显著性生成
        for i in range(0, len(sampled_frames), self.bs):
            chunk = sampled_frames[i:i + self.bs]
            sizes = [(f.shape[0], f.shape[1]) for f in chunk]
            inp = self._frames_to_inputs(chunk)
            cls_maps = self._lastk_rollout_cls_attn_maps(inp, last_k=last_k, head_agg=head_agg, entropy_temp=entropy_temp, add_residual=add_residual)  # (B, num_patches)
            B = cls_maps.shape[0]
            for b in range(B):
                oh, ow = sizes[b]
                sal = self._vector_to_saliency(cls_maps[b], oh, ow)
                raw_sals.append(sal)
        return raw_sals

    def extract_dino_video_pixel_last(self, frames: list[np.ndarray], indices=None):
        assert isinstance(frames, list)
        raw_sals: list[np.ndarray] = []

        # 如果提供了indices，则只处理采样帧
        if indices is not None:
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            sampled_frames = frames

        # 每帧显著性生成
        for i in range(0, len(sampled_frames), self.bs):
            chunk = sampled_frames[i:i + self.bs]
            sizes = [(f.shape[0], f.shape[1]) for f in chunk]
            inp = self._frames_to_inputs(chunk)
            cls_maps = self._last_cls_attn_maps(inp)  # (B, num_p)
            B = cls_maps.shape[0]
            for b in range(B):
                oh, ow = sizes[b]
                sal = self._vector_to_saliency(cls_maps[b], oh, ow)
                raw_sals.append(sal)
        return raw_sals

    def extract_dino_video_patch_last(self, frames: list[np.ndarray], indices=None):
        """
        提取视频的DINOv3显著性图
        
        Args:
            frames: 视频帧列表
            indices: 采样帧的索引列表，如果为None则处理所有帧
            
        Returns:
            raw_sals: 显著性图列表
        """
        assert isinstance(frames, list)
        raw_sals: list[np.ndarray] = []

        # 如果提供了indices，则只处理采样帧
        if indices is not None:
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            sampled_frames = frames

        # 逐帧处理生成显著图
        for i in range(0, len(sampled_frames), self.bs):
            chunk = sampled_frames[i:i + self.bs]
            inp = self._frames_to_inputs(chunk)
            cls_maps = self._last_cls_attn_maps(inp)  # (B, num_patches)
            # 直接将整个batch的注意力向量添加到结果中，提高效率
            # 使用切片操作一次性处理整个batch
            raw_sals.extend([cls_maps[b] for b in range(cls_maps.shape[0])])
        return raw_sals

    def extract_dino_video_patch_rollout(self, frames: list[np.ndarray], indices=None, last_k: int = 4, head_agg: str = "entropy", entropy_temp: float = 1.0, add_residual: bool = True):
        """
        提取视频的DINOv3显著性图
        
        Args:
            frames: 视频帧列表
            indices: 采样帧的索引列表，如果为None则处理所有帧
            
        Returns:
            raw_sals: 显著性图列表
        """
        assert isinstance(frames, list)
        raw_sals: list[np.ndarray] = []

        # 如果提供了indices，则只处理采样帧
        if indices is not None:
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            sampled_frames = frames

        # 逐帧处理生成显著图
        for i in range(0, len(sampled_frames), self.bs):
            chunk = sampled_frames[i:i + self.bs]
            inp = self._frames_to_inputs(chunk)
            cls_maps = self._lastk_rollout_cls_attn_maps(inp, last_k=last_k, head_agg=head_agg, entropy_temp=entropy_temp, add_residual=add_residual)  # (B, num_patches)
            # 直接将整个batch的注意力向量添加到结果中，提高效率
            # 使用切片操作一次性处理整个batch
            raw_sals.extend([cls_maps[b] for b in range(cls_maps.shape[0])])
        return raw_sals
    
    def video_with_overlay_pixel(self, frames: list[np.ndarray], raw_sals, indices=None):
        # 如果提供了indices，则只处理采样帧
        if indices is not None:
            sampled_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        else:
            sampled_frames = frames

        # 转伪彩并叠加原图
        heatmaps: list[np.ndarray] = []
        overlays: list[np.ndarray] = []
        for frame, sal in zip(sampled_frames, raw_sals):
            s_u8 = (np.clip(sal, 0.0, 1.0) * 255.0).astype(np.uint8)
            bgr = cv2.applyColorMap(s_u8, cv2.COLORMAP_JET)
            rgb = bgr[..., ::-1].copy()
            heatmaps.append(rgb)

            # 确保frame是RGB格式
            if frame.shape[-1] == 3:
                frame_rgb = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame_rgb = frame
                
            overlay = cv2.addWeighted(frame_rgb, 1 - self.alpha, rgb, self.alpha, 0)
            overlays.append(overlay)

        return heatmaps, overlays

    def visualize_attention_maps_patch(self, raw_sals):
        """
        将注意力图可视化为热力图，仅用于观察注意力变化
        
        Args:
            raw_sals: 注意力图列表
            
        Returns:
            heatmaps: 可视化的注意力热力图列表
        """
        heatmaps: list[np.ndarray] = []
        
        for sal in raw_sals:
            sal_np = sal.detach().cpu().numpy()
            
            # 对每个注意力图进行帧内归一化到[0, 1]范围
            sal_min = sal_np.min()
            sal_max = sal_np.max()
            if sal_max > sal_min:
                sal_norm = (sal_np - sal_min) / (sal_max - sal_min)
            else:
                sal_norm = np.zeros_like(sal_np)        
            
            # 将归一化的注意力图转换为可视化图像
            sal_u8 = (sal_norm * 255.0).astype(np.uint8)
            
            # 将1D注意力图重塑为方形图像以便可视化
            # 计算接近方形的维度
            n_patches = sal_u8.shape[0]
            h = int(np.sqrt(n_patches))
            w = int(np.ceil(n_patches / h))
            
            # 调整数组大小以适应矩形形状
            if h * w > n_patches:
                padded = np.pad(sal_u8, (0, h * w - n_patches), mode='constant')
                sal_img = padded.reshape(h, w)
            else:
                sal_img = sal_u8.reshape(h, w)
            
            # 调整图像大小以便更好地可视化
            target_size = 224  # 使用固定大小便于观察
            resized_sal = cv2.resize(sal_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            # 应用伪彩色映射
            bgr = cv2.applyColorMap(resized_sal, cv2.COLORMAP_JET)
            rgb = bgr[..., ::-1].copy()  # BGR转RGB
            heatmaps.append(rgb)
        return heatmaps