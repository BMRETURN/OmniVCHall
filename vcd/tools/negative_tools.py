import cv2
import numpy as np


class ReverseVideo:
    """视频反转处理器"""
    def __init__(self):
        self.name = 'ReverseVideo'
        self.type = 'temporal'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        反转帧顺序，返回反转后的帧列表（RGB 格式，NumPy 数组）
        
        Args:
            frames: 输入帧列表，每个元素是(H, W, 3)形状的NumPy数组
            
        Returns:
            反转后的帧列表
        """
        reversed_frames = frames[::-1]  # 反转帧顺序
        return reversed_frames  # list of (H, W, 3) RGB arrays

class SampleVideo:
    """按比例抽帧处理器"""
    def __init__(self):
        self.name = 'SampleVideo'
        self.type = 'temporal'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        按照指定比例对视频帧进行均匀抽样，同时强制保留首尾帧
        
        Args:
            frames: 输入的视频帧列表，每个元素是一个numpy数组表示的图像帧
            ratio: 抽样比例，取值范围(0, 1]
            
        Returns:
            抽样后的视频帧列表，包含首尾帧和中间按比例抽取的帧
        """
        ratio = 0.4
        N = len(frames)
        # 修改条件判断，当不满足条件时直接返回原帧序列
        if N < 7 or not (0 < ratio <= 1.0):
            print(f"Warning: 不满足抽帧条件 (帧数: {N}, 抽样比例: {ratio})，返回原始帧序列")
            return frames.copy()

        # 目标采样点数（至少 2 个，包含首尾）
        M = max(2, int(np.round(N * ratio)))

        # 线性均匀采样 + 强制首尾（linspace 本身就包含端点）
        idx = np.linspace(0, N - 1, num=M)
        idx = np.rint(idx).astype(int)

        # 去掉可能的重复（极端 N*ratio 很小时），再把首尾补上保证存在
        idx = np.unique(np.clip(idx, 0, N - 1))
        if idx[0] != 0:
            idx = np.insert(idx, 0, 0)
        if idx[-1] != N - 1:
            idx = np.append(idx, N - 1)

        return [frames[i] for i in idx.tolist()]

class ShuffleVideo:
    """帧乱序处理器"""
    def __init__(self):
        self.name = 'ShuffleVideo'
        self.type = 'temporal'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        全局乱序：可选保留首尾帧不参与乱序
        
        Args:
            frames: 输入的视频帧列表
            
        Returns:
            打乱顺序后的帧列表
        """           
        seed=2025
        rng = np.random.default_rng(seed)
        N = len(frames)

        # 保留首尾，打乱中间
        middle = np.arange(1, N - 1)
        rng.shuffle(middle)
        idx = np.concatenate(([0], middle, [N - 1]))
        return [frames[i] for i in idx.tolist()]

class BlurVideo:
    """高斯模糊处理器"""
    def __init__(self):
        self.name = 'BlurVideo'
        self.type = 'frame'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        对视频帧应用高斯模糊效果
        
        Args:
            frames: 输入的视频帧列表，每个元素是(H, W, 3)形状的NumPy数组
            kernel_size: 高斯核大小，必须是正奇数，越大模糊效果越强，默认为15
            sigma_x: X方向的标准差，0表示自动计算，默认为0.0
            
        Returns:
            应用模糊效果后的帧列表
            
        Raises:
            AssertionError: 当kernel_size不是正奇数时抛出异常
        """
        kernel_size = 15
        sigma_x = 0.0
        assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size必须是正奇数"
        
        if not frames:
            return frames
            
        blurred_frames = []
        
        for frame in frames:
            # 应用高斯模糊
            blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma_x)
            blurred_frames.append(blurred_frame)
        return blurred_frames

class NoiseVideo:
    """添加高斯噪声处理器"""
    def __init__(self):
        self.name = 'NoiseVideo'
        self.type = 'frame'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        给视频帧添加高斯噪声
        
        Args:
            frames: 输入的视频帧列表，每个元素是(H, W, 3)形状的NumPy数组
            mean: 高斯噪声的均值，默认为0
            std: 高斯噪声的标准差，越大噪声越明显，默认为25
            seed: 随机种子，用于结果复现，默认为None
            
        Returns:
            添加高斯噪声后的帧列表
            
        Raises:
            AssertionError: 当std为负数时抛出异常
        """
        mean = 0.0
        std = 25.0
        seed = 2025
        if not frames:
            return frames
            
        # 设置随机种子以确保结果可复现
        if seed is not None:
            np.random.seed(seed)
            
        noisy_frames = []
        
        for frame in frames:
            # 生成与帧大小相同的高斯噪声
            noise = np.random.normal(mean, std, frame.shape)
            
            # 将噪声添加到原始帧
            noisy_frame = frame.astype(np.float32) + noise.astype(np.float32)
            
            # 确保像素值在有效范围内[0, 255]
            noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
            
            noisy_frames.append(noisy_frame)
        return noisy_frames

class HorizontalMirrorVideo:
    """水平镜像翻转处理器"""
    def __init__(self):
        self.name = 'HorizontalMirrorVideo'
        self.type = 'frame'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        对视频帧进行镜像翻转处理
        
        Args:
            frames: 输入的视频帧列表，每个元素是(H, W, 3)形状的NumPy数组
            
        Returns:
            包含镜像翻转后的帧列表和翻转类型描述的元组
        """
        if not frames:
            return frames
        
        mirrored_frames = []
        
        for frame in frames:
            mirrored_frame = cv2.flip(frame, 1)
            mirrored_frames.append(mirrored_frame)
                
        return mirrored_frames

class VerticalMirrorVideo:
    """垂直镜像翻转处理器"""
    def __init__(self):
        self.name = 'VerticalMirrorVideo'
        self.type = 'frame'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        对视频帧进行镜像翻转处理
        
        Args:
            frames: 输入的视频帧列表，每个元素是(H, W, 3)形状的NumPy数组
            
        Returns:
            包含镜像翻转后的帧列表和翻转类型描述的元组
        """
        if not frames:
            return frames
        
        mirrored_frames = []
        
        for frame in frames:
            mirrored_frame = cv2.flip(frame, 0)
            mirrored_frames.append(mirrored_frame)
                
        return mirrored_frames

class GrayscaleVideo:
    """灰度化处理器"""
    def __init__(self):
        self.name = 'GrayscaleVideo'
        self.type = 'frame'
    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        将视频帧转换为灰度图像，但保持三通道格式
        
        Args:
            frames: 输入的视频帧列表，每个元素是(H, W, 3)形状的NumPy数组
            
        Returns:
            灰度化后的帧列表，每个元素是(H, W, 3)形状的NumPy数组
        """
        if not frames:
            return frames
            
        grayscale_frames = []
        
        for frame in frames:
            # 将彩色图像转换为灰度图像
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # 将灰度图像转换回三通道格式
            gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            grayscale_frames.append(gray_frame_3ch)
        
        return grayscale_frames