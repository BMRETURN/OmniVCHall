import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
import re
from vcd_new.models.selector import QFormerToolRouter
from vcd_new.models.gate import QueryVisualFusionGater


class VCDPolicy(nn.Module):
    def __init__(self, router: QFormerToolRouter, gater: QueryVisualFusionGater, tool_names):
        super().__init__()
        self.router = router
        self.gater = gater
        self.tool_names = tool_names
        
    def forward(self, video_embeddings, tool_embeddings_dict):
        """
        前向传播，返回分布参数
        """
        # 1. Router Logits
        # 注意：这里我们使用 logits 直接构建 Bernoulli 分布
        # 虽然原代码用了 Softmax，但在 RL 中使用独立 Bernoulli 允许选择多个工具，
        # 且允许梯度独立回传，这比 Categorical 更适合多工具组合场景。
        router_logits = self.router(video_embeddings, tool_embeddings_dict)
        
        # 2. Gater Beta (Mean)
        gater_beta = self.gater(video_embeddings)
        
        return router_logits, gater_beta

    def get_action_and_log_prob(self, video_embeddings, tool_embeddings_dict, std_dev=0.1):
        """
        蒙特卡洛采样动作
        """
        router_logits, gater_mu = self(video_embeddings, tool_embeddings_dict)
        
        # --- A. Router 采样 (Bernoulli) ---
        # 使用 sigmoid 将 logits 映射到 [0, 1] 概率
        router_probs = torch.sigmoid(router_logits)
        # 创建分布
        router_dist = Bernoulli(probs=router_probs)
        
        if self.training:
            router_action = router_dist.sample() # [8] (0 or 1)
        else:
            # 推理时使用阈值截断 (保持与原逻辑一致) 或者 argmax
            router_action = (router_probs > self.router.threshold).float()
            
        # 计算 log probability (所有工具选择的联合概率)
        router_log_prob = router_dist.log_prob(router_action).sum()
        
        # 解析选中的工具名称
        selected_indices = torch.where(router_action == 1)[0]
        selected_tools = [self.tool_names[i] for i in selected_indices.cpu().numpy()]
        
        # --- B. Gater 采样 (Normal) ---
        # gater_mu 范围是 [0, 1] (因为最后是 Sigmoid)
        if self.training:
            # 训练时添加高斯噪声进行探索
            gater_dist = Normal(loc=gater_mu, scale=std_dev)
            gater_action = gater_dist.sample()
            # 必须截断到 [0, 1] 才有物理意义，但截断会影响梯度，
            # 这里的 log_prob 计算使用未截断的值是近似做法，或者使用 Beta 分布会更严谨。
            # 为简化，我们使用 Normal 并 clamp 动作值，但 log_prob 基于原始采样计算。
            gater_log_prob = gater_dist.log_prob(gater_action)
            beta_final = torch.clamp(gater_action, 0.0, 1.0)
        else:
            beta_final = gater_mu
            gater_log_prob = torch.tensor(0.0, device=video_embeddings.device)
            
        return selected_tools, beta_final, router_log_prob, gater_log_prob


class VCDTrainer:
    def __init__(self, policy, optimizer, accumulation_steps=32):
        self.policy = policy
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps # 新增：梯度累积步数
        self.baseline = 0.0
        self.alpha = 0.95
        self.current_step = 0 # 计数器
        self.optimizer.zero_grad() # 初始化梯度

    def compute_reward(self, prediction, ground_truth):
        pred_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(prediction).lower())
        pred_token = pred_clean.group(1) if pred_clean else str(prediction).lower().strip()
        gt_clean = re.search(r'\b(a|b|c|d|yes|no)\b', str(ground_truth).lower())
        gt_token = gt_clean.group(1) if gt_clean else str(ground_truth).lower().strip()
        return 1.0 if pred_token == gt_token else -1.0

    def step(self, reward, router_log_prob, gater_log_prob):
        # 1. 更新 Baseline
        self.baseline = self.alpha * self.baseline + (1 - self.alpha) * reward
        advantage = reward - self.baseline
        
        # 2. 计算 Loss 并进行缩放
        # Loss = - log_prob * advantage / accumulation_steps
        loss = - (router_log_prob + gater_log_prob) * advantage
        loss = loss / self.accumulation_steps
        
        # 3. 反向传播 (积累梯度)
        loss.backward()
        
        self.current_step += 1
        
        # 4. 只有达到累积步数时才更新参数
        if self.current_step % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item() * self.accumulation_steps # 返回还原后的 Loss 方便打印
            
        return loss.item() * self.accumulation_steps
