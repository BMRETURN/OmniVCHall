# vcd_ste

`vcd_ste` 是基于 `vcd_new` 优化链路的 TriCD 消融版本：

- backbone: `Qwen3-VL-8B-Instruct`
- 去掉 RL，改为 STE 监督训练
- Router: 工具 logits -> sigmoid 概率 -> 阈值累计选工具（非 Top1/Top2）
- Gater: 样本级 beta 学习融合 `motion` 与 `visual` 显著性
- 复用 `vcd_new/train/saliency_cache` 加速

## 关键脚本

- `train_ste.py`: 训练 STE Router + Gater（冻结 backbone）
- `eval_ste.py`: 用训练好的 checkpoint 在 test 评测
- `run_smoke.sh`: 冒烟（训练 + 小规模测试）
- `run_full_test.sh`: 全量（训练 + 完整 test）

## 默认资源

- 环境：`videoproject`
- 显卡：`CUDA_VISIBLE_DEVICES=0,2,7`

