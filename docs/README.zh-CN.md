# STRATA 使用说明

STRATA 把优化器状态放在嵌套旗流形上，用不同时间尺度的协方差草图跟踪梯度方向。
模型权重本身不受正交或流形约束。本仓库包含全部实验实现、基线、数据准备、消融、
数值验证、实验配置、绘图工具，以及脱敏后的论文快照。

## 安装与快速检查

```bash
git clone https://github.com/Hik289/manifold-strata-optimization.git
cd manifold-strata-optimization
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python examples/quickstart.py --steps 40
python experiments/test_numerical.py
```

这两个检查在 CPU 上运行，不下载数据、不启动正式实验。
运行完整任务前安装 `python -m pip install -e ".[experiments]"`，再按
[实验指南](EXPERIMENTS.md)准备数据和执行命令。多数视觉、图、语言训练需要 CUDA；
时序预测、推荐和传感器任务默认支持 CPU。队列工具建议在 Linux 使用。

## 结果与复现规则

- 配置目录保存全部 856 个候选方案；这不表示 856 个方案均已完成。
- 论文快照中的全部实验使用 seed 42，不声称多种子统计显著。
- 等全部指定方法和候选完成后，按验证集末段均值选参，再读取测试结果。
- 正式结果与 `--smoke` 检查必须使用不同输出目录。
- 完整基线、不利结果与失败记录保留在论文和证据说明中。
- 原始数据、训练检查点、运行队列与任何连接凭据均不随代码发布。

方法接口见 [API.md](API.md)，完整数学说明见 [algorithm.md](algorithm.md)，
论文源码见 [paper/main.tex](../paper/main.tex)，PDF 可在本地编译生成。
代码采用 [MIT](../LICENSE) 许可证，
第三方基线和数据集遵守各自的许可证与使用条款。
