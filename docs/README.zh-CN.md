# STRATA 使用说明

STRATA 是一个 PyTorch 优化器。它用不同长度的梯度历史，估计哪些方向持续
出现、哪些方向变化较快，再用一组相互包含的子空间组织更新。这种嵌套结构
在数学上称为“旗”。模型权重本身仍然可以自由更新。

仓库包含优化器、对比方法、数据准备和训练脚本、实验配置、数值检查、绘图
工具、已保存的结果表，以及论文源码。

## 安装和快速检查

在仓库根目录执行：

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

这两个检查使用 CPU，不下载数据。第一个训练一个小型回归模型，第二个检查
正交性、子空间嵌套、梯度分解等 13 项数值性质。

## 运行正式实验

先在仓库根目录安装额外依赖：

```bash
python -m pip install -e ".[experiments]"
```

再按照[实验指南](EXPERIMENTS.md)准备数据，并从 `experiments/` 目录运行训练
命令。图像、图学习、表格和语言模型的扩展训练脚本需要 CUDA；预测、评分推荐
和传感器任务的 `train_new_families.py` 默认使用 CPU。批量调度工具建议在
Linux 下运行。

每次训练使用独立的输出目录。想先检查命令能否运行，可以使用脚本支持的
`--smoke` 短运行选项，并把输出放到另一个目录。

## 如何阅读结果

- 一个“实验配置”是一组固定的任务、方法和超参数。856 个配置中也包含未运行的项目。
- 本次公开结果使用 seed 42；它们不能代替多随机种子的统计检验。
- 一组方法比较需要完成预先指定的全部配置，再按最后 100 次评估的验证集均值选参。
- 选好配置后读取对应的测试集成绩，不用测试集决定学习率。
- 完整方法比较、未胜出的结果和失败说明保存在论文源码及 `results/` 说明中。
- 原始数据、训练检查点、运行中的任务队列和连接凭据不随代码发布。

参数用法见 [API.md](API.md)，计算步骤和公式见 [algorithm.md](algorithm.md)。
论文源码位于 [paper/main.tex](../paper/main.tex)，PDF 可在本地编译生成。
代码使用 [MIT](../LICENSE) 许可证；第三方代码和数据集遵守各自的使用条款。
