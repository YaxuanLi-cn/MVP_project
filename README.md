# Learning Structural Primitive Decomposition from 3D Gaussian Splatting

这是一个可运行的 MVP，用于把已经训练好的 3D Gaussian Splatting 表示直接分解成紧凑的解析几何 primitive。它不需要原始多视图图像，也不重新训练 Gaussian reconstruction。

实现包含：

- 3DGS `.ply` 与实验用 `.npz` 读取，支持位置、log-scale、四元数方向、opacity 和可选 appearance features；
- point-cloud baseline：只使用 Gaussian centers 拟合 superquadrics；
- Gaussian-aware 方法：在软分配和 primitive 更新中使用每个 Gaussian 的 covariance、orientation 与 opacity；
- `superquadric` 与 `superfrustum` 两种输出；
- 源几何点云的 Chamfer、coverage、压缩率评估；
- 属性分析和四组消融：positions-only、opacity-only、covariance+orientation、all attributes。

## 方法

对第 `i` 个 Gaussian，位置为 `x_i`，协方差为 `Σ_i`，可见性权重为 `w_i`。primitive `j` 的中心和形状由软责任 `r_ij` 更新：

```text
r_ij = softmax_j(-d(x_i, P_j; Σ_i) / T)
μ_j   = Σ_i r_ij w_i x_i / Σ_i r_ij w_i
C_j   = Cov_r(x_i) + λΣ_i r_ij w_i Σ_i / Σ_i r_ij w_i
```

其中 `d` 使用 primitive 协方差和 Gaussian covariance 的 trace-aware Mahalanobis 距离。最终优化记录以下项：

```text
L = L_data + λcompact L_compact + λoverlap L_overlap
```

baseline 令 `Σ_i = 0`，因此退化为从 Gaussian center point cloud 做 primitive decomposition。这个设计提供了清晰的 ablation 对照：如果 aware 方法优于 baseline，说明 Gaussian 的二阶几何属性在中心点之外携带了有效结构信息。

## 快速运行

```bash
cd /root/mvp_project
source /venv/main/bin/activate

# 生成一个含 3 个结构部件的 synthetic 3DGS cloud
python -m gaussian_decomp.cli generate-synthetic \
  --output artifacts/synthetic.npz --n-per-part 180

# 查看 Gaussian 属性统计
python -m gaussian_decomp.cli analyze artifacts/synthetic.npz

# baseline
python -m gaussian_decomp.cli fit artifacts/synthetic.npz \
  --source artifacts/synthetic.source.npy --method baseline \
  --primitive-type superquadric -k 3 \
  --output artifacts/baseline.json --report artifacts/baseline.report.json

# Gaussian-aware superquadric
python -m gaussian_decomp.cli fit artifacts/synthetic.npz \
  --source artifacts/synthetic.source.npy --method gaussian-aware \
  --primitive-type superquadric -k 3 \
  --output artifacts/aware.json --report artifacts/aware.report.json

# Gaussian-aware superfrustum
python -m gaussian_decomp.cli fit artifacts/synthetic.npz \
  --source artifacts/synthetic.source.npy --method gaussian-aware \
  --primitive-type superfrustum -k 3 --output artifacts/frusta.json

# 四组属性消融
python scripts/run_ablation.py artifacts/synthetic.npz \
  --source artifacts/synthetic.source.npy -k 3
```

## ShapeSplat / 3DGS PLY

标准 3DGS PLY 通常包含：`x y z`、`scale_0..2`、`rot_0..3`、`opacity`。直接运行：

```bash
python -m gaussian_decomp.cli analyze /path/to/object/point_cloud.ply
python -m gaussian_decomp.cli fit /path/to/object/point_cloud.ply \
  --method gaussian-aware --primitive-type superfrustum -k 16 \
  --output artifacts/object.json --report artifacts/object.report.json \
  --source /path/to/source_geometry.npy
```

ShapeSplat 的 source geometry 可以先采样为 `N x 3` 的 `.npy` 或 `.npz`（key 为 `points`、`vertices` 或 `positions`）。如果只想评估 Gaussian 内部结构，可省略 `--source`，此时仍会输出 fitting loss 和 assignment entropy。

## 真实 ShapeSplat 数据

真实 ShapeSplat 文件在 Hugging Face gated repository 中。先在 Hugging Face 接受 `ShapeSplats/modelnet_3dgs_preprocessed` 的访问条款，并把 token 放入环境变量：

```bash
export HF_TOKEN=hf_...
python scripts/download_shapesplat.py \
  --category bowl --split test --object bowl_0065 \
  --output data/shapesplat
```

然后转换真实对象目录并拟合：

```bash
python -m gaussian_decomp.cli convert-shapesplat \
  data/shapesplat/bowl/test/bowl_0065 \
  --output artifacts/bowl_0065.npz

python -m gaussian_decomp.cli fit artifacts/bowl_0065.npz \
  --method gaussian-aware --primitive-type superfrustum -k 16 \
  --output artifacts/bowl_0065.primitives.json \
  --report artifacts/bowl_0065.report.json
```

批量跑真实对象并生成对比表：

```bash
python scripts/run_shapesplat_subset.py \
  --objects bowl_0065,bowl_0066,bowl_0067,bowl_0068 \
  --output artifacts/shapesplat_subset.json
```

脚本会输出 baseline、Gaussian-aware superquadric、Gaussian-aware
superfrustum，并对匹配的 ModelNet40 `.off` source mesh 计算 Chamfer 和
coverage；同时保留基于高 opacity Gaussians 的 `opacity_surface_proxy` 对照。

下载脚本只请求单个对象的 `.npy` 文件，不会下载整个类别的多 GB 压缩包；访问 gated 数据集时需要配置 `HF_TOKEN`。

本次真实运行已经下载并处理 `bowl/test/bowl_0065` 至 `bowl_0068`，每个对象含 3 维 `color.npy` appearance features。完整汇总见 [real_results.md](docs/real_results.md) 和 `artifacts/shapesplat_subset.json`。这些是 3DGS 内部拟合及 opacity surface proxy 指标；预处理对象目录没有独立 mesh 文件，因此不能把 proxy 当作 source-mesh ground truth。

## 代码结构

- `gaussian_decomp/data.py`：3DGS 数据结构、covariance 计算、PLY/NPZ I/O；
- `gaussian_decomp/primitives.py`：Superquadric / SuperFrustum、采样和 JSON 序列化；
- `gaussian_decomp/fitting.py`：weighted k-means 初始化、软分配、Gaussian-aware moment refinement、loss；
- `gaussian_decomp/evaluate.py`：Chamfer、coverage、compression；
- `gaussian_decomp/cli.py`：命令行入口；
- `scripts/run_ablation.py`：属性消融实验；
- `tests/`：数据 round-trip、两种 fitter 和 frustum 序列化测试。

## 研究扩展

当前实现的 primitive shape exponents `e1/e2` 保持稳定的可解释默认值，重点验证 Gaussian covariance/orientation 是否改善结构分解。下一步可以在此接口上加入：可学习的 `e1/e2`、局部法向估计、appearance feature 的边界一致性项、可微渲染投影损失，以及 ShapeSplat 全量 benchmark 的类别级统计。

结构一致性评估和 SceneSplat-7K 场景扩展见 [scene_extension.md](docs/scene_extension.md)。

可直接放入项目报告的实验设置、ShapeSplat 数值结果和 SceneSplat-7K
场景结果见 [report_results.md](docs/report_results.md)。

扩展到 67 个真实对象的跨类别 benchmark 结果见
[large_benchmark_results.md](docs/large_benchmark_results.md)。

## 一键下载实验数据

仓库不包含数据文件。接受对应 Hugging Face gated terms 后运行：

```bash
export HF_TOKEN=hf_...
python scripts/download_data.py
```

该命令下载报告使用的四个 ShapeSplat bowl 对象、匹配的 ModelNet40 OFF
source meshes，以及 `scene0000_00` 的 SceneSplat ScanNet Gaussian PLY 到
`data/`。可用 `--skip-scene` 只下载对象数据。

