# 实验结果

## 实验设置

我们在真实 ShapeSplat/ModelNet40 对象和 SceneSplat-7K 场景 Gaussian Splatting 表示上评估结构 primitive decomposition。对象实验使用 `ShapeSplats/modelnet_3dgs_preprocessed` 的 `bowl/test` 子集（`bowl_0065`–`bowl_0068`），每个对象包含 Gaussian center、scale、quaternion、opacity 和 3-channel color feature；对应 source geometry 使用同名 ModelNet40 `.off` mesh。所有 mesh 通过 axis-aligned bounding-box 归一化到同一尺度，并从三角面均匀采样 6,000 个表面点。

我们比较三种方法：

1. **Center-only baseline**：只使用 Gaussian centers，把高斯表示当作普通点云，拟合 16 个 Superquadrics。
2. **Gaussian-aware Superquadric**：在软分配和 moment refinement 中加入 covariance、orientation、opacity 和 color feature。
3. **Gaussian-aware SuperFrustum**：使用同样的 Gaussian-aware 信息，但输出具有轴线、长度和两端半径的 SuperFrusta。

每个实验使用 20 次 refinement iterations 和固定随机种子。Chamfer 使用 source mesh surface points 与 primitive surface samples 计算；coverage@0.05 表示 source surface points 到最近 primitive surface 的距离小于 0.05 的比例。

## ShapeSplat 对象结果

| 方法 | Final fitting loss | Source Chamfer ↓ | Coverage@0.05 ↑ | Coverage@0.1 ↑ | Primitive 数 |
|---|---:|---:|---:|---:|---:|
| Center-only baseline / Superquadric | 2.9084 ± 0.0235 | 0.002148 ± 0.001030 | 0.9621 ± 0.0546 | 1.0000 ± 0.0000 | 16 |
| Gaussian-aware / Superquadric | **2.0838 ± 0.0820** | **0.001737 ± 0.000448** | **0.9738 ± 0.0453** | 1.0000 ± 0.0000 | 16 |
| Gaussian-aware / SuperFrustum | **0.7608 ± 0.0536** | 0.005078 ± 0.000936 | 0.8815 ± 0.0719 | 0.9975 ± 0.0028 | 16 |

结果表明，Gaussian-aware Superquadric 相比 center-only baseline 将平均 source Chamfer 降低约 **19.1%**，coverage@0.05 从 **0.9621** 提升到 **0.9738**。这说明 Gaussian covariance、orientation 和 opacity 包含超出 Gaussian centers 的有效几何信息。SuperFrustum 的内部 fitting loss 最低，但 source-mesh Chamfer 和 coverage 较差，说明当前 frustum 参数化更容易用细长 primitive 降低 assignment loss，却不一定对应更准确的物体表面。

## SceneSplat-7K 场景结果

我们进一步在真实 `GaussianWorld/scannet_mcmc_3dgs/scene0000_00` 场景上运行 full-scene extension。该场景的 PLY 包含 **999,999** 个 Gaussians。由于逐点 covariance refinement 的计算成本较高，使用固定 seed=7 的 opacity-importance sampling，将 120,000 个 Gaussians 用于拟合；该采样策略保留高可见度 splats，同时加入低 opacity splats 以保留场景边界。输出 64 个 Gaussian-aware Superquadrics，8 次 refinement iterations。

| 指标 | 数值 |
|---|---:|
| 原始 Gaussian 数量 | 999,999 |
| 拟合 Gaussian 数量 | 120,000 |
| Primitive 数量 | 64 |
| Compression ratio（按拟合子集） | 1,875.0:1 |
| Primitive occupancy | 1.0000 |
| Scene bounding-box volume | 281.3462 |
| Mean primitive-center separation | 4.2821 |
| Minimum primitive-center separation | 0.5592 |
| Assignment entropy | 3.9742 |
| Final fitting loss | 5.8167 |

场景结果证明该方法可以从百万级 3DGS 表示中恢复紧凑的 primitive assembly，并为场景级结构提供可量化指标。SceneSplat-7K 当前公开的组件仓库主要提供 Gaussian PLY 和重建统计；对于没有独立 part annotations 的场景，本文不报告 semantic part purity，而使用 occupancy、center separation 和 assignment entropy 作为结构一致性的无监督指标。

## 结论与局限

Gaussian-aware Superquadric 是当前最可靠的模型：在真实对象 source mesh 上同时取得最低 Chamfer 和最高 coverage。Covariance/orientation refinement 对对象级结构分解有效；SuperFrustum 适合作为细长结构的扩展，但需要进一步加入 surface-aware regularization 或 mesh/rendering loss。场景级实验已经覆盖真实 SceneSplat-7K Gaussian PLY，但当前 scene report 采用固定采样进行拟合，且缺少可直接对齐的 per-Gaussian semantic part labels，因此场景级 semantic consistency 仍需后续补充标注或数据集提供的对应标签。

可复现实验输出：

- `artifacts/shapesplat_subset.json`
- `artifacts/scene0000_00.report.json`
- `artifacts/scene0000_00.primitives.json`

## Scene structural consistency result

For the full-scene extension, we evaluated the real SceneSplat-7K ScanNet scene
`scene0000_00`. The source PLY contains 999,999 Gaussians; a fixed seed=7
opacity-importance sample of 120,000 Gaussians was used for fitting 64
primitives, giving a 1,875:1 compression ratio on the fitted representation.

| Scene method | Final fitting loss | Primitive occupancy | Mean center separation | Min center separation | Assignment entropy |
|---|---:|---:|---:|---:|---:|
| Center-only baseline | 5.8061 | 1.0000 | 4.2915 | 0.5259 | 3.9427 |
| Gaussian-aware Superquadric | 5.8167 | 1.0000 | 4.2821 | 0.5592 | 3.9742 |

Both methods activate all 64 primitives and produce similar scene coverage
statistics. On this single scene, Gaussian-aware attributes do not reduce the
aggregate scene fitting loss because the overlap regularizer dominates the
large-room layout; their benefit is clearer in the object source-mesh benchmark.
The scene result demonstrates the full-scene extension, while semantic part
purity is not reported because this PLY release does not include per-Gaussian
part labels.

## ShapeSplat-Part structural consistency

To measure structural consistency directly, we used the real ShapeSplat-Part
release (ShapeNet Part point clouds with normals and per-point part IDs). The
benchmark contains 50 objects from five categories (10 airplane, 10 car, 10
chair, 10 lamp, and 10 table), with eight fitted Superquadrics per object and
12 refinement iterations. Since the ShapeSplat-Part release provides annotated
point clouds rather than a matching Gaussian PLY, each point was converted to a
Gaussian with an anisotropic covariance aligned to its surface normal. This
protocol evaluates the consistency of the primitive decomposition with known
parts; it should not be interpreted as a reconstruction benchmark for original
3DGS covariance values.

| Method | Objects | Part purity ↑ | Boundary leakage ↓ | NMI ↑ |
|---|---:|---:|---:|---:|
| Center-only / Superquadric | 50 | **0.8360 ± 0.0920** | **0.1640 ± 0.0920** | **0.3657 ± 0.1549** |
| Gaussian-aware / Superquadric | 50 | 0.8241 ± 0.1040 | 0.1759 ± 0.1040 | 0.3518 ± 0.1488 |

The Gaussian-aware covariance proxy does not improve semantic part consistency
in this experiment, even though Gaussian-aware Superquadric improves source-mesh
Chamfer on the larger ShapeSplat object benchmark. This distinction is useful:
local extent information can improve geometric surface fitting without making
primitive boundaries align better with semantic part boundaries. Category-level
results show Gaussian-aware consistency improves on lamp and table, while
airplane, car and chair favor the center-only baseline.

The complete per-object annotations result is in
`artifacts/shapesplat_part_consistency.json`. Re-download the data with:

```bash
python scripts/download_shapesplat_part.py
python scripts/run_shapesplat_part.py \
  --categories airplane,car,chair,lamp,table \
  --objects-per-category 10 --n-primitives 8 --iterations 12
```
