# Literature map and implementation position

This project treats a pretrained 3DGS cloud as the input representation. The
reconstruction stage and original multiview images are outside the method.

| Line of work | Main idea | Relation to this project |
| --- | --- | --- |
| 3D Gaussian Splatting | Dense anisotropic Gaussians with position, covariance/orientation, opacity and appearance | Input representation; covariance is converted to a geometric signal |
| SuperDec | Decompose shapes into compact analytic primitives | Motivates compact structural output |
| SuperFlex | Flexible superquadric primitive fitting | Point-cloud baseline target; this MVP keeps stable superquadric exponents and fits pose/extent |
| ResFit / SuperFrusta | Fit superfrusta to geometry with interpretable part parameters | Implemented as the `superfrustum` output mode |
| GaussianBlock | Combine Gaussian splats with block-like structure during reconstruction | Related hybrid representation; this project decomposes an already-trained cloud |
| PartGS | Part-aware Gaussian representation learned jointly with reconstruction | Related structured Gaussian representation; this project avoids reconstruction retraining |
| BrepGaussian | Boundary representation / CAD-style Gaussian structure | Motivates explicit, editable geometry and future B-rep export |
| ShapeSplat | Pretrained Gaussian object representations with source geometry | Intended benchmark: PLY/NPZ loader and source-point metrics are provided |
| SceneSplat | Structured scene-level splatting | Relevant extension from object decomposition to scene-level assemblies |

## Hypotheses tested by the code

1. Gaussian centers alone provide a point-cloud baseline.
2. The covariance trace and principal directions provide extent/orientation
   evidence that should improve primitive assignment and moment fitting.
3. Opacity should downweight low-confidence or nearly invisible splats.
4. Compact primitive assemblies can be evaluated without rendering by sampling
   primitive surfaces and measuring Chamfer/coverage against source geometry.

The `scripts/run_ablation.py` experiment separates these hypotheses. For a
ShapeSplat subset, report the mean and standard deviation of the JSON metrics
over objects and include the number of Gaussians and primitive count so the
compression ratio is comparable across categories.

