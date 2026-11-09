# Real ShapeSplat run

Subset: `ShapeSplats/modelnet_3dgs_preprocessed`, category `bowl`, split
`test`, objects `bowl_0065` through `bowl_0068`.

All objects were downloaded from the real gated dataset and contain Gaussian
coordinates, scales, quaternions, opacity, normals and 3-channel color.

| Method | Mean final fitting loss | Mean source-mesh Chamfer | Mean source coverage @ 0.05 | Primitive count |
| --- | ---: | ---: | ---: | ---: |
| Center-only baseline, superquadric | 2.9084 | 0.002148 | 0.9621 | 16 |
| Gaussian-aware, superquadric | 2.0838 | 0.001737 | 0.9738 | 16 |
| Gaussian-aware, superfrustum | 0.7608 | 0.005078 | 0.8815 | 16 |

The source meshes are the matching `bowl_0065.off` through `bowl_0068.off`
files from ModelNet40 and are normalized by bounding box before comparison.
The complete per-object JSON is `artifacts/shapesplat_subset.json`; it also
contains the opacity surface proxy metrics.

