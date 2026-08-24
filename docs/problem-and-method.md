# Problem and method

## Problem

The project asks a simple question:

> When several similar objects appear in multiple views of one scene, which
> object is most unusual compared with its peers?

The system does not depend on a large reference library of normal products.
Objects visible in the uploaded scene act as references for each other.

![Verified ToysAD-8K examples](assets/toysad-examples.png)

The examples include missing parts, cracks, fractures, rotation, translation,
shape deformation, material or colour changes, and scenes with two anomalies.

## Main challenges

1. Normal objects can differ in pose, size, lighting, and visibility.
2. A defect may affect only a small part of an object.
3. Rotation and translation depend on the arrangement of peer objects.
4. Several unusual objects can weaken the peer reference.
5. ToysAD-8K stores anomalous objects first in metadata order.
6. The inspected release has no all-normal scenes.
7. The same object must be matched across views before evidence is combined.

## Processing stages

The complete data flow and trust boundary are documented in
[System architecture](architecture.md).

### 1. Object proposals

NVIDIA LocateAnything-3B proposes object bounding boxes from each RGB view. It
is an object proposal component, not an anomaly detector. The supplied dataset
geometry is used only by the evaluator and by the controlled reference route.

### 2. Object masks

SAM 2.1 receives one box prompt for each object and produces a mask. Three
routes are kept separate throughout the experiments:

| Route | Region source | Purpose |
| --- | --- | --- |
| `supplied_mask` | Dataset mask | Oracle reference for ranking |
| `reference_box` | Dataset box passed to SAM 2.1 | Segmentation upper bound |
| `automatic_box` | LocateAnything box passed to SAM 2.1 | Deployment candidate |

SAM 2.1 segments prompted regions. It does not decide whether an object is
unusual.

### 3. Common object preparation

All routes use the same mask quality checks, square crop, fixed padding,
background treatment, and encoder input size. This keeps route comparisons
focused on region quality rather than unrelated preprocessing differences.

### 4. Frozen visual representation

A frozen visual encoder converts each prepared object into a global feature
vector and a set of patch features. DINOv2-small remains the anchor model until
the bounded DS1 encoder screen is complete.

### 5. Peer comparison

The first ranking methods are:

- **B0, whole-object peer mean:** one minus the cosine similarity between an
  object and the mean feature of its peers.
- **B1, median pairwise distance:** the median cosine distance from one object
  to every other object in the scene.
- **P1, patch comparison:** local patch matching for small or spatially limited
  differences.

B0 is the simple anchor. B1 tests a more robust peer aggregation rule. P1 adds
local evidence and can produce a coarse unusual-region heatmap. That heatmap is
supporting evidence, not a pixel-accurate defect mask.

### 6. Geometry and multi-view fusion

Geometry features are evaluated separately for pose and arrangement changes.
Per-view rankings remain visible. Cross-view identity matching then combines
evidence only for confirmed object matches. The final result is a ranked list
of unusual candidates with per-view scores, masks, and supporting evidence.

## Current evidence

| Stage | Evidence | Status |
| --- | --- | --- |
| Full-dataset EDA | 7,396 catalogue scenes and 4,107 eligible scenes | Complete |
| Primary selection | 2,000 scenes split 1,000 / 600 / 400 | Complete |
| DS0 proposal run | 400 images and 1,616 proposals with no operational failures | Complete |
| DS0 SAM run | 3,256 masks across the reference-box and automatic-box routes | Complete |
| DS0 mask evaluation | Automatic-route recall at mask IoU 0.75 of 0.9396 and mean matched mask IoU of 0.9485 | Complete |
| Historical ranking baseline | B0 and B1 each ranked an anomaly first in 26 of 40 scenes | Reproducible baseline |
| DS1 representation screen | B0, B1, P1, and encoder comparison | Next |

The historical 40-scene baseline produced mean scene average precision of
0.7771 for B0 and 0.7758 for B1. These results are development evidence, not a
final test result.

## Claim limits

- LocateAnything proposes regions. It does not perform segmentation or anomaly
  detection.
- SAM 2.1 produces prompted masks. It does not classify unusual objects.
- A P1 heatmap is a coarse representation signal, not a defect mask.
- DS0 shows operational readiness and region quality on 20 scenes. It does not
  freeze the final deployment route.
- ToysAD-8K cannot establish all-normal false-positive performance because the
  audited release contains no all-normal scenes.

## References

- Bhunia et al., [Odd-One-Out: Anomaly Detection by Comparing with Neighbors](https://openaccess.thecvf.com/content/CVPR2025/html/Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.html), CVPR 2025.
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- [NVIDIA LocateAnything-3B model](https://huggingface.co/nvidia/LocateAnything-3B).
- [SAM 2 repository](https://github.com/facebookresearch/sam2).
- [Official Odd-One-Out repository](https://github.com/VICO-UoE/OddOneOutAD).
