# Problem and proposed method

## Problem

The project asks a simple question:

> When several similar objects appear in one image, which objects are unusual
> compared with the others?

The system does not start with a large library of normal product images. The
objects visible in the current scene act as references for each other.

![Verified ToysAD-8K examples](assets/toysad-examples.png)

The figure shows missing parts, cracks, fractures, rotation, translation,
shape deformation, material or colour changes, and a scene with two anomalies.

## Main challenges

1. Normal objects can have different poses, sizes and lighting.
2. Some defects affect only a small part of an object.
3. Rotation and translation are relative to the peer arrangement.
4. More than one unusual object can weaken the peer reference.
5. ToysAD-8K stores anomalous objects first in its metadata. Object order must
   not be used as a clue.
6. The inspected release has no all-normal scene examples.

## Proposed stages

### B0: whole-object peer mean

DINOv2 converts each masked object crop into a feature vector. For each object,
B0 calculates the mean feature of the other objects. Its score is one minus the
cosine similarity between the object and that peer mean. A larger value means
the object is less similar to its peers.

Important settings are the camera view, mask threshold, crop padding, feature
layer and image background. B0 is fast and easy to explain. Its main weakness
is that one whole-object vector can hide a small defect.

### B1: nearest peer

B1 will compare each object with its most similar peer instead of the mean of
all peers. This can reduce the effect of a second anomaly contaminating the
reference. It can also select a poor peer when normal objects vary strongly.

Important settings are the distance measure and the number of neighbours.

### P1: patch comparison

P1 will compare local DINOv2 patch features between objects. It should improve
the detection of cracks, missing parts and small surface changes. It requires
more memory and careful spatial matching.

Important settings are the feature layer, patch size, number of neighbours and
spatial matching rule.

## Current evidence

B0 was tested on a deterministic 40-scene training cohort. It ranked an
anomalous object first in 26 scenes and produced a mean scene average precision
of 0.7771. These results support the next experiment, but they are not final
test results.

## References

- Bhunia et al., [Odd-One-Out: Anomaly Detection by Comparing with Neighbors](https://openaccess.thecvf.com/content/CVPR2025/html/Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.html), CVPR 2025.
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- [Official Odd-One-Out repository and dataset instructions](https://github.com/VICO-UoE/OddOneOutAD).
