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

### B1: median pairwise distance

B1 compares each object with every other object separately. The object score is
the median of those cosine distances. This reduces the effect of one unusually
different peer while keeping the result easy to explain.

Important settings are the distance measure and the peer aggregation rule. A
median can still be unstable when a scene contains only a few objects.

### P1: patch comparison

P1 will compare local DINOv2 patch features between objects. It should improve
the detection of cracks, missing parts and small surface changes. It requires
more memory and careful spatial matching.

Important settings are the feature layer, patch size, number of neighbours and
spatial matching rule.

## Current evidence

B0 and B1 were tested on the same deterministic 40-scene training cohort. Both
ranked an anomalous object first in 26 scenes. B0 produced mean scene average
precision of 0.7771 and B1 produced 0.7758. B1 did not improve B0 on this
cohort. These are development results, not final test results.

## References

- Bhunia et al., [Odd-One-Out: Anomaly Detection by Comparing with Neighbors](https://openaccess.thecvf.com/content/CVPR2025/html/Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.html), CVPR 2025.
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- [Official Odd-One-Out repository and dataset instructions](https://github.com/VICO-UoE/OddOneOutAD).
