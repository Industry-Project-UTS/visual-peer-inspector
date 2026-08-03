# Visual Peer Inspector

Visual Peer Inspector is a computer vision prototype for inspecting small
groups of similar objects without requiring a stored library of normal product
images.

The objects in the current scene act as references for one another. The system
compares them, identifies unusual objects and returns one of three decisions:
pass, flag for inspection, or manual review.

## Project scope

The initial implementation focuses on:

- a bounded ToysAD-8K evaluation subset;
- a one-view core experiment;
- supplied object masks;
- frozen DINOv2 visual features;
- peer-comparison anomaly scoring;
- all-normal, single-anomaly and multi-anomaly cases; and
- reproducible experiment tracking with ClearML.

A matched two-view comparison will be considered after the one-view baseline is
working reliably.

## Team

- Sreekar Reddy
- Vaibhav Bairathi

UTS 32040 Industry Project, Spring 2026.

## Development workflow

Development takes place on short-lived branches. Changes are merged into
`main` through pull requests reviewed by the other team member.

## Experiment tracking

Experiments are tracked in the
[Visual Peer Inspector ClearML dashboard](https://app.clear.ml/projects/9a5b29af465e4b51b09af451e62e8b2b/overview).

ClearML records experiment configurations, Git revisions, metrics and selected
artifacts. Each team member uses individual credentials in the shared
workspace.

## Data and credentials

Raw datasets, downloaded models, API credentials and private university
documents are not committed to this repository. Dataset download and setup
instructions will be documented separately.

## Status

Repository setup and feasibility validation are in progress.
