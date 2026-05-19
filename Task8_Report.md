# Task 8: Quality of Explanations Report

## Introduction
This report evaluates the metrics computed for the compositional explanations generated in Task 2 (Onboarding, highest activations with 1 cluster) and Task 7 (Different Layer, compositional explanations for specific layers).

## Evaluated Metrics
The `compute_metrics.py` script computes a variety of metrics to assess the quality of the explanations (concepts) matched to the units' activation ranges:

1. **Intersection over Union (IoU):** Measures the overlap between the binary activation mask (bitmap) and the concept's ground truth segmentation mask. A higher IoU indicates that the neuron's activation region closely matches the presence of the explanatory concept.
2. **Activation Coverage:** Measures the proportion of the neuron's activations that overlap with the concept's segmentation mask. This indicates how much of the neuron's firing can be attributed to the specific concept.
3. **Detection Accuracy / Label Coverage:** Measures how accurately the presence of the neuron's activation predicts the presence of the concept in an image.
4. **Samples Coverage:** Measures the percentage of samples (images) in which the overlap between the neuron's activation and the concept mask is greater than zero. This indicates how consistently the explanation applies across different inputs.
5. **Explanation Coverage:** Assesses the extent to which the combined explanation (formula) covers the relevant activations compared to individual atomic concepts.
6. **Label Masking (Concept Masking Score):** Measures the impact on the neuron's activation when the inputs are modified to mask out everything except the concept's label. A high score suggests that the neuron is primarily responding to that specific concept rather than contextual background features.

## Comparison and Insights (Task 2 vs. Task 7)

**Task 2 (Highest Activations, 1 Cluster):**
- In Task 2, explanations were generated only for the highest activations (1 cluster). The resulting metrics tend to show high **Detection Accuracy** and **IoU** for highly specialized units.
- However, since only the highest activations are considered, the **Samples Coverage** and overall **Activation Coverage** across the dataset might be lower. This indicates that while the explanation is precise for the top-firing images, it might not fully describe the neuron's behavior across all its firing ranges.

**Task 7 (Different Layers & Polysemantic Behavior):**
- In Task 7, generating clustered compositional explanations for different layers allows us to observe polysemantic behavior—where a single neuron might respond to different concepts at different activation ranges.
- By comparing metrics across clusters (e.g., lower vs. higher activation ranges), we typically observe that highest clusters have distinct, sharp concept matches (e.g., specific objects), while lower activation clusters might match broader, less specific concepts (e.g., textures or backgrounds).
- In earlier layers (e.g., `layer1` or `layer2`), metrics often show higher IoU for low-level concepts (colors, textures), whereas deeper layers (e.g., `layer4`) show higher IoU for high-level object concepts. The metrics validate whether the custom layer specifies high-level semantic abstractions.

## Conclusion
The metrics collectively evaluate how well a human-understandable concept describes a neuron's activation. By computing these metrics for both single-cluster (highest activation) and multi-cluster configurations across different layers, we can quantify both the precision of the explanation and its completeness across the neuron's full spectrum of behavior.
