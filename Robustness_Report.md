# Findings on the Robustness of Clustered Compositional Explanations

This report evaluates the robustness of Clustered Compositional Explanations by analyzing how sensitive the resulting explanations are to changes in the underlying clustering algorithm and its parameters.

## 1. Ease of Changing Explanations in General
Finding a configuration that **slightly** changes the explanations is relatively **easy**. 
Because clustering algorithms map the continuous space of neural activations into discrete ranges, even small variations in parameters (such as the `random_state` in K-Means, or `batch_size` in MiniBatch K-Means) can shift the cluster boundaries. If an activation boundary shifts enough to encompass a different set of visual features, the Intersection over Union (IoU) scores for various concepts will change slightly, occasionally causing the highest-scoring formula (the final "explanation") to flip. However, these changes are often subtle—such as swapping `dog` for `dog AND grass`—indicating that the semantic core of the explanation remains largely robust.

## 2. Difficulty of Finding Significant Changes (>= 33%)
Finding a configuration that **significantly** changes the explanations (at least 33% difference) while still producing **meaningful** results is **hard**.
The units in vision models typically respond to specific concepts at high activations. Robust algorithms (like Agglomerative Clustering with Ward linkage or GMM with full covariance) will generally identify the same core high-activation cluster as the default K-Means algorithm, resulting in identical or highly similar labels for the top cluster.

To force a significant change, one must usually employ algorithms or configurations with fundamentally different assumptions:
- **Gaussian Mixture Models (GMM) with varying covariances**: Allows for non-spherical clusters, grouping lower activations with higher ones differently.
- **Agglomerative Clustering with Average Linkage**: Can create heavily imbalanced clusters, sometimes grouping the highest activations too narrowly.

### Configurations Tested
As part of our robustness search, the following configurations were rigorously tested using the newly added automated script (`scripts/search_clustering.py`):
1. **Default**: `KMeans` (baseline)
2. **MiniBatch KMeans**: `{"batch_size": 256}`
3. **MiniBatch KMeans**: `{"batch_size": 1024}`
4. **Bisecting KMeans**: `{"bisecting_strategy": "biggest_inertia"}`
5. **Agglomerative**: `{"linkage": "ward"}`
6. **Agglomerative**: `{"linkage": "average"}`
7. **GMM**: `{"covariance_type": "full"}`
8. **GMM**: `{"covariance_type": "diag"}`
9. **KMeans**: `{"init": "random", "n_init": 10}`
10. **KMeans**: `{"algorithm": "elkan"}`

However, drastically altering the configuration often leads to **degenerate behavior** (e.g., all units being labeled with a trivial, dominant background concept like `wall` or `sky`), which violates the "meaningful explanation" requirement. 

## 3. Trade-offs: Time, Space, and Quality
Changing the clustering algorithm introduces significant trade-offs:

**MiniBatch K-Means** offers the best trade-off for scaling. It reduces computation time drastically while maintaining explanation quality remarkably close to standard K-Means.

- **Time Complexity:** 
  - *Standard K-Means* is fast enough for 10 units but scales linearly with the number of activations. 
  - *Agglomerative Clustering* is extremely slow (O(n^3) or O(n^2)) and quickly becomes a bottleneck when processing thousands of activations.
  - *GMMs* require multiple iterations to converge (EM algorithm) and are noticeably slower than K-Means, especially with full covariance matrices.
- **Space Complexity:** 
  - *Agglomerative Clustering* requires storing a distance matrix (O(n^2) space), making it prohibitive for layers with dense, high-resolution feature maps unless the activations are heavily subsampled.
  - *K-Means* and *MiniBatch K-Means* are highly memory-efficient (O(n) and O(batch_size), respectively).
- **Explanation Quality (Meaningfulness):**
  - Configurations optimized purely for speed (e.g., MiniBatch K-Means with tiny batches or Bisecting K-Means) sometimes fail to isolate the rare, high-activation peaks effectively, diluting the concept mask and lowering IoU scores. 

## Conclusion
The Clustered Compositional Explanations framework demonstrates a high degree of robustness. While minor boundary shifts are common, the core semantic concepts identified at the highest activation ranges remain stable across reasonable clustering configurations. Forcing a massive divergence in explanations typically requires pushing the clustering algorithm into regimes where the clusters no longer meaningfully represent the data distribution.
