## geometric/

### Overview

This module focuses on Analyzing SAEs using Geometric Methods.

### Background

Our primary object of study is the **SAE decoder matrix**, denoted as $$W_{dec}\in \mathbb R^{d_{sae}\times d_{model}}$$

This matrix represents a linear map $$f:\mathbb R^{d_{sae}}\to\mathbb R^{d_{model}}$$

Since each dimension $\mathbf e_i\in\mathbb R^{d_{sae}}$ is a well-defined monosemantic direction (feature), we can study $f$ by examining the vector $$\mathbf v_i:=f(\mathbf e_i)\in \mathbb R^{d_{model}},\quad i=1,2,\cdots,d_{sae}$$

These vectors correspond to the slices of $W_{dec}$ along dimension $0$ ( `v[i] = W_dec[i,:]` in code). Each $\mathbf v_i$ acts as a feature vector embedded in the original semantic space of dimension $d_{model}$ .

It is important to note that the number of $\mathbf v_i$ (features), $d_{sae}$, is significantly larger than the model dimension $d_{model}$ (i.e. $d_{sae}\gg d_{model}$ ). Despite this, the vectors $\mathbf v_i$ tend to be nearly orthogonal. We investigate this phenomenon through the lens of Superposition Geometry.

### Implementation

We use **Cosine Similarity** as the primary distance metric for NLP tasks. Then we get the following analysis tools

- Similarity Matrix ( $N\times N$ )
- Top K Similarity Matrix ( $N\times K$ )
- Dimensionality Reduction Visualization

more...is coming
