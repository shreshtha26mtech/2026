---
layout: distill
title: "Approximating Faster Transformers"
description: This post offers a comprehensive overview of sketching and sampling algortihms for DistilBert
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: Anonymous
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations: Anonymous
# must be the exact same name as your blogpost
bibliography: 2026-04-27-fastermatrices.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

Optimizing matrix multiplication is a problem as old as time. The product of two matrices $\mathbf{A}$ and $\mathbf{B}$ can be looked at as the inner product of rows of $\mathbf{A}$ with columns of $\mathbf{B}$ or the outer product of columns of $\mathbf{A}$ with rows of $\mathbf{B}$. 

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/innerproduct.png' | relative_url }}" alt="Inner Product">

With traditional algorithms having a time complexity of $O(N^3)$, and researchers like Strassen coming up with better recursive approaches, the time complexity is still only down to about $O(N^{2.81})$, which is still a pretty heavy task.


Ever since they have been introduced, transformers have gained tremendous popularity. However, a major bottleneck in the attention mechanism is  matrix multiplication that powers the core of the transformers. Optimizing matrix multiplication would mean an improvement in the speed of these models. This blog aims at utilizing several techniques from RandNLA to improve DistilBERT.

## Introduction
RandNLA is a field of linear algebra that uses randomization to improve very large-scale algorithms in linear algebra (see [RandNLA](https://arxiv.org/abs/2302.11474)). The "Sketch and Solve" paradigm refers to using a sketching matrix to bring down the size of the problem and then solving the problem for a compressed size.

### Sketching for Matrix Multiplication

When applying the sketch and solve paradigm to the multiplication of two massive matrices $\mathbf{A}$ ($m \times n$) and $\mathbf{B}$ ( $n \times p$), the sketching matrix $\mathbf{S}$ is used.

Instead of computing the exact product $\mathbf{C} = \mathbf{A}\mathbf{B}$, which requires $O(mnp)$ operations, we compute an approximate product using a sketching matrix $\mathbf{S}$ of size $k \times n$ (where $k \ll n$). The approximation is constructed by compressing the columns of $\mathbf{A}$ and the rows of $\mathbf{B}$:

$$\tilde{\mathbf{C}} = (\mathbf{A} \mathbf{S}^\top) (\mathbf{S} \mathbf{B})$$

Where:
* $\mathbf{A} \mathbf{S}^\top$ is an $m \times k$ matrix (a compressed version of $\mathbf{A}$).
* $\mathbf{S} \mathbf{B}$ is a $k \times p$ matrix (a compressed version of $\mathbf{B}$).


<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sketch_and_solve.png' | relative_url }}" alt="Sketching Matrix">

For the approximation $\tilde{\mathbf{C}}$ to be accurate, the sketching matrix $\mathbf{S}$ must satisfy specific some properties 

* The sketching matrix, when squared, acts as the identity matrix in expectation. If $\mathbf{S}$ is drawn from an appropriate distribution, then $\mathbb{E}[\mathbf{S}^\top \mathbf{S}] = \mathbf{I}_n$. 
  This ensures that the approximation is an unbiased estimator of the true product: $\mathbb{E}[\tilde{\mathbf{C}}] = \mathbb{E}[\mathbf{A} (\mathbf{S}^\top \mathbf{S}) \mathbf{B}] = \mathbf{A}\mathbf{B}$.

* A matrix $\mathbf{S}$ has the property that if, for any vector $\mathbf{x}$, the norm satisfies 
  
  $(1 - \epsilon)\|\mathbf{x}\|_2^2 \le \|\mathbf{S}\mathbf{x}\|_2^2 \le (1 + \epsilon)\|\mathbf{x}\|_2^2$. 



* A sketching matrix $\mathbf{S}$ should also satisfy
  
* $\| \mathbf{A}\mathbf{B} - (\mathbf{A}\mathbf{S}^\top)(\mathbf{S}\mathbf{B}) \|_F \leq \epsilon \| \mathbf{A} \|_F \| \mathbf{B} \|_F$. 
  
Here we are bounding the error ensuring accurate results

## Sketching and Sampling

The sketches can be of any type e.g.  Gaussian, Count Sketch, Hadamard, and Learned sketches.
* **Gaussian sketches** involve projecting data using a matrix where entries are sampled independently from a normal distribution $\mathcal{N}(0, \frac{1}{k})$ to preserve distances [Sobczyk and Luisier, 2022]. 
*  **Count Sketch** offers a sparser alternative by using hash functions to map rows to buckets and randomly flipping their signs [Clarkson and Woodruff, 2017]
*   **Hadamard-based sketches** (like the PHD matrix) combine randomized Hadamard transforms with uniform subsampling for structured efficiency [Clarkson and Woodruff, 2017].

For the broader problem of matrix multiplication, not just sketching techniques but sampling algorithms have also been studied which have shown good promise.

The core idea behind the sampling algorithms is to select a subset of the original data. Several algorithms like uniform sampling, random sampling, priority sampling, threshold sampling, and leverage score sampling are popular. They basically "look" and find the "most important" parts of the entire matrix and then perform the multiplication on the smaller subset of the matrices.

* **Uniform Sampling** samples each row with the probability $p=k/n$ of selection, effectively treating every single row as equally important[Cohen et al., 2015].
* **Leverage Score Sampling** selects rows based on their statistical influence called leverage scores which are either calculated using SVD or QR-decomposition [Drineas et al., 2012].
* **Priority and Threshold Sampling**  Is an innovative method which selects the rows based on their "importance" i.e. rows beyond a certain threshold are selected [Daliri et al., 2025].

###  Sketch and Solve for Transformers

Transformers as an architecture is the industry standard becauase it has improved context as compared to previous architectures such as LSTM and RNNS, and provides the ability to parallelize the block which means faster training and inference 

It's architecture is split into two distinct blocks, both composed of a stack of $N$ identical layers:

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/transformers.png' | relative_url }}" alt="Transformers architecture">


**Encoder**

The Encoder’s primary function is to map an input sequence of symbols $(x_1, ..., x_n)$ into a rich, continuous representation $\mathbf{Z} = (z_1, ..., z_n)$.

Where each token is allowed to look at every other token available and hence it essentially understands the sementics

**Decoder**
The Decoder generates the output sequence $(y_1, ..., y_m)$ one element at a time, being **auto-regressive** (consuming previously generated symbols as input for the next) instead of looking at the entire sentence like the encoder does

The core logic which powers both encoder and decoder based architecture is the attention mechanism; It basically creates a better, richer representation of our input context.

##### Attention Mechanism

The input to the attention layer is our input embedding matrix  $\mathbf{X}$. We project it using three learned weight matrices:
* **Query ($\mathbf{Q}$):** What the token is looking for.
* **Key ($\mathbf{K}$):** What the token identifies as.
* **Value ($\mathbf{V}$):** The actual content the token holds.


<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/mha.png' | relative_url }}" alt="Multihead Attention">


Attention score is calculated using:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$



* First, we calculate the dot product between the Query matrix and the Transpose of the Key matrix, in a way for a token it calculates the similarity between every single token and every other token in the sequence
  
* These are then normalized by a factor of $\sqrt{d_k}$ and then we apply **Softmax** over it, which converts the raw scores into probabilities.
  
* The normalized weights are then multiplied with the matrix $\mathbf{V}$

Calculating this basically tells us the relationship of one token in a sequence to all the other tokens. However, the operation $\mathbf{Q}\mathbf{K}^\top$ is a very heavy operation.
To compute this, we have to perform operations proportional to the square of the sequence length ($N^2$) (assuming the lenght of our sequence to be $N$).

More specifically,
* If $N=512$, the attention matrix has roughly 260,000 entries.
* If $N=4096$, it jumps to over 16 million entries.

It is quite evident that improvement with $\mathbf{Q}\mathbf{K}^\top$ can lead to improvement in the overall speed of the transformers.

##### Better Transformers

In regards to optimizing the attention matrix, a lot of work has already been done. 
Instead of computing the full matrix, architectures like the **Sparse Transformer** enforce a fixed sparsity pattern, such as a sliding window where tokens only attend to their immediate neighbors [Child et al., 2019]. This reduces the complexity to approximately $O(N\sqrt{N})$. 
A more dynamic approach was introduced by the **Reformer**, which replaces exact dot-product attention with Locality-Sensitive Hashing (LSH) [Kitaev et al., 2020]. By grouping similar query and key vectors into the same hash buckets, the model computes attention only within these buckets, effectively dropping the complexity to $O(N \log N)$.

 The **Linformer** relies on the low-rank property of the attention matrix. [Wang et al., 2020].Linformer projects the Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices into a lower-dimensional space using learned linear projections ($\mathbf{E}$ and $\mathbf{F}$). By compressing the original $(N \times d)$ matrices into much smaller $(k \times d)$ matrices, the attention operation becomes linear $O(N)$ in time and space, 

**LevAttention** utilizes the concept of leverage scores (generalized $f$-sensitivities) to identify a small "universal set" of keys that dominate the attention scores for *any* query [Kannan et al., 2024]. This method proves that a subset of high-leverage keys, independent of the sequence length, captures the vast majority of the attention mass, allowing for efficient $O(N \cdot \text{poly}(d/\epsilon))$ computation.

**Matrix Product Sketching via Coordinated Sampling** proposes estimating the product $\mathbf{Q}\mathbf{K}^\top$ directly using coordinated random sampling [Daliri et al., 2025]. Unlike traditional linear sketching (such as Johnson-Lindenstrauss projections) which can be inefficient for sparse data, coordinated sampling (specifically Priority Sampling) selects rows from $\mathbf{Q}$ and $\mathbf{K}$ based on their norms using a shared random seed. 

We have used several of these aforementioned algortihms to approximate transformers and see how these perform against Indic Language dataset for various tasks. The goal here is not to come up with a one size fits all solution but rather to find out how each of these perform in practice and the pitfalls (if any for several of these algortihms)

---

### 1. Learned Sketch (Linformer)
**The Concept:**
The Linformer is based on the **Low-Rank Approximation** of the attention matrix. Instead of a full $N \times N$ interaction, it assumes that the self-attention mechanism can be approximated by projecting the Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices into a lower-dimensional space [Wang et al., 2020].

**Definition (Learned Projection):**
We define two learned projection matrices, $\mathbf{E} \in \mathbb{R}^{k \times N}$ and $\mathbf{F} \in \mathbb{R}^{k \times N}$, where $k \ll N$. These matrices project the high-dimensional sequence length $N$ down to a compressed dimension $k$.
$$\mathbf{K}_{proj} = \mathbf{E}\mathbf{K}, \quad \mathbf{V}_{proj} = \mathbf{F}\mathbf{V}$$

**Mathematical Pseudocode:**
```latex
Input: Q (N x d), K (N x d), V (N x d)
Parameters: E, F (Learned Projections)

1. Compress Keys and Values:
   K_proj = E * K   // Result: (k x d)
   V_proj = F * V   // Result: (k x d)

2. Compute Attention Scores:
   Scores = (Q * K_proj^T) / sqrt(d)   // Result: (N x k)

3. Compute Probabilities:
   Weights = Softmax(Scores)

4. Compute Context:
   Output = Weights * V_proj   // Result: (N x d)
````



-----

### 2\. LevAttention (Leverage Score Sampling)


LevAttention uses **Statistical Leverage Scores** to identify the most "influential" keys in the sequence. These scores measure how much a specific row of $\mathbf{K}$ exerts influence on the solution of a linear regression problem involving $\mathbf{K}$. High leverage indicates a key that is unique or critical to the subspace [Kannan et al., 2024].

For a matrix $\mathbf{K} \in \mathbb{R}^{N \times d}$, the leverage score $\tau_i$ of the $i$-th row $\mathbf{k}_i$ is defined as:
$$ \tau_i(\mathbf{K}) = \mathbf{k}_i (\mathbf{K}^\top \mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{k}_i^\top $$
where $\lambda$ is a small damping factor for stability.



```
Input: K (N x d), budget k, regularization lambda

1. Compute Gram Matrix:
   G = K^T * K + lambda * I

2. Compute Leverage Scores for each row i:
   tau_i = K_i * G^(-1) * K_i^T

3. Identify Universal Set U:
   U = {indices of the top-k largest tau_i}

4. Create Mask M:
   M_{ij} = 0 if j in U else -infinity

5. Compute Masked Attention:
   Output = Softmax((Q * K^T + M) / sqrt(d)) * V
```


-----

### 3\. Priority Sampling (Coordinated Sampling)

**The Concept:**
This method avoids the computationally expensive matrix inversion of LevAttention by using **Coordinated Sampling**. It selects keys based on their "energy" (squared norm) using a shared randomness source. This ensures that if a key is "heavy" (important), it is selected consistently across different views or distributed nodes without communication [Daliri et al., 2025].

**Definition (Priority Rank):**
For each key $\mathbf{k}_i$, we generate a random hash $h_i \sim \text{Uniform}(0, 1)$. The priority rank $r_i$ is defined as the hash scaled by the key's energy:
$$ r_i = \frac{h_i}{\|\mathbf{k}_i\|_2^2} $$
Keys with higher energy (large norms) get smaller ranks and are prioritized for selection.

**Mathematical Pseudocode:**

```latex
Input: K (N x d), budget k, random seed s

1. Generate Shared Randomness:
   h_i ~ Uniform(0, 1) for i in 1..N (generated using seed s)

2. Compute Ranks (Priority):
   r_i = h_i / ||K_i||^2   // Lower rank = Higher priority

3. Determine Threshold:
   tau = (k+1)-th smallest value in r

4. Select Keys:
   S = {i | r_i <= tau}

5. Compute Attention:
   Output = Softmax((Q * K_S^T) / sqrt(d)) * V_S
```

**Implementation Pseudocode:**


```python
def priority_sampling_attention(query, key, value, target_k):
    # 1. Calculate 'Energy' of each Key
    # Squared L2 norm of each row
    key_norms = norm(key, dim=-1) ** 2
    
    # 2. Generate Hash Values
    # Random values shared/synchronized by seed
    h = random_uniform(0, 1, size=seq_len)
    
    # 3. Compute Priority Ranks
    # Divide hash by energy; heavy keys get smaller ranks
    ranks = h / key_norms
    
    # 4. Determine Threshold
    # Find the rank value that cuts off exactly 'target_k' items
    threshold = find_kth_smallest(ranks, k=target_k)
    
    # 5. Create Mask
    mask = ranks <= threshold

    # 6. Masked Attention
    scores = matmul(query, key.transpose()) / sqrt(dim)
    scores = scores.masked_fill(~mask, -inf)
    
    return matmul(softmax(scores), value)
```

-----

### 4\. L1 Lewis Weight Sampling

**The Concept:**
This is a robust generalization of leverage scores. Instead of relying on the $L_2$ norm (which can be sensitive to outliers), it iteratively computes **$L_1$ Lewis Weights**. These weights provide a sensitivity measure for the $L_1$ norm, ensuring that the selected keys capture the geometry of the data distribution more robustly [Cohen & Peng, 2015].

**Definition (L1 Lewis Weight):**
The weights $w$ are found via an iterative process. A weight $w_i$ for row $\mathbf{k}_i$ satisfies:
$$ w_i = \tau_i(\mathbf{W}^{-1/2}\mathbf{K}) $$
where $\mathbf{W}$ is the diagonal matrix of weights, and $\tau_i(\cdot)$ is the standard leverage score of the re-weighted matrix.

**Mathematical Pseudocode:**

```latex
Input: K (N x d), iterations T

1. Initialize Weights:
   w_i = 1 for all i in 1..N

2. Iterate T times:
   a. Re-weight Matrix:
      K_tilde = K / sqrt(w)
   
   b. Compute Leverage Scores of K_tilde:
      tau = LeverageScores(K_tilde)
   
   c. Update Weights:
      w = sqrt(w * tau)

3. Select Keys:
   S = {indices of largest w_i}

4. Output: Attention Restricted to S
```

**Implementation Pseudocode:**


```python
def l1_sampling_attention(query, key, value, top_k):
    # 1. Initialize Weights
    weights = ones(seq_len)
    
    # 2. Iterative Update
    for _ in range(max_iter):
        # Scale matrix by current weights
        K_weighted = key / sqrt(weights)
        
        # Compute leverage scores of weighted matrix
        # (Using efficient QR or SVD)
        Q_matrix = qr(K_weighted)
        scores = sum(Q_matrix ** 2, dim=1)
        
        # Update weights (Geometric mean of old weight and score)
        weights = sqrt(weights * scores)

    # 3. Select Top Keys
    top_indices = topk(weights, k=top_k)
    mask = create_mask(top_indices)

    # 4. Standard Masked Attention
    scores = matmul(query, key.transpose()) / sqrt(dim)
    scores = scores.masked_fill(~mask, -inf)
    
    return matmul(softmax(scores), value)
```

```

##### Experiment Setup

For the course of our experiments, we have picked the  **DistilBERT** (6 encoder blocks, 12 heads each) architecture which distilled version of **BERT** that retains 97% of its performance while being 40% lighter.


<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/distbert.drawio (1).png' | relative_url }}" alt="DistilBert">

We have modifed the attention mechanism to use technqiues such as:
* Priority sampling
* Leverage Score sampling
* L1-score sampling
* Learned Sketches

We have then used these models for finetuning and inference across several different tasks (more on this later) 

We have used the **IndicGLUE Benchmark** (Hindi Language subsets), courtesy of [AI4Bharat](https://indicnlp.ai4bharat.org/pages/indic-glue/). 

| Task & Dataset | Description | Unique Labels | Hindi Example |
| :--- | :--- | :--- | :--- |
| **Sentiment Analysis**<br>`iitp-mr.hi` | Classifies movie reviews as positive, negative, or neutral. | `0` (Negative)<br>`1` (Neutral)<br>`2` (Positive) | *“यह फिल्म देखने लायक है, कहानी बहुत अच्छी है।”* |
| **News Classification**<br>`bbca.hi` | Classifies news articles into 14 distinct topics (e.g., Sports, India). | 14 Topics<br>(`india`, `sport`, `entertainment`, etc.) | *“भारतीय क्रिकेट टीम ने आज ऐतिहासिक जीत दर्ज की।”* |
| **Discourse Mode**<br>`md.hi` | Identifies the rhetorical role of a sentence in a narrative. | `Argumentative`<br>`Descriptive`<br>`Dialogic`<br>`Informative`<br>`Narrative` | *“एक बार की बात है, एक घना जंगल था।”*<br>(Narrative) |
| **Causal Reasoning (COPA)**<br>`copa.hi` | Selects the most plausible cause or effect for a given premise. | `0` (Choice 1)<br>`1` (Choice 2) | **Premise:** *“लड़के का पैर फिसल गया।”*<br>**Correct:** *“वह गिर गया।”* |
| **Named Entity Recognition**<br>`wiki-ner.hi` | Tags entities like Persons, Locations, and Organizations in text. | `O`, `B-PER`, `I-PER`<br>`B-ORG`, `I-ORG`<br>`B-LOC`, `I-LOC` | *“**राहुल** (B-PER) **गांधी** (I-PER) **दिल्ली** (B-LOC) में हैं।”* |
| **Section Title Prediction**<br>`wstp.hi` | Predicts the correct section title for a Wikipedia paragraph. | `0` (Title A)<br>`1` (Title B)<br>`2` (Title C)<br>`3` (Title D) | **Text:** (Paragraph about Cricket rules)<br>**Correct:** *“नियम” (Rules)* |


All the models have been fine-tuned on kaggle using the GPU P100 and are available for inference freely on huggingface