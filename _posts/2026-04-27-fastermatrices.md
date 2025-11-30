---
layout: distill
title: "Approximating Faster Transformers"
description: This post offers a comprehensive overview of sketching and sampling algorithms for DistilBERT
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
    url: "[https://en.wikipedia.org/wiki/Albert_Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)"
    affiliations:
      name: Anonymous
  - name: Anonymous
    url: "[https://en.wikipedia.org/wiki/Boris_Podolsky](https://en.wikipedia.org/wiki/Boris_Podolsky)"
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
_styles: |
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
  /* --- New Grid Layouts --- */
  .layout-seq, .layout-3-2 {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 15px;
    margin-bottom: 20px;
  }
  .layout-seq > div, .layout-3-2 > div {
    text-align: center;
    min-width: 150px;
  }
  .layout-seq img, .layout-3-2 img {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 4px;
  }
  .layout-seq p, .layout-3-2 p {
    font-family: monospace;
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
    text-align: center;
  }
  /* 5 items in a row */
  .layout-seq > div {
    flex: 1 1 18%;
  }
  /* 3 top, 2 bottom */
  .layout-3-2 > div:nth-child(-n+3) {
    flex: 1 1 30%;
  }
  .layout-3-2 > div:nth-child(n+4) {
    flex: 1 1 45%;
  }
---

Optimizing matrix multiplication is a problem as old as time. The product of two matrices $\mathbf{A}$ and $\mathbf{B}$ can be looked at as the inner product of rows of $\mathbf{A}$ with columns of $\mathbf{B}$ or the outer product of columns of $\mathbf{A}$ with rows of $\mathbf{B}$.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/innerproduct.png' | relative_url }}" alt="Inner Product">

With traditional algorithms having a time complexity of $O(N^3)$, and researchers like Strassen coming up with better recursive approaches, the time complexity is still only down to about $O(N^{2.81})$, which is still a pretty heavy task.

Ever since they were introduced, Transformers have gained tremendous popularity. However, a major bottleneck in the attention mechanism is matrix multiplication, which powers the core of the transformers. Optimizing matrix multiplication would mean an improvement in the speed of these models. This blog aims at utilizing several techniques from RandNLA to improve DistilBERT.

## Introduction
RandNLA is a field of linear algebra that uses randomization to improve very large-scale algorithms in linear algebra (see [RandNLA](https://arxiv.org/abs/2302.11474)). The "Sketch and Solve" paradigm refers to using a sketching matrix to bring down the size of the problem and then solving the problem for a compressed size.

### Sketching for Matrix Multiplication

When applying the sketch and solve paradigm to the multiplication of two massive matrices $\mathbf{A}$ ($m \times n$) and $\mathbf{B}$ ($n \times p$), the sketching matrix $\mathbf{S}$ is used.

Instead of computing the exact product $\mathbf{C} = \mathbf{A}\mathbf{B}$, which requires $O(mnp)$ operations, we compute an approximate product using a sketching matrix $\mathbf{S}$ of size $k \times n$ (where $k \ll n$). The approximation is constructed by compressing the columns of $\mathbf{A}$ and the rows of $\mathbf{B}$:

$$\tilde{\mathbf{C}} = (\mathbf{A} \mathbf{S}^\top) (\mathbf{S} \mathbf{B})$$

Where:
* $\mathbf{A} \mathbf{S}^\top$ is an $m \times k$ matrix (a compressed version of $\mathbf{A}$).
* $\mathbf{S} \mathbf{B}$ is a $k \times p$ matrix (a compressed version of $\mathbf{B}$).

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sketch_and_solve.png' | relative_url }}" alt="Sketching Matrix">

For the approximation $\tilde{\mathbf{C}}$ to be accurate, the sketching matrix $\mathbf{S}$ must satisfy specific properties:

* The sketching matrix, when squared, acts as the identity matrix in expectation. If $\mathbf{S}$ is drawn from an appropriate distribution, then $\mathbb{E}[\mathbf{S}^\top \mathbf{S}] = \mathbf{I}_n$. This ensures that the approximation is an unbiased estimator of the true product: $\mathbb{E}[\tilde{\mathbf{C}}] = \mathbb{E}[\mathbf{A} (\mathbf{S}^\top \mathbf{S}) \mathbf{B}] = \mathbf{A}\mathbf{B}$.

* A matrix $\mathbf{S}$ has the property that if, for any vector $\mathbf{x}$, the norm satisfies:
  $(1 - \epsilon)\|\mathbf{x}\|_2^2 \le \|\mathbf{S}\mathbf{x}\|_2^2 \le (1 + \epsilon)\|\mathbf{x}\|_2^2$.

* A sketching matrix $\mathbf{S}$ should also satisfy:
  $\| \mathbf{A}\mathbf{B} - (\mathbf{A}\mathbf{S}^\top)(\mathbf{S}\mathbf{B}) \|_F \leq \epsilon \| \mathbf{A} \|_F \| \mathbf{B} \|_F$.
  Here we are bounding the error ensuring accurate results.

#### Sketching and Sampling

The sketches can be of various types, e.g., Gaussian, Count Sketch, Hadamard, and Learned sketches.
* **Gaussian sketches** involve projecting data using a matrix where entries are sampled independently from a normal distribution $\mathcal{N}(0, \frac{1}{k})$ to preserve distances [Sobczyk and Luisier, 2022].
* **Count Sketch** offers a sparser alternative by using hash functions to map rows to buckets and randomly flipping their signs [Clarkson and Woodruff, 2017].
* **Hadamard-based sketches** (like the PHD matrix) combine randomized Hadamard transforms with uniform subsampling for structured efficiency [Clarkson and Woodruff, 2017].

For the broader problem of matrix multiplication, not just sketching techniques but sampling algorithms have also been studied which have shown good promise.

The core idea behind the sampling algorithms is to select a subset of the original data. Several algorithms like uniform sampling, random sampling, priority sampling, threshold sampling, and leverage score sampling are popular. They basically "look" and find the "most important" parts of the entire matrix and then perform the multiplication on the smaller subset of the matrices.

* **Uniform Sampling** samples each row with the probability $p=k/n$ of selection, effectively treating every single row as equally important [Cohen et al., 2015].
* **Leverage Score Sampling** selects rows based on their statistical influence called leverage scores which are either calculated using SVD or QR-decomposition [Drineas et al., 2012].
* **Priority and Threshold Sampling** is an innovative method which selects the rows based on their "importance," i.e., rows beyond a certain threshold are selected [Daliri et al., 2025].

### Sketch and Solve for Transformers

Transformers, as an architecture, is the industry standard because it has improved context as compared to previous architectures such as LSTM and RNNs, and provides the ability to parallelize the block which means faster training and inference.

Its architecture is split into two distinct blocks, both composed of a stack of $N$ identical layers:

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/transformers.png' | relative_url }}" alt="Transformers architecture">

**Encoder**

The Encoder’s primary function is to map an input sequence of symbols $(x_1, ..., x_n)$ into a rich, continuous representation $\mathbf{Z} = (z_1, ..., z_n)$.

Where each token is allowed to look at every other token available and hence it essentially understands the semantics.

**Decoder**
The Decoder generates the output sequence $(y_1, ..., y_m)$ one element at a time, being **auto-regressive** (consuming previously generated symbols as input for the next) instead of looking at the entire sentence like the encoder does.

The core logic which powers both encoder and decoder-based architectures is the attention mechanism; it basically creates a better, richer representation of our input context.

### Attention Mechanism

The input to the attention layer is our input embedding matrix $\mathbf{X}$. We project it using three learned weight matrices:
* **Query ($\mathbf{Q}$):** What the token is looking for.
* **Key ($\mathbf{K}$):** What the token identifies as.
* **Value ($\mathbf{V}$):** The actual content the token holds.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/mha.png' | relative_url }}" alt="Multihead Attention">

The attention score is calculated using:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

* First, we calculate the dot product between the Query matrix and the transpose of the Key matrix; in a way, for a token, it calculates the similarity between every single token and every other token in the sequence.
* These are then normalized by a factor of $\sqrt{d_k}$ and then we apply **Softmax** over it, which converts the raw scores into probabilities.
* The normalized weights are then multiplied with the matrix $\mathbf{V}$.

Calculating this basically tells us the relationship of one token in a sequence to all the other tokens. However, the operation $\mathbf{Q}\mathbf{K}^\top$ is a very heavy operation.
To compute this, we have to perform operations proportional to the square of the sequence length ($N^2$) (assuming the length of our sequence to be $N$).

More specifically,
* If $N=512$, the attention matrix has roughly 260,000 entries.
* If $N=4096$, it jumps to over 16 million entries.

It is quite evident that improvement with $\mathbf{Q}\mathbf{K}^\top$ can lead to improvement in the overall speed of the transformers.

### Better Transformers

In regards to optimizing the attention matrix, a lot of work has already been done.
Instead of computing the full matrix, architectures like the **Sparse Transformer** enforce a fixed sparsity pattern, such as a sliding window where tokens only attend to their immediate neighbors [Child et al., 2019]. This reduces the complexity to approximately $O(N\sqrt{N})$.
A more dynamic approach was introduced by the **Reformer**, which replaces exact dot-product attention with Locality-Sensitive Hashing (LSH) [Kitaev et al., 2020]. By grouping similar query and key vectors into the same hash buckets, the model computes attention only within these buckets, effectively dropping the complexity to $O(N \log N)$.

The **Linformer** relies on the low-rank property of the attention matrix [Wang et al., 2020]. Linformer projects the Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices into a lower-dimensional space using learned linear projections ($\mathbf{E}$ and $\mathbf{F}$). By compressing the original $(N \times d)$ matrices into much smaller $(k \times d)$ matrices, the attention operation becomes linear $O(N)$ in time and space.

**LevAttention** utilizes the concept of leverage scores (generalized $f$-sensitivities) to identify a small "universal set" of keys that dominate the attention scores for *any* query [Kannan et al., 2024]. This method proves that a subset of high-leverage keys, independent of the sequence length, captures the vast majority of the attention mass, allowing for efficient $O(N \cdot \text{poly}(d/\epsilon))$ computation.

**Matrix Product Sketching via Coordinated Sampling** proposes estimating the product $\mathbf{Q}\mathbf{K}^\top$ directly using coordinated random sampling [Daliri et al., 2025]. Unlike traditional linear sketching (such as Johnson-Lindenstrauss projections) which can be inefficient for sparse data, coordinated sampling (specifically Priority Sampling) selects rows from $\mathbf{Q}$ and $\mathbf{K}$ based on their norms using a shared random seed.

In one way or another, all of these methods are trying to make the models faster by leveraging properties of the matrix itself or the matrix multiplication. With that in mind, we wanted to experiment with several different types of attention mechanisms for low resource Indic languages and see how they perform across various tasks in terms of accuracy and what does the distribution of the attention matrices really look like for different modifications and hence we have conducted an exhaustice set of experiments;

### Experiment Setup

For the course of our experiments, we have picked the **DistilBERT** (6 encoder blocks, 12 heads each) architecture which is a distilled version of **BERT** that retains 97% of its performance while being 40% lighter.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/distbert.drawio (1).png' | relative_url }}" alt="DistilBert">

We have used DistilBert-Multilingual-Cased model available on hugging face (https://huggingface.co/distilbert/distilbert-base-multilingual-cased) trained on wikipedia dataset as our base model for all the tasks

We then fine-tuned this model for various different tasks using our datasets and different attention mechanisms such as:

* priority sampling
* Leverage score based sampling
* Lewis score based sampling
* Learned sketches
* vanilla fine tuning



The datasets were taken from the  **IndicGLUE Benchmark** (Hindi Language subsets), courtesy of [AI4Bharat](https://indicnlp.ai4bharat.org/pages/indic-glue/).

| Task & Dataset | Description | Unique Labels | Hindi Example |
| :--- | :--- | :--- | :--- |
| **Sentiment Analysis**<br>`iitp-mr.hi` | Classifies movie reviews as positive, negative, or neutral. | `0` (Negative)<br>`1` (Neutral)<br>`2` (Positive) | *“यह फिल्म देखने लायक है, कहानी बहुत अच्छी है।”* |
| **News Classification**<br>`bbca.hi` | Classifies news articles into 14 distinct topics (e.g., Sports, India). | 14 Topics<br>(`india`, `sport`, `entertainment`, etc.) | *“भारतीय क्रिकेट टीम ने आज ऐतिहासिक जीत दर्ज की।”* |
| **Discourse Mode**<br>`md.hi` | Identifies the rhetorical role of a sentence in a narrative. | `Argumentative`<br>`Descriptive`<br>`Dialogic`<br>`Informative`<br>`Narrative` | *“एक बार की बात है, एक घना जंगल था।”*<br>(Narrative) |
| **Causal Reasoning (COPA)**<br>`copa.hi` | Selects the most plausible cause or effect for a given premise. | `0` (Choice 1)<br>`1` (Choice 2) | **Premise:** *“लड़के का पैर फिसल गया।”*<br>**Correct:** *“वह गिर गया।”* |
| **Named Entity Recognition**<br>`wiki-ner.hi` | Tags entities like Persons, Locations, and Organizations in text. | `O`, `B-PER`, `I-PER`<br>`B-ORG`, `I-ORG`<br>`B-LOC`, `I-LOC` | *“**राहुल** (B-PER) **गांधी** (I-PER) **दिल्ली** (B-LOC) में हैं।”* |
| **Section Title Prediction**<br>`wstp.hi` | Predicts the correct section title for a Wikipedia paragraph. | `0` (Title A)<br>`1` (Title B)<br>`2` (Title C)<br>`3` (Title D) | **Text:** (Paragraph about Cricket rules)<br>**Correct:** *“नियम” (Rules)* |

All of the datasets were split into training and testing set and had a sequence lenght of 512

Naturally, we wanted our matrices to be a good representation of our entire corpus and hence we kept the values of k to be in the ranges: [64,128,256]

So for each of the dataset mentioned below, we have finetuned about 5 different models 

All the models have been fine-tuned on Kaggle using the GPU P100 and are available for inference freely on Hugging Face.

### Modified attention 

 The goal here is not to come up with a one-size-fits-all solution but rather to find out how each of these perform in practice and the pitfalls (if any) for several of these algorithms.



####  Learned Sketch

Adapting the idea from linformer, the assumption here is that the attention matrix is low rank. Hence, instead of computing the entire $N \times N$ matrix, we take smaller versions of the projected Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices. [Wang et al., 2020].

Those smaller projection matrices are defined as $\mathbf{E} \in \mathbb{R}^{k \times N}$ and $\mathbf{F} \in \mathbb{R}^{k \times N}$, where $k \ll N$ such that

$$\mathbf{K}_{proj} = \mathbf{E}\mathbf{K}, \quad \mathbf{V}_{proj} = \mathbf{F}\mathbf{V}$$

```latex
Input: Q (N x d), K (N x d), V (N x d)

Compressed matrices
   K_proj = E * K    (k x d)
   V_proj = F * V    (k x d)

Attention calculation
   Scores = (Q * K_proj^T) / sqrt(d)    (N x k)

weights
   Weights = Softmax(Scores)

Final product
   Output = Weights * V_proj   (N x d)
```



**Leverage Scores based attention**

This architecture is adopted from LevAttention, where Leverage scores are used to identify the most influential keys (rows of K), and we restrict attention computation to only these selected keys.
Instead of applying this to vision-based tasks as in the original paper, we adapt this approach for natural language tasks.

**Statistical Leverage Scores** are used here to identify the most "influential" keys in the sequence. High leverage indicates a key that is unique or critical to the subspace [Kannan et al., 2024].

For a matrix $\mathbf{K} \in \mathbb{R}^{n \times d}$ with $n > d$ and full column rank, the statistical leverage score of the $i$-th row $\mathbf{k}_i$ is formally defined as:

$$\tau_i(\mathbf{K}) = \mathbf{k}_i^\top (\mathbf{K}^\top \mathbf{K})^{-1} \mathbf{k}_i$$

This corresponds to the $i$-th diagonal element of the projection matrix $\mathbf{P} = \mathbf{K}(\mathbf{K}^\top \mathbf{K})^{-1}\mathbf{K}^\top$, which projects onto the column space of $\mathbf{K}$ [Drineas et al., 2012; Mahoney et al., 2011]. The leverage scores satisfy $0 \leq \tau_i \leq 1$ and $\sum_{i=1}^n \tau_i = d$, providing a natural probability distribution over the rows of $\mathbf{K}$.

For a matrix $\mathbf{K} \in \mathbb{R}^{N \times d}$, the leverage score $\tau_i$ of the $i$-th row $\mathbf{k}_i$ is defined as:
$$\tau_i(\mathbf{K}) = \mathbf{k}_i (\mathbf{K}^\top \mathbf{K})^{-1} \mathbf{k}_i^\top$$



```latex
K (N x d), Q (N x d), V (N x d), budget k, damping (for stability) lambda

Leverage Scores for K:
   G = K^T * K + lambda * I
   G_inv = inverse(G)
   For each row i in K:
       tau_i = K_i * G_inv * K_i^T

Universal Set U calculation:
   U = {indices of the top-k largest tau_i}

Attention Mask M:
   M = matrix of -infinity (N x N)
   For all query positions i and key positions j:
       if j in U: M[i,j] = 0
       else: M[i,j] = -infinity

Attention:
   attention_scores = (Q * K^T) / sqrt(d) + M
   attention_weights = softmax(attention_scores)
   output = attention_weights * V
```

#### L1 weights based attention

We have also tried to use Lewis weights for the key selection instead of Leverage scores for the attention mechanism.

Lewis Weights are a generalization of leverage scores for $\ell_1$ norms. They provide a sensitivity measure for the $\ell_1$ norm, ensuring that the selected keys capture the geometry of the data distribution more robustly [Cohen & Peng, 2015].

For a matrix $\mathbf{K} \in \mathbb{R}^{n \times d}$, the $\ell_1$ Lewis weights $w_i$ are defined as the unique values satisfying:
$$w_i = \tau_i(\mathbf{W}^{-1/2}\mathbf{K})$$
where $\mathbf{W} = \text{diag}(w_1, \ldots, w_n)$ is the diagonal weight matrix, and $\tau_i(\cdot)$ denotes the standard $\ell_2$ leverage score of the re-weighted matrix. 

We are using
``` latex
Input: K (N x d), Q (N x d), V (N x d)

Compute L1 Lewis Weights for K:
   Initialize: w_i = 1 for all i
   Iterate T times:
      K_tilde = K / sqrt(w)
      tau = LeverageScores(K_tilde)  
      w = sqrt(w * tau)
   S = {indices of largest w_i}

 Create Mask for Attention:
   M[i,j] = 0 if j in S else -inf

Compute Attention with Mask:
   attention_scores = (Q * K^T) / sqrt(d) + M
   attention_weights = softmax(attention_scores)
   output = attention_weights * V

```


#### Priority Sampling 

Here we are using adapting the priority sampling algorithm for distilbert.

We select the key vectors probabilistically, where the probability of selecting a key is proportional to its squared norm, using a threshold derived from the squared Frobenius norm of the key matrix [Daliri et al., 2025].

More formally,

Let \(\mathbf{ K} \in \mathbb{R}^{k\_len \times d} \) be the key matrix for a given head.

 The squared Frobenius norm of the key matrix is given by:
    \[
    \mathbf{A_{\text{norm}}^2 }= \|\mathbf{K}\|_F^2 = \sum_{i=1}^{k\_len} \|\mathbf{k}_i\|_2^2
    \]

For a target sample size \( k \), the threshold \( \tau \) is:
    \[
    \tau = \frac{k}{\mathbf{A_{\text{norm}}^2}}
    \]

 For each key vector \( \mathbf{k}_i \), generate a random hash \( h_i \sim \text{Uniform}(0, 1) \). The key is selected if:
    \[
    h_i \leq \tau \cdot \|\mathbf{k}_i\|_2^2
    \]

```latex
Input: K (N x d), k, random seed s

Generate Shared Randomness:
   h_i ~ Uniform(0, 1) for i in 1..N 

Compute Ranks :
   r_i = h_i / ||K_i||^2   // Lower rank = Higher priority

Determine Threshold:
   tau = (k+1)-th smallest value in r

Select Keys:
   S = {i | r_i <= tau}

Attention:
   Output = Softmax((Q * K_S^T) / sqrt(d)) * V_S
```


### Results

Looking at the experiments and the attention mechanism, it is clear that there is some speedup while maintaining accuracy. However, interesting thing to note is that the benifits offered by the modifications are not uniform across all the tasks suggesting either the need for optimizing the code further or looking for alternative ways to make the methods work for more complex tasks


###### Sentiment Analysis

A good idea to see how is our model predicting is using saliency plots
<div class="layout-seq">
<div><img src="assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_vanilla/inference_vanilla/prediction_dashboard_Instance_Analysis.png"> <p>Vanilla Attention</p></div>
<div><img src="assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_priority/inference_priority_sampling/prediction_dashboard_Instance_Analysis.png"> <p>Priority Sampling</p></div>
<div><img src="assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_levattention/inference_levattention/prediction_dashboard_Instance_Analysis.png"> <p>Levattention</p></div>
<div><img src="assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_learnedsketch/inference_learned_sketch/prediction_dashboard_Instance_Analysis.png"> <p>Learned Sketch</p></div>
<div><img src="assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_l1_attention/inference_l1/prediction_dashboard_Instance_Analysis.png"> <p>L1 attention</p></div>
</div>

Let us also check the data distribution for the attention mechanism
<div class="layout-seq">
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla Attention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

###### NER

###### WSTP

###### NLI

###### Discourse

##### Comparing accuracies across tasks 
<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/attention_accuracy.png' | relative_url }}" alt="Accuracy">

##### Comparing the inference across tasks
<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/corrected_time_diff_heatmap.png' | relative_url }}" alt="Time Spent">

