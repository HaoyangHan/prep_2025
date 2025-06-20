# Mastering Positional Encodings in Transformers: An Interview Guide

> This guide provides a comprehensive overview of Positional Encoding (PE) techniques used in Transformer models, a critical topic in modern NLP and Large Language Model interviews. It begins with foundational concepts, such as why Transformers require explicit position information, and progresses through various families of PE: Absolute, Relative, and advanced hybrid methods like RoPE. We delve into the mathematical underpinnings, practical implementations, and the crucial challenge of sequence length extrapolation, equipping you with the deep knowledge required to excel in theoretical discussions and coding challenges.

## Knowledge Section

### 1. The Fundamental Need for Positional Encoding

Unlike Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs), which process tokens sequentially or through local filters, the core self-attention mechanism of a Transformer is **permutation-invariant**. This means that if you shuffle the input tokens, the self-attention output (without positional encoding) for each token would be an identically shuffled version of the original outputs. The model has no inherent sense of word order or sequence.

For example, the sentences "Man bites dog" and "Dog bites man" would be indistinguishable to a pure self-attention layer. To remedy this, we must inject information about the position of each token into the model. This is the sole purpose of Positional Encoding.

There are two primary strategies for this:
1.  **Absolute Positional Encoding:** Integrate position information directly into the input embeddings. The encoding for a token at position `k` depends only on `k`.
2.  **Relative Positional Encoding:** Modify the attention mechanism itself to be aware of the relative distance between any two tokens when calculating attention scores.

---

### 2. Absolute Positional Encodings

Absolute PE adds a position-specific vector, $p_k$, to each input token embedding, $x_k$. The model's input becomes $x_k' = x_k + p_k$.

#### 2.1. Trainable Positional Encoding

This is the simplest and most direct approach, famously used by BERT.

*   **How it works:** A lookup table (an embedding matrix) of size $(L_{max}, d_{model})$ is created, where $L_{max}$ is the maximum sequence length the model can handle (e.g., 512 for BERT-Base) and $d_{model}$ is the embedding dimension. This matrix is initialized randomly and then learned jointly with the rest of the model parameters during training. For a token at position `k`, we simply look up the `k`-th row of this matrix and add it to the token embedding.

*   **Pros:**
    *   Extremely simple to implement.
    *   Flexible; the model can learn the optimal encoding for the task and data distribution.

*   **Cons:**
    *   **Poor Extrapolation:** The primary drawback. The model has never seen positions beyond $L_{max}$ during pre-training. If asked to process a sequence of length $L_{max} + 1$, it has no learned embedding for that position. While one could randomly initialize it and fine-tune, performance typically degrades significantly.

#### 2.2. Sinusoidal (Trigonometric) Positional Encoding

Proposed in the original "Attention Is All You Need" paper, this is a fixed, non-learned method based on sine and cosine functions of different frequencies.

*   **How it works:** For a token at position $k$ and dimension $i$ in the $d$-dimensional embedding vector, the PE is defined as:

$$
PE(k, i) = \begin{cases} \sin(k / 10000^{2i/d}) & \text{if } i \text{ is even} \\ \cos(k / 10000^{(2i-1)/d}) & \text{if } i \text{ is odd} \end{cases}
$$

This can be written more compactly for pairs of dimensions $(2i, 2i+1)$:

$$
\begin{align*}
p_{k, 2i} &= \sin\left(\frac{k}{10000^{2i/d}}\right) \\
p_{k, 2i+1} &= \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{align*}
$$

*   **Why it works (The Relative Position Property):** The key advantage of this formulation is that the positional encoding for $p_{k+m}$ can be expressed as a linear transformation of $p_k$. This property arises from the trigonometric sum identities:
    *   $\sin(A+B) = \sin A \cos B + \cos A \sin B$
    *   $\cos(A+B) = \cos A \cos B - \sin A \sin B$

    This means that for any fixed offset $m$, the model can learn a linear transformation to map the encoding of position $k$ to the encoding of position $k+m$, effectively allowing it to learn relative position information.

*   **Pros:**
    *   No parameters to learn.
    *   Deterministic and computationally efficient.
    *   Theoretically allows for extrapolation to unseen sequence lengths, as the functions are defined for any $k$.

*   **Cons:**
    *   In practice, the extrapolation performance is still limited. As $k$ becomes very large, the periodicity of the functions can cause ambiguity, and models trained on shorter sequences may not generalize well to the different value ranges encountered at large $k$.

#### 2.3. Other Absolute PE Variants

*   **Recursive Encoding (e.g., FLOATER):** Instead of a fixed formula, one can generate position vectors recursively, e.g., $p_{k+1} = f(p_k)$, where $f$ is a small neural network. This mimics how RNNs handle position. The extreme version models this as a continuous Ordinary Differential Equation (ODE), $dp_t/dt=h(p_t,t)$. This offers good extrapolation but sacrifices the parallelism of Transformers by introducing a sequential dependency.
*   **Multiplicative Encoding:** Some research suggests that combining token and position embeddings via multiplication ($x_k \odot p_k$) might yield better results than addition ($x_k + p_k$), potentially creating a more dynamic interaction.

---

### 3. Relative Positional Encodings (RPE)

Instead of modifying the input, RPEs modify the self-attention calculation to directly incorporate the relative distance between the *query* token (at position $i$) and the *key* token (at position $j$).

The standard attention score is $a_{i,j} = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}}$. RPE methods add terms to this calculation that depend on the relative position $i-j$.

This idea stems from expanding the attention score for absolute PEs:
$$
\text{Score}(i,j) = (x_i W_Q + p_i W_Q)(x_j W_K + p_j W_K)^T \\
= \underbrace{x_i W_Q W_K^T x_j^T}_{\text{(a) content-content}} + \underbrace{x_i W_Q W_K^T p_j^T}_{\text{(b) content-position}} + \underbrace{p_i W_Q W_K^T x_j^T}_{\text{(c) position-content}} + \underbrace{p_i W_Q W_K^T p_j^T}_{\text{(d) position-position}}
$$
Different RPE methods can be seen as selectively keeping, removing, or reformulating these four terms.

#### 3.1. Classic RPE (Shaw et al., 2018)

*   **How it works:** This method introduces learnable relative position embeddings. The attention score and value vector calculations are modified:
    $$
    a_{i, j} = \frac{(x_i W_Q)(x_j W_K + \mathbf{r}_{i-j}^K)^T}{\sqrt{d_k}}
    $$
    $$
    o_i = \sum_{j} \text{softmax}(a_{i,j}) (x_j W_V + \mathbf{r}_{i-j}^V)
    $$
    Here, $\mathbf{r}_{i-j}^K$ and $\mathbf{r}_{i-j}^V$ are learnable embedding vectors corresponding to the relative distance $i-j$. To avoid having an infinite number of embeddings, the distance is often clipped to a certain range $[-L_{clip}, L_{clip}]$. Any distance outside this range maps to the boundary embeddings $\mathbf{r}_{-L_{clip}}$ or $\mathbf{r}_{L_{clip}}$.

#### 3.2. Transformer-XL / XLNet-style RPE

This is a more sophisticated approach that was critical to the success of Transformer-XL and XLNet.

*   **How it works:** It takes the full expansion of the attention score and makes several key changes:
    1.  Replace the absolute position vector $p_j$ in term (b) with a sinusoidal relative position vector $\mathbf{R}_{i-j}$.
    2.  Replace the query's absolute position vector $p_i$ in terms (c) and (d) with two separate, learnable vectors, $\mathbf{u}$ and $\mathbf{v}$. This is because the query's position information should be independent of the content of other tokens.
    3.  A separate projection matrix, $W_{K,R}$, is used for the relative position embeddings.
    4.  The bias term in the value calculation is dropped.

    The final score becomes a sum of four components:
    $$
    \text{Score}(i,j) = \underbrace{\mathbf{q}_i^T \mathbf{k}_j}_{\text{content-content}} + \underbrace{\mathbf{q}_i^T W_{K,R} \mathbf{R}_{i-j}}_{\text{content-relative pos}} + \underbrace{\mathbf{u}^T \mathbf{k}_j}_{\text{global content bias}} + \underbrace{\mathbf{v}^T W_{K,R} \mathbf{R}_{i-j}}_{\text{global position bias}}
    $$
    where $\mathbf{q}_i=x_iW_Q$ and $\mathbf{k}_j=x_jW_K$.

#### 3.3. T5-style RPE

T5 simplifies the relative position idea significantly.

*   **How it works:** It argues that relative position information can be captured by a simple scalar bias added directly to the pre-softmax attention logits.
    $$
    a_{i,j} = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}} + \beta_{i-j}
    $$
    The term $\beta_{i-j}$ is a learnable scalar that depends on the relative position. Instead of a unique parameter for each possible distance, T5 uses **bucketing**:
    *   It maintains a small, fixed number of learnable bias parameters (e.g., 32).
    *   Close relative positions (e.g., 0 to 7) might each get their own unique bucket (and thus a unique $\beta$).
    *   As the distance increases, multiple distances are grouped (or "bucketed") together to share the same $\beta$ parameter. This is based on the intuition that precise position is crucial for nearby tokens but less so for distant ones.

#### 3.4. DeBERTa's Disentangled Attention

DeBERTa proposes another variation, arguing that the content embedding and position embedding should be "disentangled".

*   **How it works:** It keeps the content-position (b) and position-content (c) terms from the full expansion but drops the position-position (d) term.
    $$
    \text{Score}(i,j) = (x_i W_Q)(x_j W_K)^T + (x_i W_Q)(\mathbf{R}_{i-j})^T + (\mathbf{R}_{j-i})(x_j W_K)^T
    $$
    Here, $\mathbf{R}$ represents the relative position embeddings. The key insight is to use the relative position from the key's perspective ($i-j$) when interacting with the query's content, and the relative position from the query's perspective ($j-i$) when interacting with the key's content.

---

### 4. Advanced & Hybrid Encodings

#### 4.1. Rotary Position Embedding (RoPE)

RoPE, introduced by EleutherAI and popularized by models like LLaMA and PaLM, is a novel method that achieves relative positioning through an absolute formulation. It is currently one of the most effective and widely used techniques.

*   **Core Idea:** Instead of adding positional information, RoPE **rotates** the query and key vectors based on their absolute position. The inner product between a rotated query (at position $m$) and a rotated key (at position $n$) will naturally depend only on their relative position ($m-n$).

*   **Mathematical Formulation:**
    1.  Start with the goal: Find a function $f(\mathbf{x}, p)$ such that the inner product $\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle$ depends only on the content vectors $\mathbf{q}, \mathbf{k}$ and the relative position $m-n$.
    2.  Working in the complex plane for simplicity, this condition becomes $f(\mathbf{q}, m) f^*(\mathbf{k}, n) = g(\mathbf{q}, \mathbf{k}, m-n)$.
    3.  A solution to this functional equation is $f(\mathbf{x}, p) = \mathbf{x} e^{ip\theta}$. The inner product then becomes $\operatorname{Re}[(\mathbf{q}e^{im\theta})(\mathbf{k}e^{in\theta})^*] = \operatorname{Re}[\mathbf{q}\mathbf{k}^* e^{i(m-n)\theta}]$, which depends on $m-n$.
    4.  To implement this with real vectors, we group dimensions into pairs $(x_0, x_1), (x_2, x_3), \dots$. For each pair, applying the rotation $m\theta_i$ is equivalent to multiplying by a 2D rotation matrix:
        $$
        \mathbf{R}_{m, \theta_i} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
        $$
        So, the rotated vector for a pair $(q_{2i}, q_{2i+1})$ is:
        $$
        \begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
        $$
    5.  The "frequency" $\theta_i$ is chosen similarly to sinusoidal PE: $\theta_i = 10000^{-2i/d}$. This gives high-frequency rotations for early dimensions (capturing fine-grained local order) and low-frequency rotations for later dimensions (capturing coarse-grained global order).

*   **Pros:**
    *   Excellent performance, becoming the standard for many state-of-the-art LLMs.
    *   Achieves relative encoding properties while operating on absolute positions, which is elegant.
    *   The norm of the vectors is preserved after rotation.

*   **Cons:**
    *   Still suffers from extrapolation issues, which led to the development of interpolation techniques.

#### 4.2. Attention with Linear Biases (ALiBi)

ALiBi is another technique designed specifically for superior length extrapolation. It's used in models like BLOOM and MPT.

*   **How it works:** ALiBi discards positional embeddings entirely. Instead, it adds a static, non-learned bias to the attention scores before the softmax operation.
    $$
    a_{i,j} = \frac{(x_i W_Q)(x_j W_K)^T}{\sqrt{d_k}} + m \cdot (j-i)
    $$
    This is applied to query $i$ attending to key $j$, where $j \le i$ (for causal masking).
    *   The bias is a linear penalty proportional to the distance between the query and key.
    *   The slope $m$ is a fixed, pre-determined value unique to each attention head. For an 8-head model, the slopes might be $1/2^1, 1/2^2, \dots, 1/2^8$. This gives each head a different "attentional span." Heads with smaller slopes can attend further back, while heads with larger slopes are heavily penalized for distance and focus locally.

*   **Pros:**
    *   **Exceptional Extrapolation:** Because the bias is a simple linear function, the model can generalize to distances it has never seen during training, resulting in much less performance degradation on long sequences.
    *   Simple and requires no extra parameters.

*   **Cons:**
    *   Performance within the pre-training length might be slightly worse than RoPE in some benchmarks, as it's a less expressive form of position information.

---

### 5. The Length Extrapolation Problem

This is a critical issue for LLMs: how to handle inputs longer than the context window they were trained on (e.g., train on 2048 tokens, test on 8192 tokens).

**Causes:**
1.  **Out-of-Distribution (OOD) Positions:** Positional encoding methods (learnable, sinusoidal, RoPE) may not generalize well to positions far beyond the training range.
2.  **Attention Distribution Shift:** As sequence length grows, the softmax denominator increases, causing attention scores to become more diffuse or "diluted." The model, trained on shorter sequences, may not be robust to this shift.

**Solutions:**
1.  **ALiBi:** As discussed, this is a built-in solution that works well out-of-the-box.
2.  **Position Interpolation (for RoPE):** This is the most common fix for RoPE-based models. Instead of extrapolating, we *interpolate*. If a model was trained on length $L_{train}$ and we need to handle length $L_{new}$, we rescale the position indices. The position $k$ is mapped to $k' = k \cdot (L_{train} / L_{new})$. We then compute RoPE using this down-scaled position $k'$. This "squeezes" the new, larger position space into the original, smaller one the model is familiar with. This usually requires a small amount of fine-tuning to adapt to the "denser" position space.
3.  **NTK-Aware Scaled RoPE:** A more advanced interpolation technique. It observes that simple linear interpolation disproportionately compresses high-frequency (local) dimensions. The NTK-aware method scales the RoPE frequencies ($\theta_i$) by a factor related to the length increase, effectively changing the "base" of the positional encoding to accommodate a larger range of numbers without requiring fine-tuning.

---

## Interview Questions

### Theoretical Questions

**Q1: Why do Transformer models require an explicit Positional Encoding, while models like LSTMs or GRUs do not?**
**A1:** The self-attention mechanism at the core of the Transformer is permutation-invariant. It processes all input tokens in parallel as a set, with no inherent knowledge of their order. The attention score between token $i$ and token $j$ is calculated based only on their vector representations, not their positions. Therefore, without an external signal, "dog bites man" and "man bites dog" are computationally identical.
In contrast, LSTMs and GRUs are recurrent models. They process input sequentially, one token at a time. The hidden state at time step $t$ is a function of the input at $t$ and the hidden state from $t-1$. This sequential dependency inherently encodes the order of the tokens, making a separate positional encoding mechanism unnecessary.

---
**Q2: Explain the key insight behind Sinusoidal Positional Encoding and prove mathematically how it enables the model to learn relative positions.**
**A2:** The key insight is that for any fixed relative offset, the positional encoding of a future position can be represented as a fixed linear transformation of the current position's encoding. This allows the attention mechanism, which is fundamentally based on linear transformations (queries, keys, values), to easily learn to attend based on relative offsets.

**Proof:**
Let $\mathbf{p}_k \in \mathbb{R}^d$ be the PE vector for position $k$. Its components are defined as:
$p_{k, 2i} = \sin(\omega_i k)$ and $p_{k, 2i+1} = \cos(\omega_i k)$, where $\omega_i = 1/10000^{2i/d}$.

Consider the position $k+m$. The corresponding PE components are:
$p_{k+m, 2i} = \sin(\omega_i (k+m))$
$p_{k+m, 2i+1} = \cos(\omega_i (k+m))$

Using the angle sum identities:
$\sin(A+B) = \sin A \cos B + \cos A \sin B$
$\cos(A+B) = \cos A \cos B - \sin A \sin B$

We can expand the terms for position $k+m$:
$p_{k+m, 2i} = \sin(\omega_i k)\cos(\omega_i m) + \cos(\omega_i k)\sin(\omega_i m) = p_{k, 2i}\cos(\omega_i m) + p_{k, 2i+1}\sin(\omega_i m)$
$p_{k+m, 2i+1} = \cos(\omega_i k)\cos(\omega_i m) - \sin(\omega_i k)\sin(\omega_i m) = p_{k, 2i+1}\cos(\omega_i m) - p_{k, 2i}\sin(\omega_i m)$

This can be written as a matrix multiplication for each pair of dimensions:
$$
\begin{pmatrix} p_{k+m, 2i} \\ p_{k+m, 2i+1} \end{pmatrix} =
\begin{pmatrix}
\cos(\omega_i m) & \sin(\omega_i m) \\
-\sin(\omega_i m) & \cos(\omega_i m)
\end{pmatrix}
\begin{pmatrix} p_{k, 2i} \\ p_{k, 2i+1} \end{pmatrix}
$$
This shows that the vector for position $k+m$ is a linear transformation (specifically, a rotation) of the vector for position $k$. Crucially, the transformation matrix depends only on the relative offset $m$, not the absolute position $k$. A neural network can easily learn such a linear relationship.

---
**Q3: Compare and contrast RoPE and ALiBi, especially in the context of sequence length extrapolation.**
**A3:** RoPE and ALiBi are two state-of-the-art methods for handling position information, but they take fundamentally different approaches.

*   **Mechanism:**
    *   **RoPE (Rotary Position Embedding):** Applies position-dependent *rotations* to the query and key vectors before the dot product. It's an absolute encoding (the rotation depends on absolute position $m$) that cleverly results in relative attention scores (the dot product depends on relative position $m-n$).
    *   **ALiBi (Attention with Linear Biases):** Does not modify the input vectors at all. It adds a static, non-learned *bias* directly to the attention logits. The bias is a linear penalty proportional to the distance between the query and key.

*   **Length Extrapolation:**
    *   **RoPE:** Suffers from extrapolation issues. While theoretically defined for all positions, models trained with RoPE on a fixed length (e.g., 2048) see a specific range of rotation angles. At much longer lengths, the unseen, larger angles become out-of-distribution, leading to performance degradation. This is typically solved with fine-tuning using **Position Interpolation**, which scales down the new positions to fit into the seen range.
    *   **ALiBi:** Was designed specifically for extrapolation and excels at it. Because the bias is a simple, monotonic linear penalty, the model learns a general heuristic: "tokens that are further away are less important." This simple rule generalizes seamlessly to distances it has never seen during training, resulting in much more robust extrapolation performance without any fine-tuning.

*   **Performance:**
    *   Within the pre-training context length, **RoPE** often shows slightly better performance on perplexity-based benchmarks, likely because its rotational mechanism is more expressive.
    *   Beyond the pre-training length, **ALiBi** is typically superior out-of-the-box.

*   **In Summary:** RoPE is a high-performance, expressive encoding that is the de-facto standard in many top models (LLaMA, PaLM), but requires a specific strategy (interpolation) to handle longer contexts. ALiBi is a simpler, more robust alternative that sacrifices a small amount of in-context performance for excellent, zero-shot extrapolation capability.

---
**Q4: How do the T5 and DeBERTa relative position schemes differ in their modification of the attention score?**
**A4:** Both T5 and DeBERTa modify the attention score to include relative position information, but they do so by manipulating different parts of the conceptual attention score expansion.

The full expansion is:
`Score = content-content + content-position + position-content + position-position`

*   **T5's Approach (Simple Bias):**
    *   T5 simplifies this dramatically. It assumes the most important terms are the `content-content` interaction and a general, position-based bias.
    *   It effectively discards the `content-position` and `position-content` interaction terms and replaces the `position-position` term with a single, learnable scalar bias, $\beta_{i-j}$.
    *   **Formula:** `Score(i,j) = (query_i * key_j) + β_{i-j}`
    *   This is extremely efficient. The bias is learned and uses "bucketing" to group distant positions, saving parameters.

*   **DeBERTa's Approach (Disentangled Attention):**
    *   DeBERTa takes the opposite approach. It argues that the interactions between content and position are crucial, but the interaction between pure positions is less important.
    *   It keeps the `content-content`, `content-position`, and `position-content` terms, but completely drops the `position-position` term.
    *   **Formula:** `Score(i,j) = (content_query_i * content_key_j) + (content_query_i * pos_key_{i-j}) + (pos_query_{j-i} * content_key_j)`
    *   The "disentangled" part refers to using separate embeddings for content and relative position and carefully defining their interactions.

*   **Key Difference:** T5 boils relative position down to a simple, symmetric bias term. DeBERTa models a more complex, asymmetric interaction where a token's content is separately compared against the other token's content and its relative position.

---
**Q5: What is the "attention dilution" problem in length extrapolation, and how does it relate to positional encoding?**
**A5:**
The **attention dilution** or **attention entropy** problem refers to the phenomenon where, as the sequence length increases, the softmax function distributes the attention probability mass over a larger number of tokens. This can make the distribution "flatter" or more diffuse, meaning each individual token receives a smaller attention score on average.

A model trained on shorter sequences (e.g., 2048 tokens) learns to operate with a certain level of attention entropy. When it encounters a much longer sequence at inference time (e.g., 8192 tokens), the attention scores become significantly more diluted. The model may not be robust to this statistical shift, causing it to lose focus and perform poorly.

This problem is intertwined with, but distinct from, the positional encoding OOD problem. Even with a perfect PE scheme like ALiBi that extrapolates well, the model still has to contend with the change in the attention score distribution. Some research has proposed techniques like adding a `log(N)` term to the attention logits to counteract this dilution effect, helping the model maintain focus at longer sequence lengths.

### Practical & Coding Questions

**Q1: Implement the Sinusoidal Positional Encoding from "Attention Is All You Need" from scratch using PyTorch. Then, visualize the result.**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_sinusoidal_positional_encoding(max_len: int, d_model: int):
    """
    Generates sinusoidal positional encoding.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): The dimension of the model embeddings.

    Returns:
        torch.Tensor: A tensor of shape (max_len, d_model) with positional encodings.
    """
    # Create a matrix to hold the positional encodings
    pe = torch.zeros(max_len, d_model)

    # Create a vector for positions [0, 1, ..., max_len-1]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: [max_len, 1]

    # Create the divisor term for the frequencies.
    # The term is 10000^(2i/d_model)
    # The dimensions are i = [0, 1, ..., d_model/2 - 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) # shape: [d_model/2]

    # Apply sin to even indices in the array; 2i
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd indices in the array; 2i+1
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Add a batch dimension if you want to add it directly to batched inputs
    # pe = pe.unsqueeze(0) # shape: [1, max_len, d_model]

    return pe

# --- Example Usage and Visualization ---
max_len = 100
d_model = 512

pe_matrix = get_sinusoidal_positional_encoding(max_len, d_model)

plt.figure(figsize=(15, 5))
plt.imshow(pe_matrix.numpy(), cmap='viridis', aspect='auto')
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.show()

print("Shape of the PE matrix:", pe_matrix.shape)
```

**Q2: Implement the core logic of Rotary Position Embedding (RoPE) in PyTorch.**

```python
import torch

def get_rope_embeddings(d_model: int, max_len: int):
    """
    Generates the rotary frequency embeddings (cos and sin terms).
    This is pre-computed.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (cos, sin) tensors,
                                           each of shape (max_len, d_model/2)
    """
    # Create the theta term: 10000^(-2i/d)
    theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    
    # Create the position sequence: [0, 1, ..., max_len-1]
    seq_idx = torch.arange(max_len).float()
    
    # Create the m*theta matrix
    idx_theta = torch.outer(seq_idx, theta) # shape: [max_len, d_model/2]
    
    # The final embeddings are cos(m*theta) and sin(m*theta)
    freqs_cos = torch.cos(idx_theta)
    freqs_sin = torch.sin(idx_theta)
    
    return freqs_cos, freqs_sin
    

def apply_rotary_pos_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    """
    Applies RoPE to a query or key tensor.

    Args:
        x (torch.Tensor): Input tensor (query or key) of shape (batch, seq_len, heads, head_dim).
        freqs_cos (torch.Tensor): Pre-computed cosine values of shape (seq_len, head_dim/2).
        freqs_sin (torch.Tensor): Pre-computed sine values of shape (seq_len, head_dim/2).
        
    Returns:
        torch.Tensor: The input tensor with rotary embeddings applied.
    """
    # Reshape x to deal with pairs of dimensions
    # x -> (batch, seq_len, heads, head_dim/2, 2)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x1 = x_reshaped[..., 0] # Real part
    x2 = x_reshaped[..., 1] # Imaginary part
    
    # Reshape freqs for broadcasting
    # freqs -> (1, seq_len, 1, head_dim/2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    # Apply the rotation using the complex number multiplication logic
    # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    x_out_1 = x1 * freqs_cos - x2 * freqs_sin # New real part
    x_out_2 = x1 * freqs_sin + x2 * freqs_cos # New imaginary part
    
    # Stack and reshape back to original shape
    x_out = torch.stack([x_out_1, x_out_2], dim=-1)
    x_out = x_out.flatten(3)
    
    return x_out.type_as(x)


# --- Example Usage ---
batch_size = 2
seq_len = 64
n_heads = 8
d_model = 512
head_dim = d_model // n_heads # 64

# Create dummy query tensor
query = torch.randn(batch_size, seq_len, n_heads, head_dim)

# 1. Pre-compute RoPE frequencies
# Note: The dimension for freqs is the head dimension, not d_model
freqs_cos, freqs_sin = get_rope_embeddings(head_dim, seq_len)
print(f"Shape of freqs_cos: {freqs_cos.shape}")

# 2. Apply RoPE to the query tensor
rotated_query = apply_rotary_pos_emb(query, freqs_cos, freqs_sin)

print(f"Original query shape: {query.shape}")
print(f"Rotated query shape: {rotated_query.shape}")
# Verify norms are preserved (should be very close to 1.0)
torch.testing.assert_close(torch.linalg.norm(query), torch.linalg.norm(rotated_query))
print("Norm of tensors is preserved after rotation.")
```

**Q3: Create a function in PyTorch that generates the ALiBi bias mask for a given number of heads and sequence length.**

```python
import torch
import math
import matplotlib.pyplot as plt

def get_alibi_mask(n_heads: int, max_len: int):
    """
    Generates the ALiBi bias mask.

    Args:
        n_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.

    Returns:
        torch.Tensor: A tensor of shape (n_heads, max_len, max_len) with the ALiBi biases.
    """
    # Step 1: Calculate the slopes 'm' for each head
    # The slopes are chosen as powers of 2, from 2**(-8/n_heads) down to 2**(-1)
    def get_slopes(n):
        def get_next_power_of_2(n):
            return 2 ** math.floor(math.log2(n))
        
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    slopes = torch.Tensor(get_slopes(n_heads))
    
    # Step 2: Create the distance matrix. The bias is m * (j - i).
    # We only care about the lower triangle for causal attention.
    # The matrix should be [0, -1, -2, ...], [0, 0, -1, ...], etc.
    arange_tensor = torch.arange(max_len)
    # The matrix of relative positions (j-i)
    # This creates a matrix where entry (i, j) is j - i
    relative_positions = arange_tensor[None, :] - arange_tensor[:, None]
    
    # Make it causal by taking the absolute value and negating.
    # We want j-i for j<=i, so we want negative values or zero.
    alibi = -torch.abs(relative_positions)
    
    # Step 3: Multiply slopes with the distance matrix
    # Slopes: (n_heads) -> (n_heads, 1, 1) for broadcasting
    # Alibi: (max_len, max_len)
    # Result: (n_heads, max_len, max_len)
    alibi_mask = slopes.unsqueeze(-1).unsqueeze(-1) * alibi.unsqueeze(0)
    
    return alibi_mask

# --- Example Usage and Visualization ---
n_heads = 8
max_len = 128

alibi_mask = get_alibi_mask(n_heads, max_len)
print(f"Shape of ALiBi mask: {alibi_mask.shape}")

# Visualize the masks for a few heads
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
heads_to_show = [0, 2, 4, 7]

for i, head_idx in enumerate(heads_to_show):
    ax = axes[i]
    im = ax.imshow(alibi_mask[head_idx].numpy(), cmap='coolwarm')
    ax.set_title(f"Head {head_idx+1}")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    fig.colorbar(im, ax=ax)

plt.suptitle("ALiBi Mask Visualization (Linear Bias Penalty)", fontsize=16)
plt.show()
```