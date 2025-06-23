# A Comprehensive Guide to Fine-Tuning and Parameter-Efficient Fine-Tuning (PEFT) for Large Language Models

> This guide provides a deep dive into the concepts of model fine-tuning and the increasingly critical field of Parameter-Efficient Fine-Tuning (PEFT). We begin by establishing the fundamentals of full fine-tuning, exploring its process, advantages, and significant drawbacks like high computational cost and catastrophic forgetting. We then introduce PEFT as a solution, systematically categorizing and explaining its primary techniques, including Adapters, Prompt Tuning, and the highly influential Low-Rank Adaptation (LoRA). The guide concludes with a curated set of theoretical and practical interview questions, complete with detailed answers and code implementations, designed to prepare candidates for rigorous data science and machine learning interviews.

## Knowledge Section

### 1. The Foundation: Transfer Learning and Full Fine-Tuning

Fine-tuning is a powerful technique rooted in the broader machine learning paradigm of **transfer learning**. The core idea of transfer learning is to leverage knowledge gained from solving one problem and apply it to a different but related problem. In the context of Large Language Models (LLMs), this means taking a model pre-trained on a massive, general-purpose corpus of text (like Wikipedia and Common Crawl) and adapting it to perform well on a specific, often smaller, downstream task (e.g., medical text classification, legal document summarization).

#### 1.1 What is Full Fine-Tuning?

Full Fine-Tuning (FFT) is the classic approach to transfer learning. It involves unfreezing all the parameters of a pre-trained model and continuing the training process using a new, task-specific dataset. The model's weights, already initialized to a highly knowledgeable state, are updated via backpropagation to specialize for the nuances of the new task.

The typical workflow for full fine-tuning is as follows:

1.  **Select a Pre-trained Model**: Choose a powerful, general-purpose model suitable for the task. Examples include BERT for understanding-based tasks or GPT-family models for generation.
2.  **Prepare the Dataset**: Curate a labeled dataset specific to the downstream task. This dataset should be formatted to match the model's expected input and output structure.
3.  **Add a Task-Specific Head**: The pre-trained model provides the "body" or "backbone" for feature extraction and representation. To perform a specific task, a new, typically small, neural network layer called a "head" is appended to the model. For instance:
    *   **Classification**: A single linear layer followed by a softmax activation function.
    *   **Token Classification (NER)**: A linear layer applied to each token's output representation.
    *   **Question Answering**: Two linear layers to predict the start and end token positions of the answer span.
4.  **Initialize Parameters**: The model's backbone parameters are initialized with the downloaded pre-trained weights. The new task-specific head is typically initialized with random weights.
5.  **Train the Entire Model**: The entire model (backbone + head) is trained end-to-end on the new dataset. The learning process uses a standard optimization algorithm (like AdamW) to minimize a task-specific loss function. All parameters, from the first layer to the last, are updated.
6.  **Hyperparameter Tuning**: Optimize hyperparameters such as learning rate, batch size, and the number of training epochs to achieve the best performance on a validation set.
7.  **Evaluation**: Assess the final model's performance on a held-out test set using relevant metrics (e.g., accuracy, F1-score, ROUGE).

### 2. The Rise of Parameter-Efficient Fine-Tuning (PEFT)

While full fine-tuning is effective, it becomes prohibitively expensive as models scale into the billions or trillions of parameters. Training a model like GPT-3 (175 billion parameters) requires immense computational resources (hundreds of high-end GPUs) and time. Storing a separate, full-sized copy of the model for every single downstream task is also highly impractical.

**Parameter-Efficient Fine-Tuning (PEFT)** emerged to address these challenges. PEFT methods aim to adapt a pre-trained model to a downstream task by **tuning only a small subset of the model's parameters** (or a small number of newly added parameters), while keeping the vast majority of the original pre-trained weights frozen.

#### 2.1 Why is PEFT Necessary?

PEFT offers several compelling advantages over full fine-tuning:

1.  **Reduced Computational and Storage Costs**: By training and storing only a small fraction of the parameters, PEFT dramatically lowers the barrier to entry for customizing large models. Instead of storing a 100GB model for each task, one might only need to store a few megabytes of "delta" weights.
2.  **Mitigation of Catastrophic Forgetting**: Catastrophic forgetting occurs when a model, upon learning a new task, loses its proficiency on the original tasks it was trained on. Since PEFT leaves the core pre-trained weights untouched, it inherently preserves the model's general knowledge, leading to more robust performance.
3.  **Better Performance in Low-Data Regimes**: When the task-specific dataset is small, full fine-tuning runs a high risk of overfitting. PEFT, with its much smaller number of trainable parameters, acts as a form of regularization, often leading to better generalization on small datasets.
4.  **Portability and Modularity**: The small, task-specific weights trained by PEFT methods are highly portable. One can maintain a single, shared copy of the large pre-trained model and "plug in" different lightweight PEFT modules for different tasks, making deployment and management far simpler.
5.  **Comparable Performance**: Remarkably, many state-of-the-art PEFT techniques have been shown to achieve performance on par with full fine-tuning across a wide range of benchmarks, despite training less than 1% of the model's parameters.

### 3. A Taxonomy of PEFT Methods

PEFT is a rapidly evolving field. Based on the paper "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning," we can categorize the main strategies as follows.

![Taxonomy of Parameter-Efficient Fine-Tuning Methods](image/image_PqXGwJv0Yq.png)
*Figure 1: A taxonomy of PEFT methods, adapted from He et al. (2021).*

The main categories are:

1.  **Additive Methods**: These methods add new, trainable components to the model architecture while keeping the original model frozen.
2.  **Selective Methods**: These methods select a small subset of the existing model parameters to unfreeze and train.
3.  **Re-parameterization Methods**: These methods leverage low-rank representations to reduce the number of trainable parameters for weight updates.

Let's explore the most prominent examples in each category.

#### 3.1 Additive Methods

##### Adapter Layers
**Adapters** are small, bottleneck-style neural network modules inserted between the layers of a pre-trained model. During fine-tuning, only the parameters of these newly added adapter layers are trained.

*   **How it works**: An adapter module typically consists of a down-projection linear layer that reduces the feature dimension, a non-linear activation function (like ReLU or GeLU), and an up-projection linear layer that restores the original dimension. A residual connection is added around the adapter.
*   **Pros**: Excellent parameter efficiency; strong isolation of task-specific knowledge.
*   **Cons**: Can introduce a small amount of inference latency because of the additional network layers that must be traversed.

##### Soft Prompts (Prompt Tuning, Prefix-Tuning)
Instead of modifying the model's weights, these methods focus on manipulating the input.

*   **Prompt Tuning**: This technique prepends a sequence of continuous, learnable embeddings (a "soft prompt") to the input sequence. The model's weights remain completely frozen, and only these prompt embeddings are updated during training. The model learns to interpret these special embeddings to condition its behavior for the specific task.
*   **Prefix-Tuning**: A more powerful variant that adds learnable prefixes not just to the input embeddings but to the key and value vectors in each attention layer of the model. This gives the model more fine-grained control over its internal representations.
*   **Pros**: Minimal number of trainable parameters; no change to the model architecture.
*   **Cons**: Can sometimes be less expressive or stable than methods that modify weights more directly.

#### 3.2 Selective Methods

This is the most straightforward approach: freeze the entire model except for a small, strategically chosen subset of its existing parameters.

*   **Layer Tuning**: Only unfreezing and training the top few layers of the model, based on the assumption that later layers capture more task-specific features.
*   **Sparse Fine-Tuning / BitFit**: An extreme but effective example is **BitFit**, which proposes to tune only the *bias* parameters of the model (and the task-specific head), keeping all other weight matrices frozen. This is surprisingly effective for many tasks and represents a tiny fraction of the total parameters.

#### 3.3 Re-parameterization Methods: LoRA

**Low-Rank Adaptation (LoRA)** has become one of the most popular and effective PEFT methods. It is based on the hypothesis that the change in weights during model adaptation has a low "intrinsic rank."

*   **The Core Idea**: For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, full fine-tuning computes an update $\Delta W$, resulting in a new weight matrix $W' = W_0 + \Delta W$. LoRA hypothesizes that $\Delta W$ can be effectively approximated by a low-rank decomposition.
*   **Mathematical Formulation**: LoRA approximates the update matrix $\Delta W$ with the product of two much smaller matrices, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$.
    $$ W' = W_0 + \Delta W \approx W_0 + B A $$
    During training, $W_0$ is frozen, and only the parameters of matrices $A$ and $B$ are updated. The forward pass is modified as:
    $$ h = W_0 x + B A x $$
*   **Why it's Efficient**: The number of trainable parameters is $d \times r + r \times k$, which is significantly smaller than the $d \times k$ parameters in the original $\Delta W$. For example, if $d=4096$, $k=4096$, and we choose a rank $r=8$, we train only $4096 \times 8 + 8 \times 4096 \approx 65,000$ parameters instead of over 16 million.
*   **Key Advantage**: During inference, the learned matrices can be merged back into the original weight matrix: $W' = W_0 + BA$. This means LoRA introduces **zero inference latency** compared to the original model, a major advantage over methods like adapters.

### 4. Challenges and Best Practices in PEFT Evaluation

Despite its success, the field of PEFT faces challenges that make direct comparisons between methods difficult.

**Current Challenges**:

*   **Inconsistent Parameter Counting**: Different papers may report "trainable parameters" differently. Some count only the new parameters, while others might include parameters that are trained but not stored (e.g., in complex optimizers). LoRA's effectiveness is best measured by the number of parameters in A and B, which directly relates to storage cost.
*   **Varying Model Sizes**: The effectiveness of a PEFT method can change depending on the size of the base model. A method that works well on a 1B parameter model may not be the best for a 100B parameter model.
*   **Lack of Standardized Benchmarks**: Different studies use different combinations of models, datasets, and evaluation metrics, making it hard to draw universal conclusions.
*   **Code Reproducibility**: Many research codebases are complex, poorly documented, and tightly coupled to specific versions of libraries, hindering reproducibility and adoption.

**Best Practices for Research and Application**:

*   **Be Explicit**: When reporting results, clearly state which parameters are being counted (e.g., "storage-relevant trainable parameters").
*   **Evaluate at Scale**: Test PEFT methods across various model sizes to understand their scaling properties.
*   **Standardize Benchmarks**: Use established benchmarks and compare against well-known methods like full fine-tuning and LoRA as baselines.
*   **Prioritize Clean Code**: Implement PEFT techniques in a modular, minimal, and well-documented way to promote reusability. The Hugging Face `peft` library is an excellent example of this.

---

## Interview Questions

### Theoretical Questions

#### Question 1: What is the fundamental difference between full fine-tuning and parameter-efficient fine-tuning?

**Answer:**
The fundamental difference lies in **which parameters of a pre-trained model are updated** during adaptation to a new task.

*   In **Full Fine-Tuning (FFT)**, the entire set of the model's parameters is unfrozen and updated via backpropagation. This means every weight and bias in the model is subject to change. It is conceptually simple but computationally and storage-intensive. A complete, new copy of the model must be saved for each task.

*   In **Parameter-Efficient Fine-Tuning (PEFT)**, the vast majority of the pre-trained model's parameters (often >99%) are kept frozen. Only a small, carefully selected subset of parameters (or a small number of newly added parameters) are trained. This dramatically reduces memory and storage requirements, making it feasible to adapt massive models to many tasks simultaneously.

The trade-off is that FFT has the potential to achieve the highest possible performance by adjusting the entire model, but at a great cost and with a higher risk of catastrophic forgetting. PEFT offers a highly efficient alternative that often matches FFT's performance while being more scalable and robust against forgetting.

---

#### Question 2: Explain the mathematical principle behind Low-Rank Adaptation (LoRA). Why is it "parameter-efficient" and how does it avoid introducing inference latency?

**Answer:**

**1. Mathematical Principle:**
LoRA is based on the empirical observation that the update to a model's weight matrices during fine-tuning often has a low "intrinsic rank." This means the change matrix, $\Delta W$, can be effectively approximated by a much simpler structure.

Let $W_0 \in \mathbb{R}^{d \times k}$ be a weight matrix from the pre-trained model. In full fine-tuning, we would learn a dense update matrix $\Delta W \in \mathbb{R}^{d \times k}$ such that the new weight is $W' = W_0 + \Delta W$.

LoRA proposes to approximate this update using a low-rank decomposition. Specifically, it represents $\Delta W$ as the product of two smaller matrices:
$$ \Delta W \approx B A $$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. The hyperparameter $r$ is the **rank** of the adaptation, and it is chosen such that $r \ll \min(d, k)$.

The forward pass for a layer modified with LoRA becomes:
$$ h = W_0 x + (B A) x $$
During training, $W_0$ is frozen, and only the parameters in matrices $A$ and $B$ are updated.

**2. Why it is Parameter-Efficient:**
The efficiency comes from the reduction in the number of trainable parameters.
-   The original update matrix $\Delta W$ has $d \times k$ parameters.
-   The LoRA approximation $BA$ has $(d \times r) + (r \times k) = r(d+k)$ parameters.

Since the rank $r$ is very small (e.g., 4, 8, or 16), the number of trainable parameters in LoRA is drastically smaller than in the original matrix.
*   **Example**: For a typical Transformer layer where $d=k=4096$, $\Delta W$ has $4096 \times 4096 \approx 16.7$ million parameters. With LoRA using $r=8$, the number of trainable parameters is just $8 \times (4096 + 4096) = 65,536$, which is a **~256x reduction** in parameters for that layer.

**3. How it Avoids Inference Latency:**
This is a key advantage of LoRA. While the training-time forward pass is $h = W_0x + BAx$, this can be simplified for deployment. Since $W_0$, $B$, and $A$ are all constant matrices during inference, the matrix product $BA$ can be pre-computed to get a single matrix $\Delta W_{lora} = BA$. This matrix can then be directly added to the original weight matrix:
$$ W' = W_0 + \Delta W_{lora} $$
The result is a single weight matrix $W'$ of the same dimensions as the original $W_0$. The forward pass during inference becomes $h = W'x$, which is computationally identical to the original, unmodified layer. Therefore, LoRA introduces **zero additional inference latency**. This is a major advantage over methods like Adapters, which add extra layers that must be traversed during every forward pass.

---

#### Question 3: Compare and contrast Adapter-based methods and LoRA. What are the key trade-offs, particularly concerning inference latency?

**Answer:**
Both Adapters and LoRA are "additive" PEFT methods in the sense that they introduce new parameters to be trained while keeping the base model frozen. However, they do so in fundamentally different ways.

**Adapters:**
*   **Mechanism**: Inserts small, self-contained neural network modules (typically two linear layers with a non-linearity in between) *between* the existing layers of the Transformer architecture.
*   **Data Flow**: The output of a Transformer sub-layer (e.g., self-attention) is passed *sequentially* through the adapter module before being fed to the next sub-layer.
*   **Parameters**: The parameters of the adapter layers are trained.
*   **Inference Latency**: **Introduces latency**. Because adapters are separate modules, each forward pass requires executing these additional layers, increasing the overall computation time and model depth.

**LoRA (Low-Rank Adaptation):**
*   **Mechanism**: Modifies the forward pass of existing layers (typically linear or attention projection layers) by adding a *parallel* low-rank path. It approximates the weight update matrix $\Delta W$ as a product of two smaller matrices, $BA$.
*   **Data Flow**: The low-rank path $BAx$ is computed in parallel with the original frozen path $W_0x$, and their results are added together: $h = W_0x + BAx$.
*   **Parameters**: The parameters of the low-rank matrices $A$ and $B$ are trained.
*   **Inference Latency**: **Introduces zero latency**. After training, the learned matrices $B$ and $A$ can be multiplied and added directly to the frozen weight matrix $W_0$ to produce a new, single weight matrix $W' = W_0 + BA$. The inference forward pass is then just $h = W'x$, which is identical in cost to the original model's forward pass.

**Key Trade-offs:**

| Feature               | Adapters                                        | LoRA                                                        |
| --------------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| **Mechanism**         | Sequential insertion of new layers.             | Parallel path added to existing layers.                     |
| **Inference Latency** | **Non-zero**. Adds computational steps.         | **Zero**. Merges into original weights for inference.         |
| **Parameter Count**   | Very low.                                       | Very low, tunable with rank `r`.                            |
| **Implementation**    | Requires modifying the model's `forward` graph. | Can be implemented cleanly as a wrapper around layers.      |
| **Performance**       | Generally strong.                               | Generally state-of-the-art, often outperforming adapters.   |
| **Primary Use Case**  | When modularity is key and slight latency is acceptable. | When inference speed is critical and performance is paramount. |

In summary, the most critical trade-off is **inference latency**. LoRA's ability to be merged back into the original model architecture makes it the superior choice for production environments where every millisecond of latency counts. Adapters might be simpler to reason about as "black box" modules but come with an unavoidable performance cost at inference time.

---

### Practical & Coding Questions

#### Question 4: Implement a LoRA-enabled Linear layer from scratch in PyTorch. It should inherit from `torch.nn.Linear` and handle the LoRA logic.

**Answer:**

This implementation demonstrates how to create a custom `LoRALinear` layer. The key is to add the LoRA matrices `lora_A` and `lora_B`, freeze the original weights, and modify the `forward` pass to include the low-rank path.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    A PyTorch Linear layer augmented with Low-Rank Adaptation (LoRA).
    This layer freezes the original pre-trained weights (W0) and
    introduces two trainable low-rank matrices (A and B) to approximate
    the weight update (delta W = B * A).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_alpha: float = 1.0,
        **kwargs
    ):
        """
        Args:
            in_features (int): Number of input features for the linear layer.
            out_features (int): Number of output features for the linear layer.
            rank (int): The rank of the low-rank adaptation. A smaller rank means fewer trainable parameters.
            lora_alpha (float): A scaling factor for the LoRA output, similar to the one used in the official LoRA implementation.
        """
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha

        # The original, frozen linear layer
        self.linear = nn.Linear(in_features, out_features, **kwargs)

        # Create the low-rank matrices A and B
        # B is initialized with zeros, A with Kaiming uniform
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Initialize A like a standard linear layer

        # Freeze the original weights of the linear layer
        self.linear.weight.requires_grad = False
        # If the original layer has a bias, it can be optionally trained or frozen.
        # Here we freeze it for consistency with the core LoRA principle.
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.scaling = self.lora_alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass combines the output of the frozen original layer
        with the output of the low-rank adaptation path.
        """
        # Original frozen path
        frozen_output = self.linear(x)

        # Low-rank adaptation path
        # Note the order: x -> A -> B
        lora_update = (self.lora_B @ (self.lora_A @ x.transpose(-2, -1))).transpose(-2, -1)

        # Combine and scale
        return frozen_output + lora_update * self.scaling

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        super().train(mode)
        # Ensure the original linear layer's weights remain frozen during training
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        return self

    def extra_repr(self) -> str:
        return f'in_features={self.linear.in_features}, out_features={self.linear.out_features}, rank={self.rank}, bias={self.linear.bias is not None}'

# --- Example Usage ---
# Configuration
in_dim = 512
out_dim = 1024
lora_rank = 8
batch_size = 4
seq_len = 10

# Create the LoRA layer
lora_layer = LoRALinear(in_features=in_dim, out_features=out_dim, rank=lora_rank)
print(lora_layer)

# Create some dummy input data
input_tensor = torch.randn(batch_size, seq_len, in_dim)

# Pass data through the layer
output = lora_layer(input_tensor)
print(f"\nInput shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")

# Verify which parameters are trainable
total_params = 0
trainable_params = 0
print("\n--- Parameter Trainability Check ---")
for name, param in lora_layer.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"✅ Trainable: {name} - Shape: {param.shape} - Params: {param.numel()}")
    else:
        print(f"❌ Frozen:    {name} - Shape: {param.shape} - Params: {param.numel()}")

print("\n--- Summary ---")
original_params = lora_layer.linear.weight.numel()
lora_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
print(f"Original Linear Layer Params (Frozen): {original_params:,}")
print(f"LoRA Trainable Params: {trainable_params:,} (A: {lora_layer.lora_A.numel():,}, B: {lora_layer.lora_B.numel():,})")
print(f"Parameter Efficiency Ratio (Trainable/Original): {trainable_params / original_params:.2%}")

```
---

#### Question 5: Using the Hugging Face `peft` library, demonstrate how to apply LoRA to a pre-trained model like `bert-base-uncased`.

**Answer:**

This code demonstrates the standard workflow for applying LoRA to a Hugging Face Transformer model using the `peft` library. It shows how simple it is to convert a large model into a parameter-efficient one.

**You will need to install the required libraries first:**
`pip install transformers datasets peft accelerate`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# 1. Define Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model for a sequence classification task with 2 labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# --- Helper function to print trainable parameters ---
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

print("--- Before applying PEFT (Full Fine-Tuning) ---")
print_trainable_parameters(model)
# Expected output will show that 100% of parameters are trainable.

# 2. Define LoRA Configuration
# This configuration tells the `peft` library how to apply LoRA.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Task type: Sequence Classification
    r=8,                       # Rank of the update matrices
    lora_alpha=16,             # Alpha parameter for scaling
    lora_dropout=0.1,          # Dropout probability for LoRA layers
    bias="none",               # Do not train bias terms
    # Target the query and value projection layers in the self-attention modules
    target_modules=["query", "value"]
)

# 3. Apply PEFT to the model
peft_model = get_peft_model(model, lora_config)

print("\n--- After applying PEFT with LoRA ---")
print_trainable_parameters(peft_model)
# Expected output will show that only a tiny fraction of parameters are trainable.

# 4. Show the modified model architecture
print("\n--- PEFT Model Architecture (excerpt) ---")
print(peft_model.bert.encoder.layer[0].attention.self)
# You will see the original `Linear` layers for query/value are now replaced
# with a `peft.tuners.lora.Linear` layer which contains the LoRA logic.

# 5. Example usage (Inference / Training)
# The model can be used just like a regular Hugging Face model.
# The `peft` wrapper handles all the LoRA logic internally.
text = "This is a test sentence for our PEFT model."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    # The forward pass automatically combines the base model and LoRA adapter outputs
    outputs = peft_model(**inputs)
    logits = outputs.logits

print(f"\nOutput logits shape: {logits.shape}")
print("The model is now ready for training. Only the LoRA weights will be updated.")
```

This practical example clearly shows the power and simplicity of the `peft` library. With just a few lines of code, a massive model can be prepared for efficient fine-tuning, with the library automatically identifying and replacing the target layers with their LoRA-enabled counterparts.

---

### References
*   He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2021). *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning*. arXiv preprint arXiv:2110.04366.