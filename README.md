# LLM From Scratch

This repository contains an implementation of a Language Model built from scratch in PyTorch, inspired by Andrej Karpathy's YouTube series on building LLMs. The code provides a foundational understanding of how modern Large Language Models work, starting from a simple bigram model and extending to a transformer-based architecture.

## Overview

This project demonstrates the core components of transformer-based language models, helping to demystify how models like GPT work under the hood. The implementation focuses on clarity and educational value rather than performance optimization.

## How It Works

### The Bigram Model

The simplest model in this repo is a bigram language model:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

This model predicts the next token based solely on the current token through a direct embedding lookup. While simple, it demonstrates the fundamentals of language modeling: predicting the next token in a sequence.

### Transformer Architecture

The full implementation includes a transformer-based model with the following key components:

#### 1. Self-Attention Mechanism

The attention mechanism is the core innovation that powers modern LLMs:

```python
# simplified representation
def attention(q, k, v, mask=None):
    # compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # apply mask and softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    
    # get weighted values
    output = torch.matmul(attention_weights, v)
    return output
```

**What Attention Does**:
- Allows the model to focus on different parts of the input sequence when producing each output token
- Enables capturing long-range dependencies in text
- Creates a dynamic, content-based weighting system where each token "pays attention" to relevant tokens

#### 2. Add & Normalize Layers

```python
# residual connection + layer normalization
x = x + sublayer(x)  # Add (residual connection)
x = nn.LayerNorm(x)  # Normalize
```

**What Add & Normalize Does**:
- **Residual Connections (Add)**: Allow gradients to flow more easily through the network, preventing vanishing gradient problems and enabling training of much deeper networks
- **Layer Normalization**: Stabilizes the learning process by normalizing the inputs across features, making training faster and more stable

#### 3. Feed-Forward Neural Networks (MLP)

```python
class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),  # Non-linear activation
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

**What the MLP Does**:
- Processes each position independently with the same neural network
- Introduces non-linearity into the model, allowing it to learn more complex patterns
- Transforms the representation space after attention has mixed information across positions

## Training Process

The model is trained to predict the next token in a sequence by minimizing cross-entropy loss:

```python
loss = F.cross_entropy(logits, targets)
```

During generation, we sample from the probability distribution of the next token:

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

## Special Thanks

This implementation is based on the excellent educational content by **Andrej Karpathy**. His step-by-step tutorials on building neural networks and language models from scratch have been instrumental in demystifying these complex systems. I highly recommend checking out his ["Building GPT from Scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) YouTube series for a deeper understanding of how these models work.

## Getting Started

1. Clone this repository
2. Install the required dependencies: `pip install torch`
3. Run the training script: `python train.py`
4. Generate text with the trained model: `python generate.py`


## License

[MIT License](LICENSE)
