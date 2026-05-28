#  Arithmetic Transformer  
*A Transformer-based Neural Network for Learning Addition and Subtraction*


## Overview

This project implements a Transformer-based sequence-to-sequence model inspired by the paper **“Attention Is All You Need”** to perform arithmetic operations such as addition and subtraction.

Instead of using symbolic rules or hard-coded logic, the model learns arithmetic patterns from data by training on thousands of synthetic examples. The goal is to analyze whether a Transformer can:

- Learn digit-level arithmetic  
- Handle carry propagation  
- Generalize to unseen number ranges  
- Exhibit algorithmic reasoning behavior  


## Motivation

Arithmetic is a structured, rule-based problem that requires:

- Long-range dependencies (carry propagation)
- Sequential reasoning
- Digit alignment and positional awareness

This project explores whether a Transformer — originally designed for language modeling and translation — can learn structured arithmetic reasoning purely from data.


##  Model Architecture

The implementation follows the standard Encoder–Decoder Transformer architecture:

### 🔹 Encoder
- Token Embedding  
- Positional Encoding  
- Multi-Head Self-Attention  
- Feedforward Network (Linear → ReLU → Linear)  
- Residual Connections + Layer Normalization  

### 🔹 Decoder
- Masked Self-Attention  
- Encoder–Decoder Cross-Attention  
- Feedforward Network  
- Residual Connections + Layer Normalization  
- Final Linear Layer + Softmax  


## Data Pipeline

### Example Input

23 + 58 = 81



### Pipeline Steps
1. Character-level tokenization  
2. Convert tokens to token IDs  
3. Embedding lookup  
4. Add positional encoding  
5. Encoder–decoder forward pass  
6. Cross-entropy loss computation  
7. Backpropagation & weight updates  


## Training Details

- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Decoding:** Autoregressive  
- **Training Strategy:** Teacher Forcing  
- **Masking:** Future tokens masked in decoder  

The model predicts one output token at a time while preventing access to future tokens during training.


##  Key Concepts Demonstrated

- Self-attention mechanism  
- Multi-head attention  
- Positional encoding  
- Masked decoding  
- Residual learning  
- Emergent carry behavior  
- Generalization vs memorization  


##  Experimental Observations

- High accuracy within training number ranges  
- Performance drops on larger unseen numbers  
- Model learns patterns but may not perfectly generalize algorithmic reasoning  


##  Limitations

- Attention complexity is O(n²)  
- Performance degrades for much longer sequences  
- No explicit symbolic reasoning — learning is data-driven  


##  How to Run

### Clone Repository

git clone https://github.com/itzswarnadeep/Arithmetic-Transformer-Neural-Arithmetic-Reasoning-main.git
