# Neural Networks and Deep Learning

Educational implementation of neural networks based on Michael Nielsen's book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com). This repository provides a progressive learning path from basic neural networks to advanced convolutional architectures.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [What You'll Learn](#what-youll-learn)
- [Learning Path](#learning-path)
- [Core Concepts](#core-concepts)
- [Hands-On Exercises](#hands-on-exercises)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Installation

### Python Version Compatibility

- **Python 3.10**: ✅ Full support (all modules including CNNs)
- **Python 3.11**: ⚠️  Partial support (`network.py` and `network2.py` only)
- **Python 3.13+**: ❌ Not recommended

### Quick Install

```bash
# 1. Create virtual environment (use Python 3.10 or 3.11)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

### What Works with Your Python Version

**Python 3.11** (Main learning modules):
- ✅ `network.py` - Basic neural networks
- ✅ `network2.py` - Improved techniques  
- ✅ All core concepts and exercises
- ❌ `network3.py` - CNNs (requires Python 3.10)

**Python 3.10** (Full features):
- ✅ Everything above, plus
- ✅ `network3.py` - Convolutional Neural Networks
- ✅ GPU acceleration with Theano

### Verify Installation

```python
import sys
sys.path.append('src')
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(f"✓ {len(training_data)} training samples loaded")
print(f"✓ Installation successful!")
```

---

## Quick Start

### Your First Neural Network (5 minutes)

Create `examples/first_network.py`:

```python
import sys
sys.path.append('../src')
import mnist_loader
import network

# Load MNIST data
print("Loading data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network: 784 inputs → 30 hidden → 10 outputs
print("Creating network...")
net = network.Network([784, 30, 10])

# Train for 30 epochs
print("Training...")
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
```

Run it:
```bash
cd examples
python first_network.py
```

**Expected output**: ~95% accuracy after 30 epochs

### Quick Experiments

```python
# 1. Change architecture
net = network.Network([784, 100, 10])  # More neurons

# 2. Adjust learning rate
net.SGD(training_data, 30, 10, eta=1.0, test_data=test_data)

# 3. Use improved version (network2.py)
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, 
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)
```

---

## What You'll Learn

### Three Progressive Implementations

**1. `network.py` - Fundamentals**
- Feedforward neural networks
- Backpropagation algorithm  
- Stochastic Gradient Descent
- Sigmoid activation
- **Goal**: Understand core concepts
- **Accuracy**: ~95%

**2. `network2.py` - Optimization**
- Cross-entropy cost function
- L2 regularization
- Better weight initialization
- Model save/load
- **Goal**: Learn best practices
- **Accuracy**: ~96-97%

**3. `network3.py` - Advanced (Python 3.10 only)**
- Convolutional Neural Networks
- GPU acceleration (Theano)
- Dropout regularization
- Multiple activation functions (ReLU, tanh)
- **Goal**: Master deep learning
- **Accuracy**: ~98-99%

---

## Learning Path

### Prerequisites

- **Python**: Intermediate level (classes, NumPy basics)
- **Math**: Linear algebra, calculus basics
- **No ML experience required!**

### Recommended Learning Sequence

#### Week 1: Fundamentals (`network.py`)
1. Read Chapter 1 of Nielsen's book
2. Understand feedforward and backpropagation
3. Train your first network
4. Experiment with hyperparameters

**Checkpoint**: Train a network achieving >90% accuracy

#### Week 2: Improvements (`network2.py`)
1. Read Chapter 3 of Nielsen's book
2. Learn about cost functions
3. Implement regularization
4. Compare results with network.py

**Checkpoint**: Achieve >96% accuracy with regularization

#### Week 3-4: Deep Learning (`network3.py`)
1. Read Chapter 6 of Nielsen's book  
2. Understand convolutional layers
3. Build your first CNN
4. Experiment with architectures

**Checkpoint**: Build a CNN achieving >98% accuracy

---

## Core Concepts

### Neural Network Architecture

```
Input Layer (784) → Hidden Layer(s) → Output Layer (10)
                    Each neuron: σ(w·x + b)
```

**Components**:
- **Weights (w)**: Connection strengths
- **Biases (b)**: Activation thresholds
- **Activation (σ)**: Non-linear transformation

### Forward Propagation

```python
def feedforward(self, a):
    """Compute network output from input"""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a
```

### Backpropagation

Four fundamental equations:
1. **Output error**: δ^L = ∇C ⊙ σ'(z^L)
2. **Error propagation**: δ^l = (w^(l+1))^T δ^(l+1) ⊙ σ'(z^l)
3. **Gradient (bias)**: ∂C/∂b = δ
4. **Gradient (weight)**: ∂C/∂w = δ·a^T

### Stochastic Gradient Descent

```python
# Update weights using mini-batches
for mini_batch in mini_batches:
    gradients = compute_gradients(mini_batch)
    weights = weights - (learning_rate / batch_size) * gradients
```

**Key parameters**:
- **Learning rate (η)**: Step size (typical: 0.1-3.0)
- **Mini-batch size**: Samples per update (typical: 10-100)
- **Epochs**: Full passes through data (typical: 10-60)

### Cost Functions

**Quadratic** (network.py):
```
C = (1/2n)Σ||y - a||²
```

**Cross-Entropy** (network2.py - faster learning):
```
C = -(1/n)Σ[y·ln(a) + (1-y)·ln(1-a)]
```

### Regularization

**L2 Regularization** (prevents overfitting):
```
C = C₀ + (λ/2n)Σw²
Weight update: w → (1 - ηλ/n)w - (η/m)∇C₀
```

---

## Hands-On Exercises

### Exercise 1: First Neural Network

```python
import sys
sys.path.append('src')
import mnist_loader
import network

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create and train network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

**Expected**: ~95% accuracy

**Experiments**:
1. Try different hidden layer sizes: 15, 50, 100
2. Adjust learning rates: 0.5, 1.0, 5.0
3. Change mini-batch sizes: 1, 20, 100
4. Add more layers: `[784, 30, 30, 10]`

**Questions**:
- How does network size affect accuracy?
- What happens with very high/low learning rates?
- Why does mini-batch size matter?

### Exercise 2: Improved Techniques

```python
import network2

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create improved network
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# Train with monitoring
net.SGD(training_data, 30, 10, 0.5,
        lmbda=5.0,  # L2 regularization
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True)

# Save model
net.save("my_network.json")
```

**Expected**: ~96-97% accuracy

**Experiments**:
1. Compare `QuadraticCost` vs `CrossEntropyCost`
2. Try λ values: 0.0, 0.1, 1.0, 5.0, 10.0
3. Compare weight initializations
4. Plot training vs validation curves

### Exercise 3: Convolutional Networks (Python 3.10)

```python
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

# Load data
training_data, validation_data, test_data = network3.load_data_shared()

# Build CNN
net = Network([
    ConvPoolLayer(
        image_shape=(10, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2, 2),
        activation_fn=ReLU
    ),
    FullyConnectedLayer(n_in=20*12*12, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)
], mini_batch_size=10)

# Train
net.SGD(training_data, 60, 10, 0.1, validation_data, test_data)
```

**Expected**: ~98-99% accuracy

**Experiments**:
1. Try different architectures (shallow vs deep)
2. Compare activation functions (sigmoid vs ReLU)
3. Experiment with dropout
4. Visualize learned filters

### Exercise 4: Understanding Through Experiments

**Overfitting Detection**:
```python
# Train without regularization
net.SGD(training_data, 30, 10, 0.5, lmbda=0.0, ...)

# Train with regularization
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, ...)

# Compare training vs validation accuracy
```

**Learning Rate Effects**:
```python
for eta in [0.1, 0.5, 1.0, 3.0, 10.0]:
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 10, 10, eta, test_data=test_data)
```

---

## Advanced Topics

### Data Augmentation

```python
import expand_mnist

# Generate augmented dataset
# Creates rotated/shifted versions of digits
# Saves to: data/mnist_expanded.pkl.gz

# Use expanded data for training
from conv import expanded_data
expanded_data(n=100)
```

### Hyperparameter Tuning

**Learning Rate**:
- Too high → unstable training
- Too low → slow convergence
- Strategy: Start high (3.0), decay over time

**Architecture**:
- More layers → more capacity, harder to train
- More neurons → more representation power
- Trade-off: Complexity vs overfitting

**Regularization (λ)**:
- Too high → underfitting
- Too low → overfitting  
- Sweet spot: Usually 0.1-10.0

### Ensemble Learning

```python
# Train multiple networks
nets = []
for i in range(5):
    net = network.Network([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0)
    nets.append(net)

# Combine predictions by voting
def ensemble_predict(x):
    votes = [net.feedforward(x) for net in nets]
    return most_common(votes)
```

**Why it works**:
- Different networks make different errors
- Averaging reduces variance
- Typically 1-2% improvement

---

## Troubleshooting

### Installation Issues

**"Cannot import setuptools.build_meta"**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**NumPy/Theano won't install**
- Use Python 3.10 or 3.11 (not 3.13+)
- Old packages aren't compatible with newest Python

**"No module named 'theano'"**
- Expected with Python 3.11
- `network.py` and `network2.py` work fine without it
- For CNNs, use Python 3.10

### Training Issues

**Vanishing Gradients** (weights barely change):
- Use better initialization (network2.py)
- Try ReLU instead of sigmoid
- Check learning rate

**Exploding Gradients** (NaN values):
- Lower learning rate
- Better initialization
- Gradient clipping

**Overfitting** (training >> validation accuracy):
- Add L2 regularization
- Use dropout (network3.py)
- Get more training data
- Reduce network size

**Underfitting** (both accuracies low):
- Increase network capacity
- Train longer
- Reduce regularization
- Check learning rate

**Slow Training**:
- Use ReLU instead of sigmoid
- Reduce epochs/neurons for testing
- Use GPU (network3.py with Python 3.10)

### Common Errors

```python
# AttributeError: can't set attribute
# Fix: Use network2.default_weight_initializer()

# Shape mismatch errors
# Fix: Check input dimensions (should be 784×1)

# Poor accuracy (<80%)
# Fix: Check learning rate, train longer, or increase network size
```

---

## Project Structure

```
neural-networks-and-deep-learning/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── src/
│   ├── network.py         # Basic implementation
│   ├── network2.py        # Improved version
│   ├── network3.py        # CNNs with Theano
│   ├── conv.py            # CNN experiments
│   ├── mnist_loader.py    # Data loading
│   ├── expand_mnist.py    # Data augmentation
│   └── mnist_svm.py       # SVM comparison
├── data/
│   └── mnist.pkl.gz       # MNIST dataset
└── fig/                   # Visualizations
```

### Key Files

- **network.py**: Learn backpropagation from scratch
- **network2.py**: Production-ready techniques
- **network3.py**: High-performance CNNs
- **conv.py**: Pre-built experiments
- **mnist_loader.py**: Data preprocessing

---

## Learning Checkpoints

### Beginner (network.py)
- [ ] Understand feedforward computation
- [ ] Grasp backpropagation algorithm
- [ ] Train network with >90% accuracy
- [ ] Experiment with hyperparameters

### Intermediate (network2.py)
- [ ] Explain cross-entropy vs quadratic cost
- [ ] Implement regularization effectively
- [ ] Achieve >96% accuracy
- [ ] Detect and prevent overfitting
- [ ] Save and load models

### Advanced (network3.py)
- [ ] Build and train CNNs
- [ ] Understand convolutional operations
- [ ] Use dropout regularization
- [ ] Achieve >98% accuracy
- [ ] Visualize learned features

---

## Resources

### Primary Resource
- **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)** by Michael Nielsen
  - Read alongside this code
  - Excellent explanations of concepts

### Additional Learning
- **3Blue1Brown**: Neural network video series (visual intuition)
- **CS231n**: Stanford's CNN course
- **Deep Learning Book**: Goodfellow, Bengio, Courville
- **fast.ai**: Practical deep learning course

### Communities
- r/MachineLearning (Reddit)
- Towards Data Science (Medium)
- Papers with Code
- Kaggle (practice competitions)

### Next Steps After This Project
1. **Modern Frameworks**: PyTorch or TensorFlow
2. **Advanced Architectures**: ResNets, Transformers
3. **Other Domains**: NLP, Reinforcement Learning
4. **Theory**: Optimization algorithms, batch normalization

---

## Quick Reference

### Typical Hyperparameters

| Parameter | network.py | network2.py | network3.py |
|-----------|-----------|-------------|-------------|
| Learning rate | 3.0 | 0.5 | 0.1 |
| Mini-batch size | 10 | 10 | 10 |
| Epochs | 30 | 30 | 60 |
| Hidden neurons | 30 | 30-100 | 100-1000 |
| Regularization | - | λ=5.0 | λ=0.1 |

### Common Commands

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Test installation
python -c "import sys; sys.path.append('src'); import network; print('✓ Ready!')"

# Train basic network
cd examples && python first_network.py
```

### Import Patterns

```python
# Basic network
import network
net = network.Network([784, 30, 10])

# Improved network
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# CNN (Python 3.10 only)
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU
```

---

## About This Fork

This is a fork of [Michael Nielsen's original repository](https://github.com/mnielsen/neural-networks-and-deep-learning), updated for Python 3.8-3.11 compatibility. The code prioritizes:

- **Clarity** over performance
- **Simplicity** over features  
- **Understanding** over efficiency

Perfect for learning, not for production!

---

## License

MIT License - See original repository for details.

Copyright (c) 2012-2018 Michael Nielsen

---

## Final Thoughts

The goal isn't just to achieve high accuracy, but to **deeply understand** how neural networks work. This codebase gives you:

1. **Clean implementations** to read and learn from
2. **Progressive complexity** from simple to advanced
3. **Hands-on experiments** to build intuition
4. **Solid foundations** for modern frameworks

**Start simple. Experiment often. Build intuition. Have fun!** 🚀

---

Questions? Issues? Open an issue or refer to [Nielsen's book](http://neuralnetworksanddeeplearning.com) for detailed explanations.

Happy learning! 🎓
