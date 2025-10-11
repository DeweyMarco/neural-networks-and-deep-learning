# Neural Network Fundamentals: A Hands-On Learning Journey

This folder contains a **progressive educational series** of 10 neural network experiments designed to teach you how neural networks work by exploring different hyperparameters and architectures on the MNIST digit classification task.

## Learning Path

Read and run these files **in order** for the best learning experience. Each file builds on knowledge from previous ones.

### Part 1: Foundation (START HERE!)

**network_1.0.0.py** - Your First Neural Network
- Architecture: `[784, 30, 10]` (23,860 parameters)
- Comprehensive introduction to neural networks
- Covers: layers, neurons, weights, biases, backpropagation, SGD
- Standard hyperparameters: η=3.0, batch=10, epochs=30
- Expected accuracy: ~95%
- **Start here!** This establishes the baseline for all comparisons.

---

### Part 2: Learning Rate Experiments

Explore how learning rate affects training speed and stability.

**network_1.0.1.py** - Lower Learning Rate (η=0.5)
- 6× smaller than baseline
- Effect: Slower but more stable learning
- Expected: ~85-90% accuracy (needs more epochs)
- Lesson: Stability vs speed trade-off

**network_1.0.2.py** - Higher Learning Rate (η=5.0)
- 1.67× larger than baseline
- Effect: Fast but noisy/unstable learning
- Expected: ~95% in 15-20 epochs (but erratic)
- Lesson: Speed vs stability risk

**Key Insight:** Learning rate is a critical hyperparameter balancing convergence speed against training stability.

---

### Part 3: Batch Size Experiments

Discover how batch size affects gradient quality and update frequency.

**network_1.0.3.py** - Medium Batch (size=32)
- 1,563 updates/epoch (vs baseline 5,000)
- Effect: More stable gradients, better computational efficiency
- Expected: ~95% accuracy, smoother training
- Lesson: Industry standard for good reason

**network_1.0.4.py** - Small Batch (size=5)
- 10,000 updates/epoch (2× baseline)
- Effect: Very frequent updates, high gradient noise
- Expected: ~95% accuracy, erratic curve, better exploration
- Lesson: Noise can help exploration but hurts efficiency

**Key Insight:** Batch size balances gradient stability, update frequency, and computational efficiency.

---

### Part 4: Training Duration Experiments

Learn how training duration affects convergence and overfitting.

**network_1.0.5.py** - Extended Training (60 epochs)
- 2× longer than baseline
- Effect: More time to learn, diminishing returns
- Expected: ~95-96% accuracy (only +0.5-1% gain)
- Lesson: Doubling training time may not double performance

**network_1.0.6.py** - Insufficient Training (10 epochs)
- 1/3 of baseline
- Effect: Underfitting - network stops before converging
- Expected: ~88-92% accuracy (3-7% loss)
- Lesson: Too few epochs leaves performance on the table

**Key Insight:** Find the sweet spot - enough epochs to converge, not so many that you waste time or overfit.

---

### Part 5: Architecture Experiments

Understand how network structure (width vs depth) affects learning.

**network_1.1.0.py** - Adding Depth (2 hidden layers)
- Architecture: `[784, 30, 30, 10]` (24,790 parameters)
- Strategy: Hierarchical feature learning (low→high level)
- Expected: ~95-96% accuracy
- Lesson: Depth enables hierarchy, parameter-efficient

**network_1.2.0.py** - Adding Width (60 neurons)
- Architecture: `[784, 60, 10]` (47,710 parameters)
- Strategy: More parallel feature detectors
- Expected: ~96-97% accuracy
- Lesson: Width provides capacity, avoids vanishing gradients

**network_1.3.0.py** - Width + Depth Combined
- Architecture: `[784, 60, 60, 10]` (51,370 parameters)
- Strategy: Maximum representational power
- Expected: ~97-98% accuracy (BEST!)
- Lesson: Modern networks use BOTH width AND depth

**Key Insight:** Width = parallel capacity, Depth = hierarchical learning, Both = maximum power!

---

## Quick Reference Table

| File | Focus | Key Change | Expected Accuracy | Key Lesson |
|------|-------|------------|-------------------|------------|
| **1.0.0** | Baseline | Standard params | ~95% | Foundation |
| **1.0.1** | Learning rate | η=0.5 (low) | ~85-90% | Slow but stable |
| **1.0.2** | Learning rate | η=5.0 (high) | ~95% (fast) | Fast but risky |
| **1.0.3** | Batch size | size=32 (medium) | ~95% | Balanced approach |
| **1.0.4** | Batch size | size=5 (small) | ~95% | Exploration vs efficiency |
| **1.0.5** | Epochs | 60 (extended) | ~95-96% | Diminishing returns |
| **1.0.6** | Epochs | 10 (short) | ~88-92% | Underfitting |
| **1.1.0** | Architecture | +Depth | ~95-96% | Hierarchy, efficient |
| **1.2.0** | Architecture | +Width | ~96-97% | Capacity, parallel |
| **1.3.0** | Architecture | +Both | ~97-98% | Maximum power |

---

## How to Use This Learning Series

### Option 1: Sequential Learning (Recommended)
Run each file in order to build understanding progressively:

```bash
cd network_1

# Start with the foundation
python network_1.0.0.py

# Explore learning rates
python network_1.0.1.py
python network_1.0.2.py

# Explore batch sizes
python network_1.0.3.py
python network_1.0.4.py

# Explore training duration
python network_1.0.5.py
python network_1.0.6.py

# Explore architectures
python network_1.1.0.py
python network_1.2.0.py
python network_1.3.0.py
```

### Option 2: Focused Learning
Jump to specific topics of interest:

**Want to understand hyperparameters?**
→ Start with 1.0.0, then explore 1.0.1-1.0.6

**Want to understand architectures?**
→ Start with 1.0.0, then jump to 1.1.0-1.3.0

**Want quick overview?**
→ Run 1.0.0, 1.0.1, 1.0.3, 1.0.6, 1.1.0, 1.2.0, 1.3.0

### Option 3: Comparative Benchmark
Run the benchmark script to see statistical comparisons:

```bash
python benchmark_runner.py
```

See the [Benchmark Configuration](#benchmark-configuration) section below for details.

---

## What Each Script Does

Every script in this series:
1. **Loads MNIST data** (50,000 training images, 10,000 test images)
2. **Creates a neural network** with the specified architecture
3. **Trains using SGD** with backpropagation
4. **Displays progress** showing accuracy after each epoch
5. **Provides educational commentary** explaining what's happening and why

### Educational Features

Each file includes:
- **Concise docstring** - Quick overview of the experiment
- **Prerequisites** - Which files to read first
- **Focused commentary** - Deep dive on ONE key concept
- **Comparison notes** - How this relates to the baseline
- **Next steps** - Where to go next in your learning

### Example Output
```
==============================================================
Training Your First Neural Network
==============================================================

1. Loading MNIST data...
   ✓ Loaded 50000 training samples
   ✓ Loaded 10000 test samples

2. Creating network...
   Architecture: [784, 30, 10]
   - Input layer: 784 neurons (28×28 pixel image)
   - Hidden layer: 30 neurons
   - Output layer: 10 neurons (digits 0-9)

3. Training network...
   Epochs: 30
   Mini-batch size: 10
   Learning rate: 3.0

------------------------------------------------------------
Epoch 0: 9015 / 10000
Epoch 1: 9238 / 10000
Epoch 2: 9338 / 10000
...
Epoch 29: 9523 / 10000

Final Accuracy: 95.23%
```

---

## Benchmark Configuration

The `benchmark_runner.py` script allows you to compare multiple networks statistically.

### Quick Configuration

Change these 4 variables at the top of `benchmark_runner.py`:

```python
NUM_TRIALS = 3          # How many times to train each network
EPOCHS = 10             # How many epochs per training run
MINI_BATCH_SIZE = 10    # Batch size for SGD
LEARNING_RATE = 3.0     # Learning rate for SGD
```

### Preset Configurations

**Ultra Quick Test (~5 minutes)** - Test the script
```python
NUM_TRIALS = 1
EPOCHS = 5
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

**Quick Test (~10-15 minutes)** ⭐ **DEFAULT**
```python
NUM_TRIALS = 3
EPOCHS = 10
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

**Standard Benchmark (~30-40 minutes)** - Better statistics
```python
NUM_TRIALS = 5
EPOCHS = 20
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

**Full Benchmark (~85-90 minutes)** - High confidence
```python
NUM_TRIALS = 10
EPOCHS = 30
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

### What the Benchmark Measures

For architecture comparisons (1.0.0, 1.1.0, 1.2.0, 1.3.0):
- **Accuracy metrics**: Mean, std dev, min, max across trials
- **Speed metrics**: Training time per epoch
- **Efficiency metrics**: Accuracy per parameter
- **Rankings**: Best to worst across dimensions

**Note:** The benchmark is designed for architecture comparison. To compare hyperparameters (learning rates, batch sizes, epochs), run the individual files manually and compare results.

---

## Key Learning Outcomes

By completing this series, you will understand:

### Hyperparameter Effects
1. **Learning Rate**: How step size affects convergence speed and stability
2. **Batch Size**: How mini-batch size affects gradient quality and training dynamics
3. **Training Duration**: How epoch count affects convergence and overfitting
4. **Trade-offs**: Speed vs stability, noise vs efficiency, time vs performance

### Architecture Choices
1. **Width**: Adding neurons increases parallel feature capacity
2. **Depth**: Adding layers enables hierarchical feature learning
3. **Combination**: Modern networks use both for maximum power
4. **Trade-offs**: Parameters vs performance, efficiency vs capacity

### Practical Skills
1. **Reading training curves** - Recognizing convergence, overfitting, underfitting
2. **Hyperparameter tuning** - Balancing competing objectives
3. **Architecture design** - Choosing width vs depth based on problem
4. **Performance analysis** - Statistical evaluation and comparison

---

## Fundamental Concepts Covered

### Neural Network Basics (network_1.0.0.py)
- **Architecture**: Layers, neurons, weights, biases
- **Forward pass**: Computing predictions layer by layer
- **Cost function**: Measuring prediction error (quadratic loss)
- **Backpropagation**: Computing gradients efficiently
- **SGD**: Updating parameters to minimize cost
- **Mini-batches**: Balancing computation and gradient quality

### Advanced Concepts (later files)
- **Learning dynamics**: How different hyperparameters affect training
- **Exploration vs exploitation**: Gradient noise and optimization
- **Convergence**: When and why training plateaus
- **Feature learning**: What neurons actually learn
- **Hierarchical representations**: Low-level to high-level features
- **Network capacity**: How architecture affects representational power

---

## Tips for Learners

### First Time Through
1. **Start with 1.0.0** - Don't skip the foundation!
2. **Read all comments** - They explain not just *what* but *why*
3. **Watch the training** - See how accuracy improves epoch by epoch
4. **Experiment** - Try changing hyperparameters and see what happens
5. **Follow the order** - Each file builds on previous knowledge

### Going Deeper
1. **Compare results** - Run experiments side by side
2. **Vary parameters** - Edit the code to test your understanding
3. **Read the code** - See how backpropagation is implemented
4. **Run benchmarks** - Get statistical comparisons
5. **Modify architectures** - Try different layer sizes

### Common Questions

**Q: Why start with network_1.0.0.py?**
A: It provides comprehensive explanation of all core concepts. Later files assume you understand these basics.

**Q: Do I need to run all 10 files?**
A: No! But running them in sequence gives the best learning experience. You can skip experiments that don't interest you.

**Q: Why does training take so long?**
A: Neural networks need many iterations to learn. 30 epochs × 50,000 images = 1.5M training examples! You can reduce epochs for faster experiments.

**Q: Can I modify the code?**
A: Absolutely! Experimentation is encouraged. Try different architectures, learning rates, etc.

**Q: What's the difference between these and network2.py/network3.py?**
A: These files (network_1.x) use basic techniques for education. network2.py and network3.py introduce advanced techniques (cross-entropy, regularization, better initialization, etc.) that significantly improve performance.

---

## Architecture Comparison Summary

| Network | Architecture | Parameters | Strategy | Accuracy | Speed | Best For |
|---------|-------------|------------|----------|----------|-------|----------|
| 1.0.0 | `[784, 30, 10]` | 23,860 | Baseline | ~95% | Fast | Learning & experimentation |
| 1.1.0 | `[784, 30, 30, 10]` | 24,790 (+3.9%) | Depth | ~95-96% | Medium | Parameter efficiency |
| 1.2.0 | `[784, 60, 10]` | 47,710 (+100%) | Width | ~96-97% | Medium-Slow | Accuracy with sigmoid |
| 1.3.0 | `[784, 60, 60, 10]` | 51,370 (+115%) | Both | ~97-98% | Slowest | Maximum accuracy |

### When to Use Each

- **1.0.0 (Baseline)**: Fast experimentation, learning, baseline comparisons
- **1.1.0 (Depth)**: Understanding hierarchical learning, parameter efficiency demonstrations
- **1.2.0 (Width)**: Best practical choice for this problem with sigmoid activation
- **1.3.0 (Width+Depth)**: Maximum accuracy when computational cost isn't a concern

---

## Beyond This Tutorial

After completing this series, you're ready for:

1. **network2.py** - Advanced techniques (cross-entropy, regularization, better initialization)
2. **network3.py** - Modern architectures and techniques
3. **Convolutional networks** - Specialized architectures for images
4. **Modern frameworks** - PyTorch, TensorFlow, JAX

### Why These Techniques Matter

This tutorial uses sigmoid activation and quadratic cost for simplicity, but modern networks use:
- **ReLU activation** - Solves vanishing gradient problem
- **Cross-entropy loss** - Better for classification
- **Batch normalization** - Stabilizes deep network training
- **Regularization** - Prevents overfitting (L2, dropout)
- **Better initialization** - Prevents saturation early in training
- **Adaptive optimizers** - Adam, RMSprop (auto-tune learning rates)

These improvements are covered in network2.py and network3.py!

---

## Credits

This educational series is part of the Neural Networks and Deep Learning book by Michael Nielsen.
The files have been structured to provide a progressive, hands-on learning experience.

**Remember**: The best way to learn is by doing. Run the code, read the comments, experiment with changes, and compare results!

---

**Happy Learning!**
