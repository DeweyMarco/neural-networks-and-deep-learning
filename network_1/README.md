# Network Architecture Comparison

This folder contains 4 different neural network architectures demonstrating the effects of **WIDTH** vs **DEPTH** on MNIST digit classification.

## Network Variants

| Network | Architecture | Parameters | Strategy | Expected Accuracy |
|---------|-------------|------------|----------|------------------|
| **network_1.0.py** | `[784, 30, 10]` | 23,860 | Baseline (narrow + shallow) | ~95% |
| **network_1.1.py** | `[784, 30, 30, 10]` | 24,790 | Depth (narrow + deep) | ~95-96% |
| **network_1.2.py** | `[784, 60, 10]` | 47,710 | Width (wide + shallow) | ~96-97% |
| **network_1.3.py** | `[784, 60, 60, 10]` | 51,370 | Width + Depth (wide + deep) | ~97-98% |

---

## Running Individual Networks

To run any single network:

```bash
cd network_1
python network_1.0.py  # or network_1.1.py, network_1.2.py, network_1.3.py
```

Each script will:
- Load MNIST data
- Train the network for 30 epochs
- Display accuracy after each epoch
- Provide detailed educational commentary

---

## Running the Benchmark

To compare all 4 networks with statistical analysis:

```bash
cd network_1
python benchmark_runner.py
```

The benchmark runner will:
- Run each network multiple times (configurable via `NUM_TRIALS`)
- Track accuracy and training time for each run
- Calculate mean, std dev, min, and max for all metrics
- Display comprehensive comparison tables on screen

---

## Configuring the Benchmark

### Quick Start

**It's easy!** Just change 4 variables at the top of `benchmark_runner.py`:

```python
NUM_TRIALS = 3          # How many times to train each network
EPOCHS = 10             # How many epochs per training run
MINI_BATCH_SIZE = 10    # Batch size for SGD
LEARNING_RATE = 3.0     # Learning rate for SGD
```

**That's it!** Everything else adapts automatically:
- ✅ Timing estimates
- ✅ Progress displays
- ✅ Statistics
- ✅ Output messages

### Configuration Variables Explained

#### `NUM_TRIALS`
**What it does:** Number of times to train each network architecture  
**Why it matters:** More trials = better statistical confidence  
**Typical values:** 1-20  
**Default:** 3 (quick testing), 10 (full benchmark)

**Example:**
```python
NUM_TRIALS = 5  # Each network will be trained 5 times
```

#### `EPOCHS`
**What it does:** Number of complete passes through training data per trial  
**Why it matters:** More epochs = higher accuracy but longer training  
**Typical values:** 5-30  
**Default:** 10 (quick testing), 30 (full benchmark)

**Example:**
```python
EPOCHS = 20  # Each trial trains for 20 epochs
```

#### `MINI_BATCH_SIZE`
**What it does:** Number of training examples to process before updating weights  
**Why it matters:** Smaller = noisier gradients but faster convergence  
**Typical values:** 1-128  
**Default:** 10 (good for MNIST)

**Example:**
```python
MINI_BATCH_SIZE = 32  # Update weights after every 32 examples
```

#### `LEARNING_RATE`
**What it does:** How big of a step to take when updating weights  
**Why it matters:** Too high = unstable training, too low = slow learning  
**Typical values:** 0.01-5.0 for sigmoid networks  
**Default:** 3.0 (works well for these networks)

**Example:**
```python
LEARNING_RATE = 1.0  # Smaller steps, more conservative learning
```

### Preset Configurations

Copy and paste one of these into `benchmark_runner.py`:

#### Ultra Quick Test (~5 minutes)
Perfect for testing the script or quick experiments
```python
NUM_TRIALS = 1
EPOCHS = 5
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

#### Quick Test (~10-15 minutes) ⭐ **CURRENT DEFAULT**
Good balance for quick insights
```python
NUM_TRIALS = 3
EPOCHS = 10
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

#### Standard Benchmark (~30-40 minutes)
Better statistics, reasonable accuracy
```python
NUM_TRIALS = 5
EPOCHS = 20
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

#### Full Benchmark (~85-90 minutes)
High statistical confidence, near-peak accuracy
```python
NUM_TRIALS = 10
EPOCHS = 30
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

#### Production-Quality Benchmark (~3 hours)
Maximum statistical rigor for research/publication
```python
NUM_TRIALS = 20
EPOCHS = 30
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

### Examples of Custom Configurations

**Test a specific learning rate:**
```python
NUM_TRIALS = 5
EPOCHS = 20
MINI_BATCH_SIZE = 10
LEARNING_RATE = 0.5  # Testing slower learning
```

**Test different batch sizes:**
```python
NUM_TRIALS = 3
EPOCHS = 15
MINI_BATCH_SIZE = 50  # Larger batches
LEARNING_RATE = 3.0
```

**Maximum accuracy (long runtime!):**
```python
NUM_TRIALS = 10
EPOCHS = 50  # Train longer
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

**Quick comparison for demos:**
```python
NUM_TRIALS = 1  # No statistics, just one run
EPOCHS = 3      # Very quick
MINI_BATCH_SIZE = 10
LEARNING_RATE = 3.0
```

---

## Benchmark Duration

**The script automatically estimates runtime based on your configuration!**

Approximate times for different configurations:

| Configuration | NUM_TRIALS | EPOCHS | Estimated Time |
|--------------|------------|--------|----------------|
| Ultra Quick | 1 | 5 | ~5 minutes |
| Quick Test | 3 | 10 | ~10-15 minutes |
| Standard | 5 | 20 | ~30-40 minutes |
| Full Benchmark | 10 | 30 | ~85-90 minutes |
| Production | 20 | 30 | ~3 hours |

**Timing factors:**
- Wider networks (more neurons) train slower
- Deeper networks (more layers) train slower
- More epochs = proportionally longer
- More trials = proportionally longer

The benchmark runner shows an estimate **before** starting!

---

## What the Benchmark Measures

### Accuracy Metrics
- Final test accuracy (%)
- Mean and standard deviation across trials
- Best and worst performance

### Speed Metrics
- Total training time
- Average time per epoch
- Training efficiency

### Derived Metrics
- Parameter efficiency (accuracy per 1000 parameters)
- Balanced score (accuracy vs speed trade-off)
- Rankings across all dimensions

---

## Example Output

Example with `NUM_TRIALS=10, EPOCHS=30` (full benchmark):

```
COMPREHENSIVE BENCHMARK COMPARISON
==================================================================================

Network         Architecture         Params     Accuracy             Speed (s/epoch)     
----------------------------------------------------------------------------------
network_1.0     [784, 30, 10]       23,860     95.23% ± 0.15%       1.42s ± 0.03s       
network_1.1     [784, 30, 30, 10]   24,790     95.87% ± 0.18%       1.58s ± 0.04s       
network_1.2     [784, 60, 10]       47,710     96.45% ± 0.12%       2.31s ± 0.05s       
network_1.3     [784, 60, 60, 10]   51,370     97.12% ± 0.14%       2.67s ± 0.06s       
----------------------------------------------------------------------------------

RANKINGS

By Final Accuracy (highest to lowest):
  1. network_1.3      97.12%  (Width + Depth (wide + deep))
  2. network_1.2      96.45%  (Width (wide + shallow))
  3. network_1.1      95.87%  (Depth (narrow + deep))
  4. network_1.0      95.23%  (Narrow + Shallow)

By Training Speed (fastest to slowest):
  1. network_1.0      1.42s/epoch  (Narrow + Shallow)
  2. network_1.1      1.58s/epoch  (Depth (narrow + deep))
  3. network_1.2      2.31s/epoch  (Width (wide + shallow))
  4. network_1.3      2.67s/epoch  (Width + Depth (wide + deep))
```

### Understanding the Output

**During Run:**
```
Trial 1/3... Done! Accuracy: 95.23%, Time: 14.2s
```
- Shows progress (Trial 1 of 3 total)
- Displays final accuracy for this trial
- Shows total time for this trial

**Summary Statistics:**
```
Final Accuracy: 95.23% ± 0.15%
(min: 95.05%, max: 95.38%)
```
- Mean ± standard deviation across all trials
- Range shows best and worst performance

**Timing:**
```
Actual time: 12.3 minutes
Estimated time: 12.5 minutes
✓ Time estimate was accurate!
```
- Compares actual vs estimated time
- Helps you plan future runs

---

## Key Insights

### Width vs Depth Trade-offs

#### WIDTH (network_1.2)
- ✅ More parallel feature detectors
- ✅ Higher representational capacity
- ✅ Less prone to vanishing gradients
- ❌ More parameters = slower training
- ❌ No hierarchical features

#### DEPTH (network_1.1)
- ✅ Hierarchical feature learning
- ✅ Parameter-efficient (+3.9% params vs baseline)
- ✅ Compositional representations
- ❌ Vanishing gradient issues with sigmoid
- ❌ Limited capacity per layer

#### WIDTH + DEPTH (network_1.3)
- ✅ Best accuracy (~97-98%)
- ✅ Combines benefits of both strategies
- ✅ Rich hierarchical features
- ❌ Most parameters (+115% vs baseline)
- ❌ Slowest training
- ❌ Higher overfitting risk

### When to Use Each

- **network_1.0**: Fast experimentation, baseline comparison
- **network_1.1**: Learning about depth with minimal overhead
- **network_1.2**: Best accuracy/speed balance for production
- **network_1.3**: Maximum accuracy when compute isn't constrained

---

## Tips and Best Practices

1. **Start small**: Use `NUM_TRIALS=1, EPOCHS=5` to test changes
2. **For serious comparisons**: Use at least `NUM_TRIALS=5, EPOCHS=20`
3. **For publications**: Use `NUM_TRIALS=10, EPOCHS=30` minimum
4. **Save your configs**: Comment out different presets you've tried
5. **Copy results**: Save terminal output if you need to reference results later

### Troubleshooting

**"Taking too long!"**  
→ Reduce `NUM_TRIALS` and `EPOCHS`

**"Results are inconsistent"**  
→ Increase `NUM_TRIALS` for better statistics

**"Networks not learning well"**  
→ Try different `LEARNING_RATE` values (try 1.0 or 5.0)

**"Want faster training"**  
→ Increase `MINI_BATCH_SIZE` (but may hurt accuracy slightly)

---

## Educational Value

This comparison demonstrates fundamental deep learning concepts:

1. **Architecture design choices** (width vs depth)
2. **Performance trade-offs** (accuracy vs speed vs parameters)
3. **Statistical evaluation** (why multiple runs matter)
4. **Practical considerations** (efficiency, scalability, production constraints)

### Key Takeaways

**Width provides CAPACITY:**
- More neurons = more parallel feature detectors
- Can represent more diverse patterns simultaneously
- Better raw performance but costs parameters

**Depth provides HIERARCHY:**
- More layers = compositional feature learning
- Low-level → high-level feature progression
- More parameter-efficient but harder to optimize

**Width + Depth provides MAXIMUM POWER:**
- Combines benefits: diverse hierarchical features
- Best accuracy but highest computational cost
- This is why modern networks are BOTH wide AND deep!

**Modern Deep Learning:**
Modern architectures (ResNet, Transformers, etc.) use **both width and depth** with advanced techniques to overcome the limitations demonstrated here:
- ReLU activations (instead of sigmoid)
- Batch normalization
- Residual connections
- Attention mechanisms

These techniques address vanishing gradients and training instability issues present in these basic networks.

---

## Advanced: Modifying the Benchmark

The benchmark script rarely needs changes beyond the 4 configuration variables. However, if you want to customize further:

- `NETWORK_CONFIGS` - The 4 architectures being tested
- `BenchmarkNetwork` class - How metrics are tracked
- Output formatting - Tables and rankings

Only modify these if you want to:
- Add new network architectures
- Change what metrics are tracked
- Customize the output format
- Add new analysis features

---

**Remember:** Just change the 4 variables, everything else adapts! 🎯
