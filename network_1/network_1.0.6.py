"""
Epochs Experiment: Insufficient Training (10 epochs)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!
ALSO SEE: network_1.0.5.py (extended training with 60 epochs) for comparison.

WHAT'S DIFFERENT:
- Extended: epochs=60 (2× baseline)
- Baseline: epochs=30
- This version: epochs=10 (1/3 baseline)

FOCUS: How insufficient training leads to underfitting.
Stopping too early prevents the network from reaching its full potential.

Expected: Lower accuracy (~88-92% vs ~95%), network stops before converging.

Next: network_1.1.0.py (architecture experiment: adding depth)

Run: python network_1.0.6.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Epochs Experiment: Insufficient Training (10 epochs)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create baseline network: [784, 30, 10] (see network_1.0.0.py)
    print("\n2. Creating network [784, 30, 10]...")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # EXPERIMENT: Insufficient Training (10 epochs)
    # ========================================================================
    # THE KEY CHANGE: epochs = 10 (vs baseline 30, vs extended 60)
    #
    # Extended: 60 epochs × 5,000 updates = 300,000 total updates
    # Baseline: 30 epochs × 5,000 updates = 150,000 total updates
    # This run: 10 epochs × 5,000 updates = 50,000 total updates (1/3 baseline!)
    #
    # WHAT INSUFFICIENT TRAINING DOES:
    #
    # ✓ Faster training:
    #   - 1/3 the time (3× faster than baseline)
    #   - Good for rapid prototyping and hyperparameter search
    #   - Quick feedback for testing ideas
    #
    # ✓ Less overfitting risk:
    #   - Network has less time to memorize training data
    #   - Training/test gap remains small
    #
    # ✗ UNDERFITTING (main problem):
    #   - Network hasn't converged - still learning when stopped
    #   - Accuracy much lower than potential (88-92% vs 95%)
    #   - Leaving 3-7% performance on the table
    #   - Loss still decreasing significantly at epoch 10
    #
    # ✗ Incomplete learning:
    #   - Hidden neurons haven't fully refined feature detection
    #   - Learns coarse patterns but misses subtle features
    #   - Weights still changing rapidly - not stabilized
    #   - Only 1/3 the optimization steps of baseline
    #
    # ✗ Unused capacity:
    #   - Network has 23,860 parameters but insufficient time to tune them
    #   - Could achieve much better performance with more training
    #
    # ✗ Misleading evaluation:
    #   - Can't tell if low accuracy is due to bad architecture or insufficient training
    #   - Always train sufficiently before judging architecture
    #
    # TYPICAL LEARNING CURVE:
    # - Epochs 1-5: Rapid improvement (10% → 80%)
    # - Epochs 6-10: Steady gains (80% → 90%)
    # - Training stops while still improving significantly!
    # - With 30 epochs, would reach ~95%
    #
    # EXPECTED OUTCOME:
    # - Final accuracy: ~88-92% (vs baseline ~95%)
    # - Accuracy still climbing strongly at epoch 10
    # - 3× faster but sacrifices 3-7% accuracy
    # - Clear underfitting - needs more epochs

    print("\n3. Training with INSUFFICIENT epochs...")
    print("   Epochs: 10 (vs baseline 30, vs extended 60)")
    print("   Mini-batch size: 10")
    print("   Learning rate: 3.0")
    print("   Total updates: 50,000 (vs baseline 150,000)")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Insufficient Training
    # ========================================================================
    # During training:
    # - Made 50,000 total updates (1/3 baseline)
    # - Saw each training example only 10 times (vs 30 in baseline)
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~88-92% (vs baseline ~95%)
    # - Accuracy still improving strongly at epoch 10 (not converged)
    # - Sacrificed 3-7% accuracy for 3× faster training
    # - Learning curve shows no plateau - needs more training
    #
    # LESSON: Insufficient epochs = underfitting
    # - Network has capacity but not enough training time
    # - Easy fix: train longer (unlike overfitting)
    # - 10 epochs good for prototyping, not final models
    # - Always ensure convergence before evaluating architecture
    #
    # EPOCH COMPARISON SUMMARY:
    # - 10 epochs: ~90% (underfit, too few)
    # - 30 epochs: ~95% (optimal for this problem)
    # - 60 epochs: ~95.5% (diminishing returns, slight gain)
    #
    # Next: network_1.1.0.py - experiment with ARCHITECTURE (adding depth)

if __name__ == "__main__":
    main()

