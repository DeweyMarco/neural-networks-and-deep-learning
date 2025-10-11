"""
Batch Size Experiment: Small Batch (size=5)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!
ALSO SEE: network_1.0.3.py (medium batch size=32) for comparison.

WHAT'S DIFFERENT:
- Baseline: mini_batch_size=10 (5,000 updates/epoch)
- Medium batch: mini_batch_size=32 (1,563 updates/epoch)
- This version: mini_batch_size=5 (10,000 updates/epoch)

FOCUS: How small batch sizes provide frequent updates and high gradient noise.
Small batches update parameters very frequently but with noisy gradients.

Expected: Similar ~95% accuracy, more erratic training curve, better exploration.

Next: network_1.0.5.py (extended training with 60 epochs)

Run: python network_1.0.4.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Batch Size Experiment: Small Batch (size=5)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create baseline network: [784, 30, 10] (see network_1.0.0.py)
    print("\n2. Creating network [784, 30, 10]...")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # EXPERIMENT: Small Batch Size (5)
    # ========================================================================
    # THE KEY CHANGE: mini_batch_size = 5 (vs baseline 10, vs medium 32)
    #
    # Batch size 32 (medium): 50,000 ÷ 32 ≈ 1,563 updates/epoch
    # Batch size 10 (baseline): 50,000 ÷ 10 = 5,000 updates/epoch
    # Batch size 5 (this run): 50,000 ÷ 5 = 10,000 updates/epoch
    #
    # WHAT SMALL BATCHES DO:
    #
    # ✓ Very frequent updates:
    #   - 10,000 updates/epoch vs 5,000 (baseline) or 1,563 (medium)
    #   - Parameters adjust 2× more often than baseline, 6.4× more than medium
    #   - Each mini-batch provides slightly different gradient estimate
    #
    # ✓ High gradient noise (implicit regularization):
    #   - Only 5 samples = high variance in gradient estimates
    #   - Noise helps escape sharp minima, explore loss landscape
    #   - Can lead to flatter minima that generalize better
    #   - Acts as regularization even without explicit techniques
    #
    # ✓ Better exploration:
    #   - Training trajectory is more stochastic and exploratory
    #   - Can help avoid poor local optima
    #   - May discover better solutions than larger batches
    #
    # ✗ Unstable/erratic convergence:
    #   - Training curves appear "jagged" with high epoch-to-epoch variance
    #   - May oscillate around optimum without settling smoothly
    #   - Harder to determine when training has converged
    #
    # ✗ Slower computation:
    #   - Cannot leverage vectorization as well as larger batches
    #   - More loop overhead (10,000 updates vs 1,563)
    #   - Slower wall-clock time per epoch despite same total samples
    #
    # ✗ Potential instability:
    #   - High variance gradients + large learning rate (3.0) can cause issues
    #   - May need to reduce learning rate for very small batches
    #
    # EXPECTED OUTCOME:
    # - Similar final accuracy (~95%) to baseline and medium batch
    # - More erratic, noisy training curve
    # - Potentially better generalization (regularization effect)
    # - Slower per-epoch time

    print("\n3. Training with SMALL batch size...")
    print("   Epochs: 30")
    print("   Mini-batch size: 5 (vs baseline 10, vs medium 32)")
    print("   Learning rate: 3.0")
    print("   Updates per epoch: 10,000 (vs baseline 5,000, vs medium 1,563)")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=30, mini_batch_size=5, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Small Batch Size
    # ========================================================================
    # During training:
    # - Made 10,000 updates/epoch × 30 epochs = 300,000 total updates!
    # - Compare to: medium (46,890), baseline (150,000), this (300,000)
    # - 2× more updates than baseline, 6.4× more than medium
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~95% (similar to other batch sizes)
    # - Training curve: erratic and noisy, high epoch-to-epoch variance
    # - Per-epoch time: slower due to poor vectorization
    # - Exploration: better stochastic exploration of loss landscape
    #
    # LESSON: Small batches trade stability for exploration
    # - Frequent updates + high noise = better exploration
    # - Can find flatter minima (better generalization)
    # - But: noisy training, slower computation, harder to tune
    # - For this problem, medium batches (32) offer better balance
    #
    # Next: network_1.0.5.py - see effects of EXTENDED training (60 epochs)

if __name__ == "__main__":
    main()
