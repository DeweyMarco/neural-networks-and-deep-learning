"""
Batch Size Experiment: Medium Batch (size=32)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!

WHAT'S DIFFERENT:
- Baseline: mini_batch_size=10 (5,000 updates/epoch)
- This version: mini_batch_size=32 (1,563 updates/epoch)

FOCUS: How medium batch sizes balance gradient stability and update frequency.
Medium batches provide stable gradients while maintaining reasonable update frequency.

Expected: Similar ~95% accuracy, slightly smoother training curve, faster per-epoch time.

Next: network_1.0.4.py (small batch size=5)

Run: python network_1.0.3.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Batch Size Experiment: Medium Batch (size=32)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create baseline network: [784, 30, 10] (see network_1.0.0.py)
    print("\n2. Creating network [784, 30, 10]...")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # EXPERIMENT: Medium Batch Size (32)
    # ========================================================================
    # THE KEY CHANGE: mini_batch_size = 32 (vs baseline 10)
    #
    # Batch size 10 (baseline): 50,000 ÷ 10 = 5,000 updates/epoch
    # Batch size 32 (this run): 50,000 ÷ 32 ≈ 1,563 updates/epoch
    #
    # WHAT MEDIUM BATCHES DO:
    #
    # ✓ More stable gradients:
    #   - Averaging over 32 samples reduces gradient noise
    #   - Smoother training curves, less epoch-to-epoch variation
    #   - More consistent convergence trajectory
    #
    # ✓ Better computational efficiency:
    #   - Can leverage vectorization/parallelization better than small batches
    #   - Faster wall-clock time per epoch on modern hardware
    #   - Good GPU utilization
    #
    # ✓ Balanced approach:
    #   - Industry standard (32-64) for good reason
    #   - Works well across diverse problems without tuning
    #   - Still enough updates per epoch for good convergence
    #
    # ✗ Fewer weight updates:
    #   - 1,563 updates vs 5,000 with batch size 10
    #   - Each epoch makes 3.2× fewer parameter adjustments
    #   - May need slightly more epochs to fully converge
    #
    # ✗ Less stochastic exploration:
    #   - Reduced gradient noise = less exploration of loss landscape
    #   - May have slightly weaker implicit regularization
    #   - Small difference for this problem
    #
    # EXPECTED OUTCOME:
    # - Similar final accuracy (~95%) to baseline
    # - Smoother, less noisy training curve
    # - Slightly faster per-epoch time
    # - Good balance of stability and update frequency

    print("\n3. Training with MEDIUM batch size...")
    print("   Epochs: 30")
    print("   Mini-batch size: 32 (vs baseline 10)")
    print("   Learning rate: 3.0")
    print("   Updates per epoch: ~1,563 (vs baseline 5,000)")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=30, mini_batch_size=32, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Medium Batch Size
    # ========================================================================
    # During training:
    # - Made ~1,563 updates/epoch × 30 epochs ≈ 46,890 total updates
    # - Compare to baseline: 5,000 updates/epoch × 30 = 150,000 updates
    # - 3.2× fewer updates but similar accuracy!
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~95% (similar to baseline size=10)
    # - Training curve: smoother, less noisy epoch-to-epoch
    # - Per-epoch time: slightly faster due to better vectorization
    # - Convergence: stable and predictable
    #
    # LESSON: Medium batches (32) are standard because they work!
    # - Good balance of gradient stability and update frequency
    # - Efficient computation, smooth training
    # - Minimal accuracy trade-off vs smaller batches
    #
    # Next: network_1.0.4.py - see what happens with SMALL batches (size=5)

if __name__ == "__main__":
    main()
