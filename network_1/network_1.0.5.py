"""
Epochs Experiment: Extended Training (60 epochs)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!

WHAT'S DIFFERENT:
- Baseline: epochs=30
- This version: epochs=60 (2× longer training)

FOCUS: How extended training affects convergence and potential overfitting.
Extended training provides more learning time but risks diminishing returns or overfitting.

Expected: Similar or slightly better accuracy (~95-96%), diminishing returns after epoch 30.

Next: network_1.0.6.py (abbreviated training with 10 epochs)

Run: python network_1.0.5.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Epochs Experiment: Extended Training (60 epochs)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create baseline network: [784, 30, 10] (see network_1.0.0.py)
    print("\n2. Creating network [784, 30, 10]...")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # EXPERIMENT: Extended Training (60 epochs)
    # ========================================================================
    # THE KEY CHANGE: epochs = 60 (vs baseline 30)
    #
    # Baseline: 30 epochs × 5,000 updates = 150,000 total updates
    # This run: 60 epochs × 5,000 updates = 300,000 total updates (2× longer!)
    #
    # WHAT EXTENDED TRAINING DOES:
    #
    # ✓ More time to learn:
    #   - 2× as many parameter updates (300,000 vs 150,000)
    #   - Can find better local minima with extended exploration
    #   - Neurons refine feature detection more thoroughly
    #
    # ✓ Potentially better accuracy:
    #   - May improve beyond 30-epoch baseline
    #   - Useful if network hasn't fully converged by epoch 30
    #   - Can learn subtle patterns missed in shorter training
    #
    # ✗ Diminishing returns:
    #   - Early epochs (1-10): rapid improvement (10% → 85%)
    #   - Middle epochs (11-30): steady gains (85% → 95%)
    #   - Late epochs (31-60): minimal gains (95% → 95.5%)
    #   - Doubling training time may only gain 0.5% accuracy
    #
    # ✗ Risk of overfitting:
    #   - Without regularization, may memorize training data
    #   - Training accuracy continues up, test accuracy plateaus
    #   - Gap between training/test performance widens
    #
    # ✗ Longer training time:
    #   - 2× epochs = 2× wall-clock time
    #   - Resources could be better spent on hyperparameter tuning
    #
    # EXPECTED OUTCOME:
    # - Final accuracy: ~95-96% (small improvement over baseline)
    # - Accuracy plateaus somewhere between epochs 30-50
    # - Gains after epoch 30 likely small (0.5-1%)
    # - 2× training time for modest benefit

    print("\n3. Training with EXTENDED epochs...")
    print("   Epochs: 60 (vs baseline 30)")
    print("   Mini-batch size: 10")
    print("   Learning rate: 3.0")
    print("   Total updates: 300,000 (vs baseline 150,000)")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=60, mini_batch_size=10, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Extended Training
    # ========================================================================
    # During training:
    # - Made 300,000 total updates (2× baseline)
    # - Saw each training example 60 times (vs 30 in baseline)
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~95-96% (small improvement over baseline ~95%)
    # - Accuracy likely plateaued between epochs 30-50
    # - Gains after epoch 30: modest (0.5-1% at most)
    # - Training time: 2× longer for small accuracy gain
    #
    # LESSON: More epochs = diminishing returns
    # - Early epochs: rapid learning (biggest gains)
    # - Middle epochs: steady improvement (good progress)
    # - Late epochs: minimal gains (plateau reached)
    # - 2× training time rarely worth it without regularization
    # - Better strategy: early stopping based on validation performance
    #
    # Next: network_1.0.6.py - see effects of INSUFFICIENT training (10 epochs)

if __name__ == "__main__":
    main()

