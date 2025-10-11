"""
Learning Rate Experiment: Lower Learning Rate (η=0.5)
=====================================================

PREREQUISITE: Read network_1.0.0.py first to understand the baseline!

WHAT'S DIFFERENT FROM BASELINE (network_1.0.0.py):
- Baseline uses learning rate η=3.0 (standard)
- This version uses η=0.5 (much smaller - 6× reduction)
- All other hyperparameters remain the same

FOCUS OF THIS EXPERIMENT:
Learn how learning rate affects:
- Training speed (convergence rate)
- Stability (smoothness of learning curve)
- Final accuracy achieved

ARCHITECTURE (same as baseline):
Input Layer (784 neurons) → Hidden Layer (30 neurons) → Output Layer (10 neurons)

Expected accuracy: ~85-90% after 30 epochs (slower learning due to small steps)

Run: python network_1.0.1.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Learning Rate Experiment: η=0.5 (Lower Learning Rate)")
    print("=" * 60)
    print("Compare with network_1.0.0.py (η=3.0) to see the difference!")
    print("=" * 60)

    # ========================================================================
    # STEP 1: Load MNIST data (same as baseline)
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(test_data)} test samples")

    # ========================================================================
    # STEP 2: Create the neural network (same architecture as baseline)
    # ========================================================================
    # Architecture: [784, 30, 10] - identical to baseline
    # For architecture details, see network_1.0.0.py
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10] (same as baseline)")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # STEP 3: Train with LOWER LEARNING RATE
    # ========================================================================
    # EXPERIMENT FOCUS: Learning Rate (η)
    #
    # Baseline (network_1.0.0.py): η = 3.0
    # This experiment: η = 0.5 (6× smaller!)
    #
    # What is learning rate?
    #   - Controls step size in gradient descent: w → w - η·∇w
    #   - Think of it as "how far to step" when adjusting weights
    #
    # EXPECTED EFFECTS OF LOWER LEARNING RATE (η=0.5):
    #
    # ✓ SLOWER CONVERGENCE
    #   - Smaller steps = slower progress toward optimal weights
    #   - May reach only ~85-90% accuracy after 30 epochs (vs ~95% with η=3.0)
    #   - Would need ~180 epochs to match baseline performance
    #
    # ✓ SMOOTHER LEARNING
    #   - More predictable, less erratic accuracy improvements
    #   - Training curve looks cleaner with less epoch-to-epoch variation
    #
    # ✓ BETTER STABILITY
    #   - Reduced risk of overshooting optimal values
    #   - More robust to noisy gradients
    #
    # ✓ FINER OPTIMIZATION
    #   - Can settle into better local minima with precise adjustments
    #
    # Trade-off: Stability and precision vs speed
    # Choosing learning rate is about balancing these factors!
    #
    # Other hyperparameters (same as baseline):
    # - Epochs: 30
    # - Mini-batch size: 10
    print("\n3. Training network with LOWER learning rate...")
    print("   Epochs: 30 (same as baseline)")
    print("   Mini-batch size: 10 (same as baseline)")
    print("   Learning rate: 0.5 *** 6× SMALLER than baseline (3.0) ***")
    print("\n   Expected: Slower but smoother learning")
    print("-" * 60)

    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.5, test_data=test_data)

    # ========================================================================
    # RESULTS: LOWER LEARNING RATE
    # ========================================================================
    #
    # Compare your results with baseline (network_1.0.0.py):
    #
    # With η=0.5 (this experiment):
    # - SLOWER convergence: likely ~85-90% accuracy after 30 epochs
    # - SMOOTHER learning curve: less fluctuation between epochs
    # - MORE STABLE: no risk of overshooting or divergence
    #
    # With η=3.0 (baseline):
    # - FASTER convergence: ~95% accuracy after 30 epochs
    # - MORE VOLATILE: larger fluctuations in accuracy
    # - Some risk of instability with large updates
    #
    # KEY LESSON: Learning rate is a speed-stability trade-off!
    # - Too small → safe but slow (this experiment)
    # - Too large → fast but risky (can diverge)
    # - Just right → fast AND stable (baseline η=3.0 works well for this problem)
    #
    # Next experiments:
    # - Try network_1.0.2.py for HIGHER learning rate (η=5.0) - see instability!
    # - Continue to network_1.0.3.py to explore batch size effects

if __name__ == "__main__":
    main()
