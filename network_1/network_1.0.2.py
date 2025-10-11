"""
Learning Rate Experiment: Higher Learning Rate (η=5.0)
======================================================

PREREQUISITE: Read network_1.0.0.py (baseline) and network_1.0.1.py (lower η)!

WHAT'S DIFFERENT FROM BASELINE:
- Baseline (network_1.0.0.py): η=3.0 (standard)
- Previous (network_1.0.1.py): η=0.5 (too slow)
- This version: η=5.0 (aggressive - 1.67× larger than baseline!)

FOCUS: Understand the risks and benefits of aggressive learning rates

ARCHITECTURE (same as baseline):
Input Layer (784 neurons) → Hidden Layer (30 neurons) → Output Layer (10 neurons)

Expected: Fast convergence (~95% in 15-20 epochs) BUT noisy and potentially unstable

Run: python network_1.0.2.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Learning Rate Experiment: η=5.0 (Higher Learning Rate)")
    print("=" * 60)
    print("Comparing three learning rates:")
    print("  network_1.0.1.py: η=0.5 (slow)")
    print("  network_1.0.0.py: η=3.0 (baseline)")
    print("  network_1.0.2.py: η=5.0 (aggressive) ← YOU ARE HERE")
    print("=" * 60)

    # Data loading and network creation (same as baseline)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")

    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10] (same as baseline)")
    net = network.Network([784, 30, 10])
    
    # ========================================================================
    # STEP 3: Train with HIGHER LEARNING RATE
    # ========================================================================
    # EXPERIMENT FOCUS: Aggressive Learning Rate (η=5.0)
    #
    # Comparing three learning rates:
    # - network_1.0.1.py: η=0.5 (6× smaller than baseline - TOO SLOW)
    # - network_1.0.0.py: η=3.0 (baseline - BALANCED)
    # - network_1.0.2.py: η=5.0 (1.67× larger than baseline - AGGRESSIVE)
    #
    # EXPECTED EFFECTS OF HIGHER LEARNING RATE (η=5.0):
    #
    # ✓ FASTER CONVERGENCE
    #   - Large steps = rapid progress
    #   - May reach ~95% in only 15-20 epochs (vs 25-30 with baseline)
    #
    # ✗ NOISIER LEARNING
    #   - Accuracy fluctuates wildly between epochs
    #   - Training curve looks jagged and erratic
    #   - May temporarily decrease before improving again
    #
    # ✗ RISK OF INSTABILITY
    #   - Can overshoot optimal values
    #   - May diverge with unlucky initialization
    #   - Harder to fine-tune in later epochs
    #
    # ✗ SATURATION PROBLEMS
    #   - Large updates can push neurons into saturation (σ(z) near 0 or 1)
    #   - When saturated, gradients ≈ 0, causing "dead neurons"
    #
    # Trade-off: Speed vs stability - high learning rates are fast but risky!
    print("\n3. Training with HIGHER learning rate...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 5.0 *** 1.67× LARGER than baseline ***")
    print("\n   Expected: Fast but noisy/unstable learning")
    print("-" * 60)
    
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=5.0, test_data=test_data)

    # ========================================================================
    # RESULTS: HIGHER LEARNING RATE
    # ========================================================================
    #
    # LEARNING RATE COMPARISON SUMMARY:
    #
    # η=0.5 (network_1.0.1.py): SLOW but STABLE
    # - ~85-90% accuracy after 30 epochs
    # - Smooth learning curve
    # - Safe, no divergence risk
    #
    # η=3.0 (network_1.0.0.py): BALANCED ← BEST CHOICE
    # - ~95% accuracy after 30 epochs
    # - Reasonable convergence speed
    # - Good stability
    #
    # η=5.0 (this experiment): FAST but RISKY
    # - ~95% accuracy in 15-20 epochs (faster!)
    # - Noisy, erratic learning curve
    # - Risk of overshooting and instability
    #
    # KEY LESSON: Choosing learning rate involves trade-offs!
    # - Modern solution: Learning rate scheduling (start high, decay over time)
    # - Or adaptive methods like Adam (adjust η automatically per parameter)
    #
    # Next: Explore batch size effects in network_1.0.3.py!

if __name__ == "__main__":
    main()
