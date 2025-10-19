"""
Quadratic Cost Function - Understanding the Learning Slowdown Problem
======================================================================

PREREQUISITES:
- Read network_1.0.0.py to understand quadratic cost basics
- Read network_2.0.0.py to understand cross-entropy cost

THIS EXPERIMENT:
Uses network2 architecture BUT with the OLD quadratic cost function.

This demonstrates WHY cross-entropy cost was invented!

KEY INSIGHT - THE LEARNING SLOWDOWN PROBLEM:
When a neuron is very wrong but saturated (confident in its wrong answer),
the quadratic cost causes learning to be VERY SLOW.

MATHEMATICAL EXPLANATION:

Quadratic Cost: C = (1/2n) Σ ||a - y||²

Gradient for quadratic cost:
∂C/∂w = (a - y) · σ'(z) · x

The problem: σ'(z) term!

σ'(z) = σ(z) · (1 - σ(z))

When neuron is saturated:
- If a ≈ 0 (very confident it's 0): σ'(z) ≈ 0
- If a ≈ 1 (very confident it's 1): σ'(z) ≈ 0

Result: Even when (a - y) is LARGE (very wrong!), gradient is TINY!
This means: SLOW LEARNING when you need FAST LEARNING!

Cross-Entropy Cost: C = -[y·ln(a) + (1-y)·ln(1-a)]

Gradient for cross-entropy:
∂C/∂w = (a - y) · x

No σ'(z) term! Learning speed proportional to error size!

COMPARISON:
- Quadratic cost:     ~94-95% accuracy, SLOW initial learning
- Cross-entropy cost: ~96-97% accuracy, FAST initial learning

WATCH FOR:
- Slower convergence in early epochs
- Lower final accuracy than cross-entropy
- Network "stuck" when making confident mistakes

Expected Results:
- Final accuracy: ~94-95% (1-2% worse than cross-entropy)
- Slower learning in first 5-10 epochs
- Some neurons may saturate and stop learning

WHAT'S NEXT:
- network_2.1.1.py: Direct side-by-side comparison
- network_2.0.0.py: Cross-entropy (THE SOLUTION!)

Run: python network_2.1.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: Demonstrating the learning slowdown problem
    # ========================================================================
    print("=" * 60)
    print("Neural Network with QUADRATIC Cost Function")
    print("=" * 60)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create network with QUADRATIC COST
    # ========================================================================
    # KEY DIFFERENCE: Using QuadraticCost instead of CrossEntropyCost
    #
    # QuadraticCost is the same cost function used in network_1.x series
    # (the basic neural network from chapter 1)
    #
    # We're using network2 architecture to ensure a fair comparison:
    # - Same code structure
    # - Same monitoring capabilities
    # - ONLY difference is the cost function
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10]")
    print("   Cost function: QUADRATIC (Mean Squared Error)")
    print("   Regularization: λ=5.0")
    net = network2.Network([784, 30, 10], cost=network2.QuadraticCost)

    # ========================================================================
    # STEP 3: Train with quadratic cost
    # ========================================================================
    # THE LEARNING SLOWDOWN PROBLEM IN DETAIL:
    #
    # Scenario: Neuron should output 1, but outputs 0.1 (very wrong!)
    #
    # Quadratic cost gradient:
    #   ∂C/∂w = (a - y) · σ'(z) · x
    #   ∂C/∂w = (0.1 - 1) · σ'(z) · x
    #   ∂C/∂w = -0.9 · σ'(z) · x
    #
    # If a = 0.1, then z is very negative (neuron is saturated)
    # σ'(z) = σ(z) · (1 - σ(z)) = 0.1 · 0.9 = 0.09
    #
    # So: ∂C/∂w = -0.9 · 0.09 · x = -0.081 · x
    #
    # Cross-entropy gradient (for comparison):
    #   ∂C/∂w = (a - y) · x
    #   ∂C/∂w = (0.1 - 1) · x = -0.9 · x
    #
    # DIFFERENCE: Cross-entropy gradient is 10× LARGER!
    # (-0.9 vs -0.081)
    #
    # This means cross-entropy learns 10× FASTER when making big mistakes!
    #
    # The worse the mistake, the bigger the difference:
    # If a = 0.01 (even more wrong):
    # - Quadratic: ∂C/∂w = -0.99 · 0.0099 · x = -0.0098 · x (tiny!)
    # - Cross-entropy: ∂C/∂w = -0.99 · x (large!)
    # - Ratio: 100× difference!
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0")
    print("\n" + "-" * 60)
    # THE LEARNING SLOWDOWN PROBLEM:
    # When neurons are saturated (confident but wrong),
    # the gradient ∂C/∂w = (a-y)·σ'(z)·x becomes TINY
    # because σ'(z) ≈ 0 for saturated neurons.
    # Watch how learning is SLOW in early epochs!
    print("-" * 60 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=5.0,
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Analyze the learning slowdown
    # ========================================================================
    print("\n" + "=" * 60)
    print("LEARNING SLOWDOWN ANALYSIS")
    print("=" * 60)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")

    # Analyze early vs late epoch improvement
    if len(evaluation_accuracy) >= 10:
        early_improvement = evaluation_accuracy[4] - evaluation_accuracy[0]  # Epochs 0-4
        late_improvement = evaluation_accuracy[-1] - evaluation_accuracy[-5]  # Last 5 epochs

        print("\n" + "-" * 60)
        print("LEARNING SPEED ANALYSIS:")
        print("-" * 60)
        print(f"Early improvement (epochs 0-4):  +{early_improvement} correct")
        print(f"Late improvement (last 5 epochs): +{late_improvement} correct")

        if early_improvement < late_improvement:
            print("\n⚠ LEARNING SLOWDOWN DETECTED!")
            print("   Early epochs improved LESS than later epochs")
            print("   This is backwards! Should learn FAST at first, slow later")
        else:
            print("\n✓ Normal learning progression")

    # COST FUNCTION COMPARISON:
    # 
    # Quadratic Cost:
    #   Formula:   C = (1/2n)·Σ||a - y||²
    #   Gradient:  ∂C/∂w = (a - y)·σ'(z)·x
    #   Problem:   σ'(z) ≈ 0 when neuron saturated → SLOW learning
    print(f"   Quadratic Cost Accuracy:  ~{final_validation_acc:.1f}%")
    # 
    # Cross-Entropy Cost:
    #   Formula:   C = -[y·ln(a) + (1-y)·ln(1-a)]
    #   Gradient:  ∂C/∂w = (a - y)·x
    #   Advantage: NO σ'(z) term → learning speed ∝ error size
    #   Accuracy:  ~96-97% (1-2% BETTER!)

    # WHY SATURATION HAPPENS:
    # At initialization, weights are random.
    # Some neurons randomly output very wrong predictions (a≈0 when y=1)
    # With quadratic cost, these neurons learn SLOWLY in early epochs
    # With cross-entropy, they learn FAST regardless of saturation

    # THE MATHEMATICAL INSIGHT:
    # Sigmoid derivative: σ'(z) = σ(z)·(1 - σ(z))
    # 
    # When a = 0.1 (wrong!): σ'(z) ≈ 0.09  → gradient × 0.09 (weak!)
    # When a = 0.5 (medium): σ'(z) = 0.25  → gradient × 0.25
    # When a = 0.9 (wrong!): σ'(z) ≈ 0.09  → gradient × 0.09 (weak!)
    # 
    # Cross-entropy REMOVES this bottleneck!

    # PRACTICAL IMPLICATIONS:
    # ✗ Lower accuracy than cross-entropy (~1-2% worse)
    # ✗ Slower initial learning (wasted computation)
    # ✗ Some neurons may never recover from bad initialization
    # ✓ Cross-entropy solves ALL these problems!

    # WHAT TO TRY NEXT:
    # • network_2.1.1.py: See side-by-side comparison with cross-entropy
    # • network_2.0.0.py: Use cross-entropy for BETTER results
    # • Compare epoch-by-epoch progress between the two

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
    print("\nKEY TAKEAWAY: Cross-entropy cost function solves the")
    print("learning slowdown problem by removing the σ'(z) term!")

if __name__ == "__main__":
    main()
