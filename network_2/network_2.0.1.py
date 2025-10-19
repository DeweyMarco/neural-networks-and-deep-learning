"""
No Regularization (λ=0.0) - Understanding Overfitting
======================================================

PREREQUISITES:
- Read network_2.0.0.py first to understand cross-entropy and regularization basics

THIS EXPERIMENT:
Demonstrates what happens when we use cross-entropy cost BUT NO REGULARIZATION.

KEY INSIGHT:
Without regularization (λ=0.0), the network can overfit to training data:
- Training accuracy will be very high
- Validation accuracy will be lower than with proper regularization
- Large weights develop because there's no penalty for them

COMPARISON TO BASELINE (network_2.0.0.py):
- Baseline: λ=5.0 → Expected ~96-97% validation accuracy
- This file: λ=0.0 → Expected ~95-96% validation accuracy
- Difference: ~1% accuracy loss due to overfitting

WHY THIS MATTERS:
Overfitting occurs when the network memorizes training data instead of learning
generalizable patterns. Without regularization:
- Weights can grow arbitrarily large
- Network fits noise in training data
- Performance on new data suffers

WATCH FOR:
- Training accuracy > validation accuracy (sign of overfitting)
- Gap between training and validation widens over epochs
- Validation accuracy may plateau or even decrease while training improves

Expected Results:
- Training accuracy: ~98-99%
- Validation accuracy: ~95-96%
- Gap: ~3% (this is overfitting!)

WHAT'S NEXT:
- network_2.0.0.py: Proper regularization (λ=5.0) - BETTER generalization
- network_2.0.2.py: Light regularization (λ=1.0) - Mild improvement
- network_2.0.3.py: Strong regularization (λ=15.0) - Too much constraint

Run: python network_2.0.1.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: Demonstrating overfitting when λ=0.0
    # ========================================================================
    print("=" * 60)
    print("Neural Network WITHOUT Regularization")
    print("=" * 60)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create network with cross-entropy cost
    # ========================================================================
    # We still use cross-entropy (better than quadratic cost)
    # But we'll set λ=0.0 during training (no regularization)
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10]")
    print("   Cost function: Cross-Entropy")
    print("   Regularization: NONE (λ=0.0)")
    print("   (Watch for overfitting: training >> validation accuracy)")
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train WITHOUT regularization
    # ========================================================================
    # KEY PARAMETER: lmbda=0.0
    #
    # With λ=0.0, the cost function is just:
    #   C = -(1/n) Σ [y·ln(a) + (1-y)·ln(1-a)]
    #
    # WITHOUT the regularization term:
    #   (λ/2n) Σw²
    #
    # Effect: No penalty for large weights!
    # - Network can develop very large weights to fit training data perfectly
    # - This includes fitting noise and peculiarities in training set
    # - Results in poor generalization to validation/test data
    #
    # The weight update rule becomes:
    #   w → w - η·∇C
    #
    # Instead of (with regularization):
    #   w → (1 - η·λ/n)·w - η·∇C
    #
    # Notice: No weight decay term pushing weights toward zero!
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 0.0 (NO REGULARIZATION)")
    print("\n" + "-" * 60)
    # Watch how training and validation accuracy diverge!
    print("-" * 60 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=0.0,  # NO REGULARIZATION - the key change!
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Analyze the overfitting
    # ========================================================================
    print("\n" + "=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100
    overfitting_gap = final_training_acc - final_validation_acc

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")
    print(f"\nOverfitting Gap:           {overfitting_gap:.2f}%")

    if overfitting_gap > 2.0:
        print("\n⚠ SIGNIFICANT OVERFITTING DETECTED!")
        print("   Training accuracy is much higher than validation accuracy.")
        print("   The network memorized training data instead of learning patterns.")
    else:
        print("\n✓ Minimal overfitting (gap < 2%)")

    # KEY OBSERVATIONS:
    # • Without regularization, weights can grow very large
    # • Network fits training data TOO well (including noise)
    # • Validation accuracy suffers from overfitting
    # • Training-validation gap indicates poor generalization

    # WHAT TO TRY NEXT:
    # • network_2.0.0.py: Add regularization (λ=5.0) to prevent overfitting
    # • network_2.0.2.py: Light regularization (λ=1.0) for mild improvement
    # • Compare validation accuracies to see regularization's benefit

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
