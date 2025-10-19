"""
Strong Regularization (λ=15.0) - Over-Regularization
=====================================================

PREREQUISITES:
- Read network_2.0.0.py first to understand regularization
- Read network_2.0.1.py and network_2.0.2.py for context

THIS EXPERIMENT:
Demonstrates OVER-REGULARIZATION with λ=15.0.

This is TOO MUCH constraint on weights, leading to underfitting!

KEY INSIGHT:
Strong regularization can be as bad as no regularization:
- Weights are forced to stay very small
- Network can't learn complex patterns
- Both training AND validation accuracy suffer

COMPARISON:
- No regularization (λ=0.0):     ~95-96% validation, high overfitting
- Light regularization (λ=1.0):  ~96% validation, mild overfitting
- Proper regularization (λ=5.0): ~96-97% validation, minimal overfitting [BEST]
- Strong regularization (λ=15.0): ~95-96% validation, NO overfitting but UNDERFITTING

THE REGULARIZATION SPECTRUM:
Too weak ← OPTIMAL → Too strong
λ=0.0   λ=1.0   λ=5.0   λ=15.0   λ=100
Overfit        BEST      Underfit

WHY λ=15.0 IS TOO STRONG:
- Weight decay term: (1 - η·λ/n)·w
- With λ=15.0, η=0.5, n=50000: (1 - 0.5·15.0/50000) = 0.99985
- Weights decay by 0.015% per update (15× stronger than λ=1.0!)

Per-update weight decay comparison:
- λ=1.0:  0.99999 → gentle constraint
- λ=5.0:  0.99995 → good constraint
- λ=15.0: 0.99985 → aggressive constraint

Over 150,000 updates (30 epochs):
- λ=5.0:  (0.99995)^150000 ≈ 0.0005 (99.95% decay)
- λ=15.0: (0.99985)^150000 ≈ 0.0000000004 (essentially zero!)

Result: Weights are forced to stay near zero, limiting capacity!

WATCH FOR:
- Lower validation accuracy than λ=5.0
- Lower training accuracy too (can't even fit training data well!)
- Very small training-validation gap (both are bad!)
- No overfitting, but underfitting instead

Expected Results:
- Training accuracy: ~96%
- Validation accuracy: ~95-96%
- Gap: <1% (no overfitting, but performance suffers)

WHAT'S NEXT:
- Go back to network_2.0.0.py: λ=5.0 is the sweet spot!
- Understand that MORE regularization ≠ BETTER performance

Run: python network_2.0.3.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: Demonstrating over-regularization (λ=15.0)
    # ========================================================================
    print("=" * 60)
    print("Neural Network with STRONG Regularization")
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
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10]")
    print("   Cost function: Cross-Entropy")
    print("   Regularization: Strong (λ=15.0)")
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train with STRONG regularization
    # ========================================================================
    # KEY PARAMETER: lmbda=15.0
    #
    # The L2 regularization modifies the cost function:
    #   C = -(1/n) Σ [y·ln(a) + (1-y)·ln(1-a)] + (λ/2n) Σw²
    #
    # With λ=15.0, the regularization term DOMINATES!
    #
    # Weight update rule:
    #   w → (1 - η·λ/n)·w - η·∇C
    #   w → (1 - 0.5·15.0/50000)·w - η·∇C
    #   w → 0.99985·w - η·∇C
    #
    # Interpretation:
    # - Each update, weights decay by 0.015%
    # - This is 15× stronger than λ=1.0
    # - This is 3× stronger than λ=5.0
    #
    # Over 150,000 updates:
    #   (0.99985)^150000 ≈ 0.0000000004
    #
    # This means weights are FORCED to essentially zero!
    #
    # Problem: THE COST FUNCTION IS UNBALANCED
    # - The regularization penalty (λ/2n)Σw² with λ=15.0 is huge
    # - The network cares more about keeping weights small
    # - Than about actually classifying digits correctly!
    # - This is like telling a student: "I don't care if you get the
    #   answer right, just make sure your handwriting is tiny!"
    #
    # Result: UNDERFITTING
    # - Network can't learn complex patterns
    # - Both training and validation accuracy suffer
    # - The model is too simple for the task
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 15.0 (STRONG REGULARIZATION)")
    print("\n" + "-" * 60)
    # Watch how performance suffers from over-regularization
    print("-" * 60 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=15.0,  # STRONG REGULARIZATION - TOO MUCH!
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Analyze the over-regularization
    # ========================================================================
    print("\n" + "=" * 60)
    print("OVER-REGULARIZATION ANALYSIS")
    print("=" * 60)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100
    overfitting_gap = final_training_acc - final_validation_acc

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")
    print(f"\nOverfitting Gap:           {overfitting_gap:.2f}%")

    # Compare to baseline (λ=5.0) which should get ~96-97%
    if final_validation_acc < 96.0:
        print("\n⚠ UNDERFITTING DETECTED!")
        print("   Validation accuracy is LOWER than with λ=5.0")
        print("   Regularization is TOO STRONG - network can't learn!")
        print("   The penalty for large weights outweighs the benefit")
    else:
        print("\n✓ Performance comparable to lighter regularization")
        print("   (This network might have benefited from the strong constraint)")

    if overfitting_gap < 0.5:
        print("\n✓ No overfitting (gap very small)")
        print("   BUT this comes at the cost of overall performance!")
        print("   Both training AND validation accuracy are suppressed")

    # REGULARIZATION SPECTRUM SUMMARY:
    # λ=0.0  (None):   95-96% validation, ~3% gap    [OVERFITTING]
    # λ=1.0  (Light):  ~96% validation, ~2% gap      [MILD OVERFITTING]
    # λ=5.0  (Good):   96-97% validation, ~1% gap    [OPTIMAL ✓]
    # λ=15.0 (Strong): 95-96% validation, <0.5% gap  [YOU ARE HERE - UNDERFITTING]

    # KEY OBSERVATIONS:
    # • Strong regularization (λ=15.0) over-constrains weights
    # • Network capacity is artificially limited
    # • BOTH training and validation accuracy suffer
    # • No overfitting, but that's because model can't learn!
    # • Cumulative weight decay: ~100% (weights → 0)

    # THE REGULARIZATION PARADOX:
    # ✗ Too little regularization (λ=0.0): Overfits training data
    # ✓ Just right (λ=5.0): Balances fitting and generalization
    # ✗ Too much regularization (λ=15.0): Can't learn anything!
    # 
    # Regularization is about BALANCE, not maximization!

    # WHEN MIGHT STRONG REGULARIZATION HELP?
    # • Very large networks (many parameters) prone to overfitting
    # • Very small training datasets
    # • When you need maximum robustness over accuracy
    #
    # For MNIST with this architecture, λ=5.0 is MUCH better!

    # WHAT TO TRY NEXT:
    # • network_2.0.0.py: Go back to optimal λ=5.0 (BEST results)
    # • Run all four experiments (λ=0.0, 1.0, 5.0, 15.0) and compare
    # • Plot validation accuracy vs λ to visualize the sweet spot

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
    print("\nREMEMBER: More regularization ≠ Better performance!")
    print("Find the balance that maximizes generalization.")

if __name__ == "__main__":
    main()
