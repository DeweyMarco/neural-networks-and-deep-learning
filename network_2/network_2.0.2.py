"""
Light Regularization (λ=1.0) - Finding the Balance
===================================================

PREREQUISITES:
- Read network_2.0.0.py first to understand regularization
- Read network_2.0.1.py to see the overfitting problem

THIS EXPERIMENT:
Demonstrates LIGHT regularization with λ=1.0.

This is a middle ground between:
- No regularization (λ=0.0) → overfitting
- Strong regularization (λ=5.0) → good generalization

KEY INSIGHT:
Light regularization provides SOME constraint on weights but may not be enough
to fully prevent overfitting. It's better than nothing but not optimal.

COMPARISON:
- No regularization (λ=0.0):     ~95-96% validation accuracy, ~3% overfitting gap
- Light regularization (λ=1.0):  ~96% validation accuracy, ~2% overfitting gap
- Proper regularization (λ=5.0): ~96-97% validation accuracy, ~1% overfitting gap

THE REGULARIZATION SPECTRUM:
λ=0.0  →  λ=1.0  →  λ=5.0  →  λ=15.0
Underfit ← OPTIMAL ZONE → Overfit
(too free)              (too constrained)

WHY λ=1.0 ISN'T OPTIMAL:
- Weight decay term: (1 - η·λ/n)·w
- With λ=1.0, η=0.5, n=50000: (1 - 0.5·1.0/50000) = 0.99999
- Weights decay by only 0.001% per update
- This is barely any constraint!

Compare to λ=5.0:
- (1 - 0.5·5.0/50000) = 0.99995
- Weights decay by 0.005% per update (5× more constraint)

WATCH FOR:
- Validation accuracy better than λ=0.0 but worse than λ=5.0
- Smaller training-validation gap than λ=0.0
- Still some overfitting visible

Expected Results:
- Training accuracy: ~97-98%
- Validation accuracy: ~96%
- Gap: ~2% (mild overfitting)

WHAT'S NEXT:
- network_2.0.0.py: Proper regularization (λ=5.0) - BEST results
- network_2.0.3.py: Strong regularization (λ=15.0) - Too much!

Run: python network_2.0.2.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: Testing light regularization (λ=1.0)
    # ========================================================================
    print("=" * 60)
    print("Neural Network with LIGHT Regularization")
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
    print("   Regularization: Light (λ=1.0)")
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train with LIGHT regularization
    # ========================================================================
    # KEY PARAMETER: lmbda=1.0
    #
    # The L2 regularization modifies the cost function:
    #   C = -(1/n) Σ [y·ln(a) + (1-y)·ln(1-a)] + (λ/2n) Σw²
    #                 \___ cross-entropy ___/     \_ regularization _/
    #
    # With λ=1.0, the regularization term is relatively small.
    #
    # Weight update rule:
    #   w → (1 - η·λ/n)·w - η·∇C
    #   w → (1 - 0.5·1.0/50000)·w - η·∇C
    #   w → 0.99999·w - η·∇C
    #
    # Interpretation:
    # - Each update, weights decay by 0.001%
    # - Over 150,000 updates (30 epochs × 5,000 batches):
    #   - Cumulative decay: (0.99999)^150000 ≈ 0.223 (77.7% decay)
    #   - This provides SOME constraint but may not be enough
    #
    # Compare to λ=5.0:
    #   - Per-update decay: 0.00005
    #   - Cumulative: (0.99995)^150000 ≈ 0.0005 (99.95% decay!)
    #   - Much stronger constraint on weights
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 1.0 (LIGHT REGULARIZATION)")
    print("\n" + "-" * 60)
    # Comparing light regularization to no regularization
    print("-" * 60 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=1.0,  # LIGHT REGULARIZATION
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Analyze the results
    # ========================================================================
    print("\n" + "=" * 60)
    print("REGULARIZATION ANALYSIS")
    print("=" * 60)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100
    overfitting_gap = final_training_acc - final_validation_acc

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")
    print(f"\nOverfitting Gap:           {overfitting_gap:.2f}%")

    if overfitting_gap > 2.5:
        print("\n⚠ SIGNIFICANT OVERFITTING")
        print("   Gap > 2.5% suggests regularization is too weak")
    elif overfitting_gap > 1.5:
        print("\n⚠ MILD OVERFITTING")
        print("   Gap between 1.5-2.5% suggests light overfitting")
        print("   Consider increasing λ to 5.0 for better generalization")
    else:
        print("\n✓ GOOD GENERALIZATION")
        print("   Gap < 1.5% indicates minimal overfitting")

    # REGULARIZATION SPECTRUM:
    # λ=0.0  (None):   95-96% validation, ~3% gap  [OVERFITTING]
    # λ=1.0  (Light):  ~96% validation, ~2% gap    [YOU ARE HERE]
    # λ=5.0  (Good):   96-97% validation, ~1% gap  [OPTIMAL]
    # λ=15.0 (Strong): 95-96% validation, <1% gap  [UNDERFITTING]

    # KEY OBSERVATIONS:
    # • Light regularization (λ=1.0) provides some weight constraint
    # • Better than no regularization but not optimal
    # • Weight decay per update: 0.001% (very small)
    # • Still allows significant overfitting to occur

    # TUNING RECOMMENDATION:
    if final_validation_acc < 96.5:
        # • Validation accuracy could be higher
        # • Try λ=5.0 for stronger regularization
        # • This will improve generalization by ~1%
        pass
    else:
        # • Current λ=1.0 is working reasonably well
        # • Still, λ=5.0 typically gives ~0.5-1% improvement
        pass

    # WHAT TO TRY NEXT:
    # • network_2.0.0.py: Proper regularization (λ=5.0) - BEST balance
    # • network_2.0.3.py: Strong regularization (λ=15.0) - See over-regularization
    # • Compare all results to understand the regularization trade-off

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
