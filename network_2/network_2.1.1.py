"""
Side-by-Side Cost Function Comparison
======================================

PREREQUISITES:
- Read network_2.1.0.py to understand quadratic cost's learning slowdown
- Read network_2.0.0.py to understand cross-entropy cost

THIS EXPERIMENT:
Trains TWO identical networks simultaneously, differing ONLY in cost function:
1. Network A: Quadratic Cost (old method)
2. Network B: Cross-Entropy Cost (improved method)

This provides a DIRECT, controlled comparison showing why cross-entropy is better!

KEY INSIGHT:
By training both networks side-by-side with:
- Same architecture [784, 30, 10]
- Same random initialization
- Same hyperparameters (η=0.5, batch=10, λ=5.0)
- Same training data

We can ISOLATE the effect of the cost function and prove that cross-entropy
provides faster learning and better final accuracy.

WHAT YOU'LL SEE:
- Cross-entropy learns MUCH faster in early epochs
- Cross-entropy achieves higher final accuracy
- Quadratic cost suffers from learning slowdown when neurons saturate
- Clear demonstration of why modern networks use cross-entropy

Expected Results After 30 Epochs:
- Quadratic Cost:     ~94-95% validation accuracy
- Cross-Entropy Cost: ~96-97% validation accuracy
- Improvement:        +1-2% from cost function alone!

Early Learning (Epoch 5):
- Quadratic Cost:     ~92-93% (slow start)
- Cross-Entropy Cost: ~94-95% (fast start)

SCIENTIFIC METHOD:
This experiment demonstrates proper experimental design:
- Controlled variables (everything same except cost function)
- Clear hypothesis (cross-entropy learns faster)
- Direct comparison (parallel training)
- Measurable outcome (accuracy difference)

WHAT'S NEXT:
After seeing this dramatic difference, you'll understand why:
- ALL modern neural networks use cross-entropy (or similar) for classification
- Quadratic cost is now considered obsolete for classification tasks
- The cost function choice is as important as architecture choice!

Run: python network_2.1.1.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def print_epoch_comparison(epoch, quad_acc, cross_acc, validation_size):
    """Print a formatted comparison of both networks for one epoch."""
    quad_pct = 100.0 * quad_acc / validation_size
    cross_pct = 100.0 * cross_acc / validation_size
    diff = cross_pct - quad_pct

    diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"

    print(f"Epoch {epoch:2d}: Quadratic: {quad_acc:5d} ({quad_pct:5.2f}%)  |  "
          f"Cross-Entropy: {cross_acc:5d} ({cross_pct:5.2f}%)  |  "
          f"Difference: {diff_str}")

def main():
    # ========================================================================
    # EXPERIMENT: Training two identical networks with different cost functions
    # to demonstrate the superiority of cross-entropy cost.
    # ========================================================================
    print("=" * 80)
    print("SIDE-BY-SIDE COST FUNCTION COMPARISON")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create TWO identical networks with different cost functions
    # ========================================================================
    print("\n2. Creating two identical networks...")
    print("   Architecture: [784, 30, 10]")
    print("   Network A: Quadratic Cost (Mean Squared Error)")
    print("   Network B: Cross-Entropy Cost")
    print()
    print("   Both networks will use:")
    print("   - Same architecture")
    print("   - Same random seed (for fair comparison)")
    print("   - Same hyperparameters (η=0.5, batch=10, λ=5.0)")
    print("   - Same training data")

    # Set random seed for reproducibility
    import random
    import numpy as np

    # Create Network A with Quadratic Cost
    random.seed(42)
    np.random.seed(42)
    net_quadratic = network2.Network([784, 30, 10], cost=network2.QuadraticCost)

    # Create Network B with Cross-Entropy Cost
    # Use same random seed to ensure identical initialization
    random.seed(42)
    np.random.seed(42)
    net_cross_entropy = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    print("   ✓ Both networks initialized with identical random weights")

    # ========================================================================
    # STEP 3: Train both networks and compare epoch-by-epoch
    # ========================================================================
    print("\n3. Training both networks...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0")
    print("\n" + "=" * 80)
    print("EPOCH-BY-EPOCH COMPARISON")
    print("=" * 80)
    # Watch cross-entropy learn FASTER, especially in early epochs!
    print("-" * 80)

    # We'll train both networks epoch by epoch so we can display comparisons
    epochs = 30
    mini_batch_size = 10
    eta = 0.5
    lmbda = 5.0

    quadratic_accuracies = []
    cross_entropy_accuracies = []

    for epoch in range(epochs):
        # Train one epoch for quadratic cost network
        # We need to manually implement epoch-by-epoch training
        # This is a simplified version - in practice, you'd call SGD with epochs=1

        # For this demonstration, we'll just call SGD on each network
        # and track progress
        pass

    # Actually, network2's SGD doesn't support epoch-by-epoch monitoring easily,
    # so we'll train both networks and then compare final results
    # Let's modify approach to be more educational

    print("\nTraining Network A (Quadratic Cost)...")
    print("-" * 80)
    _, quad_eval_acc, _, quad_train_acc = \
        net_quadratic.SGD(training_data, epochs, mini_batch_size, eta,
                          lmbda=lmbda,
                          evaluation_data=validation_data,
                          monitor_evaluation_accuracy=True,
                          monitor_training_accuracy=True)

    print("\n" + "=" * 80)
    print("\nTraining Network B (Cross-Entropy Cost)...")
    print("-" * 80)
    _, cross_eval_acc, _, cross_train_acc = \
        net_cross_entropy.SGD(training_data, epochs, mini_batch_size, eta,
                              lmbda=lmbda,
                              evaluation_data=validation_data,
                              monitor_evaluation_accuracy=True,
                              monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Detailed comparison analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON ANALYSIS")
    print("=" * 80)

    # Epoch-by-epoch comparison
    print("\n" + "-" * 80)
    print("VALIDATION ACCURACY BY EPOCH:")
    print("-" * 80)
    for i in range(min(len(quad_eval_acc), len(cross_eval_acc))):
        print_epoch_comparison(i, quad_eval_acc[i], cross_eval_acc[i], len(validation_data))

    # Early learning comparison (first 5 epochs)
    print("\n" + "-" * 80)
    print("EARLY LEARNING ANALYSIS (Epochs 0-4):")
    print("-" * 80)
    quad_early_improvement = quad_eval_acc[4] - quad_eval_acc[0]
    cross_early_improvement = cross_eval_acc[4] - cross_eval_acc[0]
    print(f"Quadratic Cost improvement:     +{quad_early_improvement} correct")
    print(f"Cross-Entropy improvement:      +{cross_early_improvement} correct")
    print(f"Difference:                     {cross_early_improvement - quad_early_improvement} more with Cross-Entropy")

    if cross_early_improvement > quad_early_improvement:
        improvement_ratio = cross_early_improvement / quad_early_improvement if quad_early_improvement > 0 else float('inf')
        print(f"\n✓ Cross-entropy learned {improvement_ratio:.2f}× FASTER in early epochs!")
        print("  This demonstrates the solution to the learning slowdown problem.")

    # Final accuracy comparison
    print("\n" + "-" * 80)
    print("FINAL RESULTS (Epoch 29):")
    print("-" * 80)
    quad_final = quad_eval_acc[-1]
    cross_final = cross_eval_acc[-1]
    quad_final_pct = 100.0 * quad_final / len(validation_data)
    cross_final_pct = 100.0 * cross_final / len(validation_data)

    print(f"Quadratic Cost:      {quad_final} / {len(validation_data)} = {quad_final_pct:.2f}%")
    print(f"Cross-Entropy Cost:  {cross_final} / {len(validation_data)} = {cross_final_pct:.2f}%")
    print(f"\nImprovement:         +{cross_final - quad_final} correct = +{cross_final_pct - quad_final_pct:.2f}%")

    if cross_final > quad_final:
        print(f"\n✓ Cross-entropy achieved {cross_final - quad_final} more correct predictions!")
        print("  This is the power of choosing the right cost function.")

    # Training accuracy comparison (overfitting check)
    print("\n" + "-" * 80)
    print("OVERFITTING ANALYSIS:")
    print("-" * 80)
    quad_train_final_pct = 100.0 * quad_train_acc[-1] / len(training_data)
    cross_train_final_pct = 100.0 * cross_train_acc[-1] / len(training_data)
    quad_gap = quad_train_final_pct - quad_final_pct
    cross_gap = cross_train_final_pct - cross_final_pct

    print(f"Quadratic Cost:")
    print(f"  Training accuracy:   {quad_train_final_pct:.2f}%")
    print(f"  Validation accuracy: {quad_final_pct:.2f}%")
    print(f"  Gap:                 {quad_gap:.2f}%")
    print()
    print(f"Cross-Entropy Cost:")
    print(f"  Training accuracy:   {cross_train_final_pct:.2f}%")
    print(f"  Validation accuracy: {cross_final_pct:.2f}%")
    print(f"  Gap:                 {cross_gap:.2f}%")

    # Why cross-entropy is better
    # WHY CROSS-ENTROPY IS BETTER: THE MATHEMATICS
    # 
    # Quadratic Cost Gradient:
    #   ∂C/∂w = (a - y) · σ'(z) · x
    #   Problem: σ'(z) ≈ 0 when neuron saturated
    #   Result: SLOW learning when output is very wrong!
    # 
    # Cross-Entropy Cost Gradient:
    #   ∂C/∂w = (a - y) · x
    #   Advantage: NO σ'(z) term!
    #   Result: Learning speed proportional to error size
    # 
    # Example: Neuron outputs a=0.01 when y=1 (very wrong!)
    #   Quadratic gradient:     (0.01-1) · 0.0099 · x = -0.0098 · x (tiny!)
    #   Cross-entropy gradient: (0.01-1) · x = -0.99 · x (large!)
    #   Ratio: Cross-entropy is 100× LARGER!

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Cross-entropy solves the learning slowdown problem")
    print(f"2. Cross-entropy learns ~{cross_early_improvement/quad_early_improvement:.1f}× faster in early epochs")
    print(f"3. Cross-entropy achieves ~{cross_final_pct - quad_final_pct:.1f}% better final accuracy")
    print("4. The cost function choice is CRITICAL for neural network performance")
    print("5. This is why ALL modern classification networks use cross-entropy!")

    # HISTORICAL CONTEXT:
    # • Early neural networks (1980s-1990s) used quadratic cost
    # • Learning slowdown was a major obstacle
    # • Cross-entropy cost was introduced to solve this problem
    # • Today, cross-entropy is the standard for classification
    # • This single change improved accuracy by 1-3% across the board!

    # WHAT TO TRY NEXT:
    # • Always use cross-entropy for classification problems
    # • Combine with regularization (network_2.0.x series)
    # • Explore architecture improvements (network_2.2.x series)
    # • Understand that cost function + regularization = modern baseline

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
