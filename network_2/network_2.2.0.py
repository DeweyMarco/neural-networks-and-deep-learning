"""
Wide Network with Modern Techniques [784, 100, 10]
===================================================

PREREQUISITES:
- Read network_1.2.0.py to understand wide networks
- Read network_2.0.0.py to understand cross-entropy + regularization

THIS EXPERIMENT:
Applies MODERN TECHNIQUES to a wide network architecture.

COMPARISON TO CHAPTER 1:
- network_1.2.0.py: Wide network [784, 60, 10] with quadratic cost, no regularization
  → Achieved ~96-97% accuracy
- THIS FILE: Wide network [784, 100, 10] with cross-entropy + regularization
  → Expected ~97-98% accuracy

KEY INSIGHT:
Modern techniques (cross-entropy + regularization) make LARGER networks practical!

Without regularization, a 100-neuron hidden layer would severely overfit.
With cross-entropy alone, learning would be faster but still overfit.
WITH BOTH: We get the capacity benefits WITHOUT the overfitting penalty!

ARCHITECTURE DETAILS:

[784, 100, 10]
- Input: 784 neurons (28×28 MNIST images)
- Hidden: 100 neurons (3.3× wider than baseline's 30)
- Output: 10 neurons (digit classes)

Parameters:
- Weights: 784×100 + 100×10 = 79,400 weights
- Biases: 100 + 10 = 110 biases
- TOTAL: 79,510 parameters (3.3× more than baseline's 23,860!)

WHY WIDTH HELPS:
- More neurons = more parallel feature detectors
- Can learn more diverse patterns simultaneously
- Each neuron specializes in different features
- Higher representational capacity

THE REGULARIZATION CHALLENGE:
With 79,510 parameters, overfitting risk is HIGH!
- More parameters = more ways to memorize training data
- Without regularization: ~95% validation (overfits badly)
- With regularization: ~97-98% validation (generalizes well!)

Expected Results:
- Validation accuracy: ~97-98% (BEST single hidden layer!)
- Training-validation gap: ~1% (good generalization)
- Improvement over network_1.2.0.py: +1-2%
- Improvement over baseline: +2-3%

WHAT'S NEXT:
- network_2.2.1.py: Deep network [784, 60, 60, 10] - hierarchy vs width
- network_2.2.2.py: Wide+Deep [784, 100, 100, 10] - maximum capacity

Run: python network_2.2.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: [784, 100, 10] with cross-entropy + regularization
    # ========================================================================
    print("=" * 70)
    print("Wide Network with Modern Techniques")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create wide network with modern techniques
    # ========================================================================
    # ARCHITECTURE: [784, 100, 10]
    #
    # This is a WIDE network:
    # - 100 neurons in hidden layer (vs 30 in baseline)
    # - 3.3× more parameters
    # - Much higher capacity
    #
    # WHY 100 NEURONS?
    # - MNIST digits have many subtle variations
    # - More neurons = more feature detectors
    # - Can learn fine-grained patterns (loops, curves, angles, etc.)
    # - Reduces "competition" between neurons
    #
    # THE MODERN APPROACH:
    # 1. Cross-Entropy Cost: Fast learning, no saturation slowdown
    # 2. L2 Regularization: Prevents overfitting despite large capacity
    # 3. Validation Monitoring: Proper evaluation methodology
    #
    # Result: We can use MORE parameters safely!
    print("\n2. Creating wide network...")
    print("   Architecture: [784, 100, 10]")
    print("   Hidden layer size: 100 neurons (3.3× baseline)")
    print("   Cost function: Cross-Entropy")
    print("   Regularization: λ=5.0")

    # Calculate parameters
    weights_1 = 784 * 100
    biases_1 = 100
    weights_2 = 100 * 10
    biases_2 = 10
    total_params = weights_1 + biases_1 + weights_2 + biases_2

    print(f"\n   Parameters:")
    print(f"   - Layer 1→2: {weights_1:,} weights + {biases_1} biases")
    print(f"   - Layer 2→3: {weights_2:,} weights + {biases_2} biases")
    print(f"   - TOTAL: {total_params:,} parameters")
    print(f"   - vs Baseline [784,30,10]: {total_params/23860:.1f}× more parameters")

    net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train with modern techniques
    # ========================================================================
    # WHY REGULARIZATION IS CRITICAL HERE:
    #
    # With 79,510 parameters and 50,000 training samples:
    # - Ratio: 1.6 parameters per training sample
    # - High risk of overfitting!
    #
    # Without regularization (λ=0.0):
    # - Network would memorize training data
    # - Validation accuracy: ~95% (poor generalization)
    # - Training-validation gap: ~4-5%
    #
    # With regularization (λ=5.0):
    # - Weight decay prevents overfitting
    # - Validation accuracy: ~97-98% (excellent!)
    # - Training-validation gap: ~1% (good generalization)
    #
    # The regularization term (λ/2n)Σw² becomes MORE important
    # as network size increases!
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0 (CRITICAL for this size!)")
    print("\n" + "-" * 70)
    # Wide network benefits:
    # • More parallel feature detectors (100 vs 30)
    # • Higher representational capacity
    # • Can learn finer-grained patterns
    #
    # Modern techniques prevent overfitting:
    # • Cross-entropy: Fast learning
    # • Regularization: Controls large parameter space
    print("-" * 70 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=5.0,  # Essential for preventing overfitting!
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Analyze the benefits of width + modern techniques
    # ========================================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100
    overfitting_gap = final_training_acc - final_validation_acc

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")
    print(f"\nOverfitting Gap:           {overfitting_gap:.2f}%")

    if overfitting_gap < 2.0:
        print("\n✓ EXCELLENT GENERALIZATION!")
        print("   Regularization successfully controlled 79,510 parameters")
    else:
        print("\n⚠ Some overfitting detected")
        print("   Consider increasing λ slightly")

    # COMPARISON TO OTHER APPROACHES:
    # 
    # network_1.0.0.py [784, 30, 10]:
    #   Quadratic cost, no regularization
    #   Accuracy: ~95%
    #   Parameters: 23,860
    # 
    # network_1.2.0.py [784, 60, 10]:
    #   Quadratic cost, no regularization, wider
    #   Accuracy: ~96-97%
    #   Parameters: 47,710
    # 
    # network_2.0.0.py [784, 30, 10]:
    #   Cross-entropy + regularization, baseline width
    #   Accuracy: ~96-97%
    #   Parameters: 23,860
    # 
    # THIS NETWORK [784, 100, 10]:
    #   Cross-entropy + regularization, WIDE
    print(f"\n   Final Accuracy: ~{final_validation_acc:.1f}%")
    print(f"   Parameters: {total_params:,}")
    print(f"   Improvement: +{final_validation_acc - 95:.1f}% over baseline!")

    # Accuracy per parameter (efficiency metric)
    accuracy_per_1k_params = final_validation_acc / (total_params / 1000)
    baseline_acc_per_1k = 95.0 / (23860 / 1000)

    print("\n" + "-" * 70)
    print("EFFICIENCY ANALYSIS:")
    print("-" * 70)
    print(f"Accuracy per 1K parameters:")
    print(f"  Baseline [784,30,10]: {baseline_acc_per_1k:.2f}%")
    print(f"  This network:         {accuracy_per_1k_params:.2f}%")

    if accuracy_per_1k_params < baseline_acc_per_1k:
        print("\nNote: Lower efficiency (diminishing returns from extra parameters)")
        print("But HIGHER absolute accuracy - that's the trade-off!")
    else:
        print("\n✓ Maintained or improved efficiency!")

    # WHY THIS COMBINATION WORKS:
    # 
    # 1. WIDTH provides capacity:
    #    • 100 neurons can learn 100 different features
    #    • More diverse pattern recognition
    #    • Less competition between neurons
    # 
    # 2. CROSS-ENTROPY prevents learning slowdown:
    #    • Fast learning even when neurons saturated
    #    • Gradient ∝ error size
    #    • All 100 neurons learn efficiently
    # 
    # 3. REGULARIZATION prevents overfitting:
    #    • Controls 79,510 parameters
    #    • Forces network to learn generalizable patterns
    #    • Weight decay keeps parameters reasonable

    # WHEN TO USE WIDE NETWORKS:
    # ✓ Good for: Problems with many diverse features
    # ✓ Good for: When you want more capacity without depth
    # ✓ Good for: Avoiding vanishing gradient issues
    # ⚠ Trade-off: More parameters = more computation
    # ⚠ Trade-off: Requires careful regularization

    # WHAT TO TRY NEXT:
    # • network_2.2.1.py: Deep network [784,60,60,10]
    #   → Compare depth vs width strategies
    # • network_2.2.2.py: Wide+Deep [784,100,100,10]
    #   → Combine both for maximum capacity
    # • Experiment with different widths (50, 150, 200)
    # • Try different λ values for this architecture

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
    print(f"\nKEY TAKEAWAY: Modern techniques (cross-entropy + regularization)")
    print(f"make WIDE networks practical, achieving ~{final_validation_acc:.1f}% accuracy!")

if __name__ == "__main__":
    main()
