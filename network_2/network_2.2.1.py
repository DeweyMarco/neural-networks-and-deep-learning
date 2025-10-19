"""
Deep Network with Modern Techniques [784, 60, 60, 10]
======================================================

PREREQUISITES:
- Read network_1.1.0.py and network_1.3.0.py to understand depth
- Read network_2.0.0.py to understand cross-entropy + regularization
- Read network_2.2.0.py to understand the wide network approach

THIS EXPERIMENT:
Applies MODERN TECHNIQUES to a deep network architecture.

DEPTH vs WIDTH COMPARISON:
- network_2.2.0.py: Wide [784, 100, 10] = 1 hidden layer, 100 neurons
- THIS FILE: Deep [784, 60, 60, 10] = 2 hidden layers, 60 neurons each

Both have similar parameter counts (~51k vs ~80k), but DIFFERENT strategies!

KEY INSIGHT:
Deep networks learn HIERARCHICAL representations:
- Layer 1: Low-level features (edges, curves, dots)
- Layer 2: Mid-level features (combinations of edges, digit parts)
- Layer 3: High-level features (complete digit patterns)

This mirrors how human visual cortex processes images!

ARCHITECTURE DETAILS:

[784, 60, 60, 10]
- Input: 784 neurons (28×28 pixels)
- Hidden 1: 60 neurons (learn low-level features)
- Hidden 2: 60 neurons (learn mid-level features)
- Output: 10 neurons (digit classes)

Parameters:
- Layer 1→2: 784×60 + 60 = 47,100
- Layer 2→3: 60×60 + 60 = 3,660
- Layer 3→4: 60×10 + 10 = 610
- TOTAL: 51,370 parameters

COMPARISON TO CHAPTER 1:
- network_1.3.0.py: [784, 60, 60, 10] with quadratic cost, no regularization
  → Achieved ~97-98% but with learning slowdown
- THIS FILE: Same architecture with cross-entropy + regularization
  → Expected ~97-98% with FASTER learning and better generalization

THE DEPTH ADVANTAGE:
- More parameter-efficient than pure width
- Hierarchical feature learning (automatic feature engineering!)
- Each layer builds on previous layer's features
- Enables compositional understanding

THE DEPTH CHALLENGE WITH OLD TECHNIQUES:
- Quadratic cost: Saturation in deeper layers → slow learning
- No regularization: More layers = more overfitting risk
- Solution: Cross-entropy keeps gradients flowing + regularization prevents overfitting!

Expected Results:
- Validation accuracy: ~97-98% (comparable to wide network)
- Training-validation gap: ~1% (good generalization)
- Fewer parameters than wide network but similar accuracy!
- Faster learning than network_1.3.0.py (thanks to cross-entropy)

WHAT'S NEXT:
- network_2.2.2.py: Wide+Deep [784, 100, 100, 10] - combine both strategies!

Run: python network_2.2.1.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: [784, 60, 60, 10] with cross-entropy + regularization
    # ========================================================================
    print("=" * 70)
    print("Deep Network with Modern Techniques")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create deep network with modern techniques
    # ========================================================================
    # ARCHITECTURE: [784, 60, 60, 10]
    #
    # This is a DEEP network with 2 hidden layers:
    # - Depth allows hierarchical feature learning
    # - More layers with fewer neurons per layer
    # - More parameter-efficient than pure width
    #
    # HIERARCHICAL LEARNING:
    # Think of it like reading comprehension:
    # - Layer 1: Recognizes letters (edges, curves)
    # - Layer 2: Recognizes words (digit parts, combinations)
    # - Output: Recognizes sentences (complete digits)
    #
    # WHY MODERN TECHNIQUES ARE CRITICAL FOR DEPTH:
    # 1. Cross-Entropy: Prevents vanishing gradients in deeper networks
    # 2. Regularization: Controls overfitting with more layers
    # 3. Together: Enable effective deep learning!
    print("\n2. Creating deep network...")
    print("   Architecture: [784, 60, 60, 10]")
    print("   Hidden layers: 2 (DEPTH strategy)")
    print("   Neurons per layer: 60 each")
    print("   Cost function: Cross-Entropy")
    print("   Regularization: λ=5.0")

    # Calculate parameters
    weights_1 = 784 * 60
    biases_1 = 60
    weights_2 = 60 * 60
    biases_2 = 60
    weights_3 = 60 * 10
    biases_3 = 10
    total_params = weights_1 + biases_1 + weights_2 + biases_2 + weights_3 + biases_3

    print(f"\n   Parameters:")
    print(f"   - Layer 1→2: {weights_1:,} weights + {biases_1} biases = {weights_1 + biases_1:,}")
    print(f"   - Layer 2→3: {weights_2:,} weights + {biases_2} biases = {weights_2 + biases_2:,}")
    print(f"   - Layer 3→4: {weights_3:,} weights + {biases_3} biases = {weights_3 + biases_3:,}")
    print(f"   - TOTAL: {total_params:,} parameters")

    # Compare to wide network
    wide_params = 79510  # From network_2.2.0.py [784, 100, 10]
    print(f"\n   Efficiency comparison:")
    print(f"   - Wide [784,100,10]: {wide_params:,} parameters")
    print(f"   - Deep [784,60,60,10]: {total_params:,} parameters")
    print(f"   - Deep uses {wide_params/total_params:.1f}× FEWER parameters!")

    net = network2.Network([784, 60, 60, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train with modern techniques
    # ========================================================================
    # WHY CROSS-ENTROPY IS CRITICAL FOR DEPTH:
    #
    # In deep networks, gradients must flow backward through multiple layers.
    # With quadratic cost:
    #   ∂C/∂w = (a - y) · σ'(z) · [more σ'(z) terms from earlier layers]
    #
    # Each layer multiplies by σ'(z), which can be very small!
    # Result: "Vanishing gradient" - early layers learn VERY slowly
    #
    # With cross-entropy:
    #   ∂C/∂w = (a - y) · [propagated through layers]
    #
    # No σ'(z) in the output error!
    # Result: Stronger gradients reach earlier layers → faster learning
    #
    # WHY REGULARIZATION IS CRITICAL FOR DEPTH:
    # - More layers = more parameters to control
    # - Each layer can potentially overfit
    # - Regularization constrains ALL layers simultaneously
    # - Prevents the network from memorizing training data
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0")
    print("\n" + "-" * 70)
    # Deep network benefits:
    # • Hierarchical feature learning (low→mid→high level)
    # • Parameter efficient (51K vs 80K for wide network)
    # • Compositional understanding of patterns
    #
    # Modern techniques enable depth:
    # • Cross-entropy: Prevents vanishing gradients
    # • Regularization: Controls multiple layers
    print("-" * 70 + "\n")

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
    # STEP 4: Analyze depth vs width trade-offs
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
        print("   Modern techniques successfully enable deep learning")

    # Accuracy per parameter
    accuracy_per_1k_params = final_validation_acc / (total_params / 1000)

    # DEPTH vs WIDTH COMPARISON:
    # 
    # Wide Network [784, 100, 10]:
    #   Parameters: 79,510
    #   Accuracy: ~97-98%
    #   Strategy: Parallel feature learning
    #   Efficiency: ~1.22% per 1K parameters
    # 
    # Deep Network [784, 60, 60, 10] (THIS NETWORK):
    print(f"\n   Parameters: {total_params:,}")
    print(f"   Accuracy: ~{final_validation_acc:.1f}%")
    print(f"   Efficiency: ~{accuracy_per_1k_params:.2f}% per 1K parameters")
    #   Strategy: Hierarchical feature learning

    if accuracy_per_1k_params > 1.8:
        print("   ✓ DEPTH is more PARAMETER-EFFICIENT!")

    # HIERARCHICAL FEATURE LEARNING:
    # 
    # What each layer learns (conceptually):
    # 
    # Hidden Layer 1 (60 neurons):
    #   • Edges at different angles
    #   • Curves and arcs
    #   • Dots and endpoints
    #   • Basic strokes
    # 
    # Hidden Layer 2 (60 neurons):
    #   • Combinations of edges (corners, T-junctions)
    #   • Loops and circles (for 0, 6, 8, 9)
    #   • Vertical/horizontal lines (for 1, 4, 7)
    #   • Digit parts (top of 5, bottom of 2)
    # 
    # Output Layer (10 neurons):
    #   • Complete digit patterns
    #   • Combines features from layer 2
    #   • Makes final classification decision

    # COMPARISON TO CHAPTER 1 (network_1.3.0.py):
    # 
    # network_1.3.0.py: [784, 60, 60, 10] with OLD techniques
    #   Cost: Quadratic (learning slowdown in depth)
    #   Regularization: None (overfitting risk)
    #   Accuracy: ~97-98% but SLOW early learning
    # 
    # THIS NETWORK: Same architecture with MODERN techniques
    #   Cost: Cross-entropy (fast learning in all layers)
    #   Regularization: λ=5.0 (controlled overfitting)
    #   Accuracy: ~{final_validation_acc:.1f}% with FASTER convergence
    # 
    # Improvement: Same accuracy, better training dynamics!

    # WHY DEPTH WORKS WELL HERE:
    # 
    # 1. HIERARCHICAL STRUCTURE matches the problem:
    #    • Digits are made of parts (compositional)
    #    • Parts are made of strokes (hierarchical)
    #    • Depth naturally captures this structure
    # 
    # 2. CROSS-ENTROPY enables effective backpropagation:
    #    • Strong gradients reach early layers
    #    • All layers learn effectively
    #    • No vanishing gradient problem
    # 
    # 3. REGULARIZATION prevents overfitting:
    #    • Controls all 51,370 parameters
    #    • Encourages simple, generalizable features
    #    • Each layer learns robust representations

    # WHEN TO USE DEEP vs WIDE:
    # 
    # Use DEPTH when:
    #   ✓ Problem has hierarchical structure
    #   ✓ Want parameter efficiency
    #   ✓ Need compositional feature learning
    #   ✓ Have good optimization techniques (cross-entropy, etc.)
    # 
    # Use WIDTH when:
    #   ✓ Need many diverse, independent features
    #   ✓ Want to avoid vanishing gradients
    #   ✓ Have enough data and computation for more parameters
    #   ✓ Problem doesn't have clear hierarchy

    # WHAT TO TRY NEXT:
    # • network_2.2.2.py: Wide+Deep [784,100,100,10]
    #   → Combine BOTH strategies for maximum capacity
    # • Compare all three: baseline [784,30,10], wide, deep
    # • Experiment with different depths (3, 4, 5 layers)
    # • Try different layer sizes (30, 40, 50 neurons)

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
    print(f"\nKEY TAKEAWAY: Modern techniques make DEEP networks practical,")
    print(f"achieving ~{final_validation_acc:.1f}% with hierarchical feature learning!")

if __name__ == "__main__":
    main()
