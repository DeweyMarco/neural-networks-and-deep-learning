"""
Wide + Deep Network: Maximum Capacity [784, 100, 100, 10]
===========================================================

PREREQUISITES:
- Read network_2.2.0.py to understand wide networks
- Read network_2.2.1.py to understand deep networks
- Read network_2.0.0.py to understand regularization

THIS EXPERIMENT:
Combines BOTH width AND depth for maximum representational power.

THE BEST OF BOTH WORLDS:
- WIDTH (100 neurons/layer): Many parallel feature detectors
- DEPTH (2 hidden layers): Hierarchical feature composition
- MODERN TECHNIQUES: Cross-entropy + regularization make it practical!

ARCHITECTURE EVOLUTION:

Chapter 1 Baseline [784, 30, 10]:
  23,860 parameters, ~95% accuracy

Chapter 1 Wide [784, 60, 10]:
  47,710 parameters, ~96-97% accuracy

Chapter 1 Deep [784, 30, 30, 10]:
  24,790 parameters, ~95-96% accuracy

Chapter 1 Wide+Deep [784, 60, 60, 10]:
  51,370 parameters, ~97-98% accuracy

THIS NETWORK [784, 100, 100, 10]:
  88,610 parameters, ~98%+ accuracy (BEST!)

KEY INSIGHT:
Without modern techniques, this network would be IMPOSSIBLE to train:
- Quadratic cost: Severe learning slowdown in deep, wide networks
- No regularization: Catastrophic overfitting with 88K parameters
- Result: Poor performance despite high capacity

WITH modern techniques:
- Cross-entropy: Fast learning in all layers
- L2 regularization: Controls 88K parameters effectively
- Result: HIGHEST accuracy we can achieve with these techniques!

ARCHITECTURE DETAILS:

[784, 100, 100, 10]
- Input: 784 neurons
- Hidden 1: 100 neurons (wide + low-level features)
- Hidden 2: 100 neurons (wide + mid-level features)
- Output: 10 neurons

Parameters:
- Layer 1→2: 784×100 + 100 = 78,500
- Layer 2→3: 100×100 + 100 = 10,100
- Layer 3→4: 100×10 + 10 = 1,010
- TOTAL: 88,610 parameters (3.7× baseline!)

THE REGULARIZATION CHALLENGE:
- 88,610 parameters to control
- 50,000 training samples
- Ratio: 1.77 parameters per training sample!
- Without λ=5.0: Severe overfitting
- With λ=5.0: Excellent generalization

Expected Results:
- Validation accuracy: ~98%+ (BEST achievable with these techniques!)
- Training-validation gap: ~1-2% (regularization working hard!)
- Improvement over baseline: +3-4%

WHAT'S NEXT:
This represents the limit of what quadratic loss + sigmoid can achieve.
For better performance, you need:
- network3.py: ReLU activation, better initialization, dropout
- Convolutional networks: Specialized architectures for images
- Modern frameworks: PyTorch, TensorFlow

Run: python network_2.2.2.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    # ========================================================================
    # EXPERIMENT: [784, 100, 100, 10] - Combining width AND depth
    # ========================================================================
    print("=" * 75)
    print("Wide + Deep Network: MAXIMUM CAPACITY")
    print("=" * 75)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")

    # ========================================================================
    # STEP 2: Create maximum capacity network
    # ========================================================================
    # ARCHITECTURE: [784, 100, 100, 10]
    #
    # This combines:
    # 1. WIDTH: 100 neurons per layer (lots of capacity)
    # 2. DEPTH: 2 hidden layers (hierarchical learning)
    #
    # Why this is powerful:
    # - Layer 1: 100 neurons learn diverse low-level features
    # - Layer 2: 100 neurons combine them into mid-level features
    # - More combinations possible than with fewer neurons
    # - Maximum representational power for this architecture type
    #
    # Why this needs modern techniques:
    # - 88K parameters need strong regularization
    # - Deep + wide = risk of vanishing gradients
    # - Cross-entropy keeps learning fast throughout
    # - Regularization prevents overfitting
    print("\n2. Creating wide + deep network...")
    print("   Architecture: [784, 100, 100, 10]")
    print("   Strategy: MAXIMUM CAPACITY (width + depth)")
    print("   Hidden layers: 2 layers × 100 neurons each")
    print("   Cost function: Cross-Entropy")
    print("   Regularization: λ=5.0 (CRITICAL!)")

    # Calculate parameters
    weights_1 = 784 * 100
    biases_1 = 100
    weights_2 = 100 * 100
    biases_2 = 100
    weights_3 = 100 * 10
    biases_3 = 10
    total_params = weights_1 + biases_1 + weights_2 + biases_2 + weights_3 + biases_3

    print(f"\n   Parameters:")
    print(f"   - Layer 1→2: {weights_1:,} weights + {biases_1} biases = {weights_1 + biases_1:,}")
    print(f"   - Layer 2→3: {weights_2:,} weights + {biases_2} biases = {weights_2 + biases_2:,}")
    print(f"   - Layer 3→4: {weights_3:,} weights + {biases_3} biases = {weights_3 + biases_3:,}")
    print(f"   - TOTAL: {total_params:,} parameters")
    print(f"\n   Scale comparison:")
    print(f"   - Baseline [784,30,10]: 23,860 parameters (1.0×)")
    print(f"   - This network: {total_params:,} parameters ({total_params/23860:.1f}×)")

    # Overfitting risk calculation
    param_per_sample = total_params / len(training_data)
    print(f"\n   Overfitting risk:")
    print(f"   - Parameters: {total_params:,}")
    print(f"   - Training samples: {len(training_data):,}")
    print(f"   - Ratio: {param_per_sample:.2f} parameters per sample")
    if param_per_sample > 1.5:
        print("   - ⚠ HIGH RISK: Regularization is ESSENTIAL!")
    else:
        print("   - ✓ Moderate risk")

    net = network2.Network([784, 100, 100, 10], cost=network2.CrossEntropyCost)

    # ========================================================================
    # STEP 3: Train with strong regularization
    # ========================================================================
    # THE NECESSITY OF REGULARIZATION:
    #
    # Without regularization (λ=0.0):
    # - Training accuracy: ~99%+ (nearly perfect on training data!)
    # - Validation accuracy: ~94-95% (poor generalization)
    # - Gap: ~4-5% (severe overfitting)
    # - Problem: Network memorizes training data
    #
    # With regularization (λ=5.0):
    # - Training accuracy: ~98%
    # - Validation accuracy: ~98% (excellent generalization!)
    # - Gap: ~1% (minimal overfitting)
    # - Success: Network learns generalizable patterns
    #
    # The weight decay term (λ/2n)Σw² with λ=5.0:
    # - Penalizes large weights across ALL 88K parameters
    # - Forces network to use many small weights
    # - Prevents any single feature from dominating
    # - Results in robust, generalizable representations
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0")
    print("\n" + "-" * 75)
    # COMBINING WIDTH + DEPTH:
    # 
    # Width benefits (100 neurons/layer):
    #   • 100 parallel feature detectors per layer
    #   • High capacity for diverse patterns
    #   • Reduced competition between neurons
    # 
    # Depth benefits (2 hidden layers):
    #   • Layer 1: 100 low-level features (edges, curves)
    #   • Layer 2: 100 mid-level features (digit parts)
    #   • 100×100 = 10,000 possible feature combinations!
    # 
    # Modern techniques enable this:
    #   • Cross-entropy: Fast learning despite depth
    #   • Regularization: Controls 88K parameters
    #   • Result: Maximum performance!
    print("-" * 75 + "\n")

    # Run training with monitoring
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=5.0,  # CRITICAL for this many parameters!
                evaluation_data=validation_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)

    # ========================================================================
    # STEP 4: Comprehensive performance analysis
    # ========================================================================
    print("\n" + "=" * 75)
    print("PERFORMANCE ANALYSIS: THE PINNACLE OF CHAPTER 2")
    print("=" * 75)

    final_training_acc = training_accuracy[-1] / len(training_data) * 100
    final_validation_acc = evaluation_accuracy[-1] / len(validation_data) * 100
    overfitting_gap = final_training_acc - final_validation_acc

    print(f"\nFinal Training Accuracy:   {training_accuracy[-1]} / {len(training_data)}")
    print(f"                           {final_training_acc:.2f}%")
    print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"                           {final_validation_acc:.2f}%")
    print(f"\nOverfitting Gap:           {overfitting_gap:.2f}%")

    if final_validation_acc >= 97.5:
        print("\n✓✓✓ EXCEPTIONAL PERFORMANCE!")
        print("    This represents near-optimal results for sigmoid + cross-entropy!")
    elif final_validation_acc >= 96.5:
        print("\n✓✓ EXCELLENT PERFORMANCE!")
        print("   Regularization successfully controlled 88K parameters")
    else:
        print("\n✓ GOOD PERFORMANCE")
        print("  Consider tuning λ or increasing epochs")

    # COMPLETE ARCHITECTURE COMPARISON:
    # 
    # CHAPTER 1 (Quadratic Cost, No Regularization):
    # network_1.0.0 [784, 30, 10]:       23,860 params → ~95.0% accuracy
    # network_1.2.0 [784, 60, 10]:       47,710 params → ~96.5% accuracy
    # network_1.1.0 [784, 30, 30, 10]:   24,790 params → ~95.5% accuracy
    # network_1.3.0 [784, 60, 60, 10]:   51,370 params → ~97.5% accuracy
    # 
    # CHAPTER 2 (Cross-Entropy, L2 Regularization):
    # network_2.0.0 [784, 30, 10]:       23,860 params → ~96.5% accuracy
    # network_2.2.0 [784, 100, 10]:      79,510 params → ~97.5% accuracy
    # network_2.2.1 [784, 60, 60, 10]:   51,370 params → ~97.5% accuracy
    # network_2.2.2 [784, 100, 100, 10]: 88,610 params → ~98%+ accuracy ★ (THIS NETWORK)

    improvement_over_baseline = final_validation_acc - 95.0
    print(f"\n   Total improvement over baseline: +{improvement_over_baseline:.1f}%")
    # Sources of improvement:
    #   • Cross-entropy cost:      ~+1.0-1.5%
    #   • Regularization:          ~+0.5-1.0%
    #   • Increased capacity:      ~+1.0-2.0%

    # Efficiency analysis
    accuracy_per_1k_params = final_validation_acc / (total_params / 1000)

    print("\n" + "-" * 75)
    print("EFFICIENCY ANALYSIS:")
    print("-" * 75)
    print("Accuracy per 1K parameters:")
    print(f"  Baseline [784,30,10]:      {95.0 / (23860/1000):.2f}%/1K params")
    print(f"  Wide [784,100,10]:         {97.5 / (79510/1000):.2f}%/1K params")
    print(f"  Deep [784,60,60,10]:       {97.5 / (51370/1000):.2f}%/1K params")
    print(f"  THIS NETWORK:              {accuracy_per_1k_params:.2f}%/1K params")
    print()
    print("Interpretation:")
    print("  • Diminishing returns from extra parameters (expected)")
    print("  • But ABSOLUTE accuracy is highest!")
    print("  • Trade-off: More compute for better performance")

    # WHY THIS ARCHITECTURE WORKS:
    # 
    # 1. MAXIMUM CAPACITY:
    #    • 100 neurons/layer = high parallelism
    #    • 2 layers = hierarchical composition
    #    • 10,000 possible feature combinations
    #    • Can learn very complex decision boundaries
    # 
    # 2. CROSS-ENTROPY ENABLES DEPTH:
    #    • No learning slowdown from saturation
    #    • Gradient ∝ error size (no σ'(z) bottleneck)
    #    • Both layers learn effectively
    #    • Fast convergence even with 88K parameters
    # 
    # 3. REGULARIZATION PREVENTS OVERFITTING:
    #    • Controls all 88,610 parameters
    #    • Weight decay: w → (1 - ηλ/n)·w
    #    • Encourages small, distributed weights
    print(f"    • Result: {overfitting_gap:.1f}% gap (excellent!)")
    # 
    # 4. VALIDATION MONITORING:
    #    • Tracks generalization during training
    #    • Prevents 'peeking' at test set
    #    • Enables proper hyperparameter tuning

    # THE LIMITS OF THIS APPROACH:
    # 
    # This network represents the PRACTICAL LIMIT of:
    #   • Sigmoid activation function
    #   • Cross-entropy cost
    #   • L2 regularization
    #   • Fully connected layers
    # 
    # To go beyond ~98% on MNIST, you need:
    # 
    # 1. BETTER ACTIVATION: ReLU instead of sigmoid
    #    • Solves vanishing gradient completely
    #    • Allows much deeper networks
    #    • Faster training
    # 
    # 2. BETTER INITIALIZATION: He/Xavier initialization
    #    • Prevents saturation at start
    #    • Better gradient flow from epoch 0
    # 
    # 3. MORE REGULARIZATION: Dropout
    #    • Randomly drops neurons during training
    #    • Forces redundancy and robustness
    #    • Reduces overfitting further
    # 
    # 4. SPECIALIZED ARCHITECTURES: Convolutional Neural Networks
    #    • Exploit spatial structure of images
    #    • Parameter sharing (far fewer parameters)
    #    • Translation invariance
    #    • Can achieve 99%+ accuracy!
    # 
    # These techniques are covered in network3.py and beyond!

    # WHAT YOU'VE LEARNED IN CHAPTER 2:
    # ✓ Cross-entropy cost solves learning slowdown
    # ✓ L2 regularization prevents overfitting
    # ✓ Validation monitoring enables proper evaluation
    # ✓ Modern techniques make large networks practical
    # ✓ Width provides capacity through parallelism
    # ✓ Depth provides hierarchy through composition
    # ✓ Combining both gives maximum performance
    print(f"\n   ✓ Achieved ~{final_validation_acc:.1f}% accuracy (vs ~95% in Chapter 1)")

    # WHAT'S NEXT:
    # • network3.py: Modern techniques (ReLU, dropout, better init)
    # • Convolutional networks: Specialized for images
    # • Modern frameworks: PyTorch, TensorFlow, JAX
    # • Deeper networks: ResNets, Transformers, etc.

    print("\n" + "=" * 75)
    print("Experiment complete!")
    print("=" * 75)
    print(f"\nCONGRATULATIONS! You've mastered the fundamentals of modern")
    print(f"neural network training, achieving ~{final_validation_acc:.1f}% accuracy!")

if __name__ == "__main__":
    main()
