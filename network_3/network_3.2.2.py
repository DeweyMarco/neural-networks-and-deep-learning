"""
Dropout vs L2 Regularization: Direct Comparison
================================================

PREREQUISITES:
- Complete network_3.2.0.py (understand dropout)
- Complete network_3.2.1.py (understand dropout rates)
- Complete Chapter 2 (understand L2 regularization)

THIS EXPERIMENT:
Direct head-to-head comparison between dropout and L2 regularization.
Uses identical CNN architectures to isolate the effect of each technique.

THE TWO REGULARIZATION APPROACHES:

L2 Regularization (Weight Decay):
  • Mechanism: Add penalty term to cost function
  • Cost: C = C₀ + (λ/2n) × Σw²
  • Effect: Penalizes large weights, encourages small weights
  • Gradient: ∂C/∂w = ∂C₀/∂w + (λ/n)w
  • Result: Weights decay toward zero during training
  
  How it prevents overfitting:
    - Large weights → complex decision boundaries → overfitting
    - L2 forces small weights → simpler boundaries → better generalization

Dropout Regularization:
  • Mechanism: Randomly drop neurons during training
  • Each neuron dropped with probability p (typically 0.5)
  • Effect: Prevents co-adaptation, creates ensemble
  • At test time: Use all neurons (scaled by 1-p)
  
  How it prevents overfitting:
    - Prevents neurons from co-adapting
    - Forces redundant representations
    - Ensemble of 2^n networks
    - More robust to variations

KEY DIFFERENCES:

1. MECHANISM
   L2: Continuous weight regularization (affects all updates)
   Dropout: Discrete neuron regularization (stochastic)

2. WHAT THEY REGULARIZE
   L2: Weight magnitudes (penalizes large weights)
   Dropout: Neural dependencies (prevents co-adaptation)

3. COMPUTATIONAL COST
   L2: Nearly free (just add penalty term)
   Dropout: Small cost (random mask generation and scaling)

4. WHEN THEY WORK BEST
   L2: Small to medium networks, moderate overfitting
   Dropout: Large networks with many parameters

5. COMBINATION
   L2 + Dropout: YES! They complement each other
   Result: Often better than either alone

THE EXPERIMENT:

We train three identical CNN architectures:

Network A (L2 Only):
  Conv → Conv → FC (no dropout) → Softmax
  L2 regularization: λ=0.1
  Expected: ~98.7%

Network B (Dropout Only):
  Conv → Conv → FC (dropout 0.5) → Softmax
  No L2 regularization: λ=0
  Expected: ~98.9%

Network C (Baseline - No Regularization):
  Conv → Conv → FC (no dropout) → Softmax
  No L2: λ=0
  Expected: ~98.0% (overfitting!)

Result: Dropout slightly better for this large network!

Expected Results:
- No regularization: ~98.0% (overfits)
- L2 only: ~98.7% (good)
- Dropout only: ~98.9% (slightly better!)
- Key lesson: Dropout > L2 for large networks with many parameters

WHY DROPOUT WINS FOR LARGE NETWORKS:

The FC layer has 640×100 = 64,000 parameters!

L2 Regularization:
  • Shrinks all 64,000 weights
  • Reduces weight magnitudes
  • Good: Prevents extremely large weights
  • Limitation: Single mechanism (weight shrinkage)

Dropout:
  • Prevents 100 neurons from co-adapting
  • Multiple mechanisms:
    - Co-adaptation prevention
    - Ensemble effect (2^100 networks!)
    - Redundant representations
  • More powerful for many-parameter networks

For Small Networks (<1000 parameters):
  L2 and dropout perform similarly

For Large Networks (>10,000 parameters):
  Dropout typically outperforms L2

For Very Large Networks (>1M parameters):
  Dropout is essential, L2 alone insufficient

WHEN TO USE EACH:

Use L2 Regularization:
  ✓ Small to medium networks
  ✓ When computational cost matters
  ✓ When want interpretable weight magnitudes
  ✓ Traditional machine learning models

Use Dropout:
  ✓ Large neural networks (default choice!)
  ✓ Deep learning models
  ✓ When severe overfitting occurs
  ✓ Modern CNN/RNN/Transformer architectures

Use Both (L2 + Dropout):
  ✓ Best generalization (see network_3.2.3.py)
  ✓ They complement each other!
  ✓ State-of-the-art architectures
  ✓ When every 0.1% accuracy matters

NEXT STEPS:
- network_3.2.3.py: Combined L2 + Dropout for maximum performance (~99.5%)

Run: python network_3.2.2.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

def main():
    # ========================================================================
    # EXPERIMENT: Dropout vs L2 Regularization
    # ========================================================================
    print("=" * 75)
    print("DROPOUT vs L2 REGULARIZATION: Direct Comparison")
    print("=" * 75)

    # Load MNIST data
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    # ========================================================================
    # EXPERIMENT A: Baseline - No Regularization
    # ========================================================================
    # PURPOSE: Establish baseline to show overfitting problem
    #
    # ARCHITECTURE:
    #   Conv (20 filters) → Pool → Conv (40 filters) → Pool
    #   → FC (100 neurons, NO DROPOUT) → Softmax
    #
    # NO REGULARIZATION:
    #   • No dropout (p_dropout=0.0)
    #   • No L2 (lmbda=0.0)
    #
    # EXPECTED BEHAVIOR:
    #   • 64,000 parameters in FC layer → severe overfitting
    #   • Training accuracy: ~100% (memorizes training data)
    #   • Validation accuracy: ~98.0%
    #   • Gap: ~2% (clear overfitting signal)
    #
    # This establishes that regularization is necessary!
    
    print("\n" + "=" * 75)
    print("[EXPERIMENT A: Baseline - NO Regularization]")
    print("=" * 75)
    
    # Build baseline network (no regularization)
    layer1_baseline = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer2_baseline = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),
        image_shape=(mini_batch_size, 20, 12, 12),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer3_baseline = FullyConnectedLayer(
        n_in=40*4*4,
        n_out=100,
        activation_fn=ReLU,
        p_dropout=0.0                            # NO DROPOUT
    )
    
    layer4_baseline = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_baseline = Network([layer1_baseline, layer2_baseline, layer3_baseline, layer4_baseline],
                          mini_batch_size)
    
    print("\nTraining baseline (NO regularization)...")
    
    # Train with NO regularization
    net_baseline.SGD(training_data, 30, mini_batch_size, 0.03,
                    validation_data, test_data, lmbda=0.0)   # NO L2!

    # ========================================================================
    # EXPERIMENT B: L2 Regularization Only
    # ========================================================================
    # L2 REGULARIZATION (Weight Decay):
    #   • Adds penalty: Cost = C₀ + (λ/2n) × Σw²
    #   • λ = 0.1 (standard value from Chapter 2)
    #   • Effect: Shrinks all weights toward zero
    #   • Prevents individual weights from getting too large
    #
    # ARCHITECTURE:
    #   Conv → Conv → FC (NO dropout) → Softmax
    #   With L2 penalty on all weights
    #
    # EXPECTED BEHAVIOR:
    #   • L2 penalizes large weights
    #   • Forces simpler decision boundaries
    #   • Training accuracy: ~99% (can't fully memorize)
    #   • Validation accuracy: ~98.7%
    #   • Gap: ~0.3% (much better than baseline!)
    #
    # L2 works by:
    #   • Weight update: w → (1-ηλ/n)w - η∇C₀
    #   • First term (1-ηλ/n) < 1 causes weight decay
    #   • Prevents weights from growing too large
    
    print("\n" + "=" * 75)
    print("[EXPERIMENT B: L2 Regularization Only]")
    print("=" * 75)
    
    # Build L2-only network
    layer1_l2 = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer2_l2 = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),
        image_shape=(mini_batch_size, 20, 12, 12),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer3_l2 = FullyConnectedLayer(
        n_in=40*4*4,
        n_out=100,
        activation_fn=ReLU,
        p_dropout=0.0                            # NO DROPOUT
    )
    
    layer4_l2 = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_l2 = Network([layer1_l2, layer2_l2, layer3_l2, layer4_l2],
                    mini_batch_size)
    
    print("\nTraining with L2 regularization only (λ=0.1)...")
    
    # Train with L2 only
    net_l2.SGD(training_data, 30, mini_batch_size, 0.03,
              validation_data, test_data, lmbda=0.1)        # L2 only!

    # ========================================================================
    # EXPERIMENT C: Dropout Only
    # ========================================================================
    # DROPOUT REGULARIZATION:
    #   • Randomly drops 50% of neurons (p=0.5)
    #   • Different neurons dropped each mini-batch
    #   • Forces neurons to work independently
    #   • Creates ensemble of 2^100 ≈ 10^30 networks!
    #
    # ARCHITECTURE:
    #   Conv → Conv → FC (dropout 0.5) → Softmax
    #   NO L2 penalty
    #
    # EXPECTED BEHAVIOR:
    #   • Dropout prevents co-adaptation
    #   • Ensemble effect improves generalization
    #   • Training accuracy: ~99% (can't memorize)
    #   • Validation accuracy: ~98.9%
    #   • Gap: ~0.1% (excellent generalization!)
    #
    # Dropout works by:
    #   • Training: Randomly set 50% of activations to 0
    #   • Each training example sees different sub-network
    #   • Network can't rely on any specific neurons
    #   • Forced to learn redundant, robust features
    #   • Testing: Use all neurons (scaled by 0.5)
    
    print("\n" + "=" * 75)
    print("[EXPERIMENT C: Dropout Only]")
    print("=" * 75)
    
    # Build dropout-only network
    layer1_dropout = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer2_dropout = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),
        image_shape=(mini_batch_size, 20, 12, 12),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer3_dropout = FullyConnectedLayer(
        n_in=40*4*4,
        n_out=100,
        activation_fn=ReLU,
        p_dropout=0.5                            # DROPOUT!
    )
    
    layer4_dropout = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_dropout = Network([layer1_dropout, layer2_dropout, layer3_dropout, layer4_dropout],
                         mini_batch_size)
    
    print("\nTraining with dropout only (p=0.5)...")
    
    # Train with dropout only
    net_dropout.SGD(training_data, 30, mini_batch_size, 0.03,
                   validation_data, test_data, lmbda=0.0)   # NO L2!

    # ========================================================================
    # Results Summary and Analysis
    # ========================================================================
    # EXPECTED RESULTS:
    # ┌───────────────────┬─────────────┬──────────────┬─────────┬──────────────┐
    # │ Regularization    │ Train Acc   │ Val Accuracy │ Gap     │ Assessment   │
    # ├───────────────────┼─────────────┼──────────────┼─────────┼──────────────┤
    # │ None (baseline)   │   ~100%     │    ~98.0%    │ ~2.0%   │ Overfitting! │
    # │ L2 only (λ=0.1)   │   ~99%      │    ~98.7%    │ ~0.3%   │ Good         │
    # │ Dropout (p=0.5)   │   ~99%      │    ~98.9%    │ ~0.1%   │ Better! ✓    │
    # └───────────────────┴─────────────┴──────────────┴─────────┴──────────────┘
    #
    # PERFORMANCE RANKING:
    #   1. Dropout:          98.9% ⭐ WINNER for single technique
    #   2. L2:               98.7% (good but slightly worse)
    #   3. No regularization: 98.0% (unacceptable overfitting)
    #
    # IMPROVEMENT OVER BASELINE:
    #   • L2:     +0.7% (70% error reduction)
    #   • Dropout: +0.9% (90% error reduction)
    #   • Winner: Dropout reduces error 29% more than L2!
    #
    # WHY DROPOUT WINS FOR THIS NETWORK:
    #
    # 1. NETWORK SIZE MATTERS
    #    FC Layer Parameters: 640×100 = 64,000 parameters
    #    
    #    L2 Approach:
    #      • Penalizes weight magnitude
    #      • Single regularization mechanism
    #      • All weights treated similarly
    #      • Good but limited
    #    
    #    Dropout Approach:
    #      • Prevents co-adaptation
    #      • Multiple mechanisms: ensemble + redundancy + independence
    #      • 2^100 ≈ 10^30 sub-networks trained!
    #      • More powerful for large parameter count
    #
    # 2. REGULARIZATION MECHANISMS COMPARED
    #
    #    L2 Mechanism:
    #      • Adds (λ/2n)Σw² to cost
    #      • Shrinks weights toward zero
    #      • Weight update: w → (1-ηλ/n)w - η∇C
    #      • Result: Prevents extreme weights, simpler boundaries
    #      • Limitation: Single mechanism, all weights regularized uniformly
    #    
    #    Dropout Mechanism:
    #      • Randomly drops neurons (p=0.5)
    #      • Forces independence between neurons
    #      • Ensemble of exponentially many networks
    #      • Result: Multiple mechanisms working together
    #      • Advantage: Stronger for large networks
    #
    # 3. CO-ADAPTATION PROBLEM (Why Dropout is Powerful)
    #    
    #    Without Dropout:
    #      Neuron 42: "I detect loops at top"
    #      Neuron 67: "If neuron 42 is active → digit 9"
    #      Problem: Neuron 67 DEPENDS on neuron 42 (fragile!)
    #    
    #    L2 Regularization:
    #      Neuron 42: Weight = 0.8 (shrunk by L2)
    #      Neuron 67: Weight = 0.7 (shrunk by L2)
    #      Helps but: Still allows co-adaptation!
    #    
    #    Dropout:
    #      Mini-batch 1: Neuron 42 dropped!
    #        → Neuron 67 must work without neuron 42
    #      Mini-batch 2: Neuron 67 dropped!
    #        → Neuron 42 must provide useful signal
    #      Result: Both neurons learn independently!
    #
    # 4. WHEN EACH REGULARIZATION WINS
    #
    #    Small Networks (<1K parameters):
    #      L2 and dropout: Similar performance (use L2 for simplicity)
    #
    #    Medium Networks (1K-100K parameters):
    #      Example: This network = ~85,000 parameters
    #      L2: 98.7%, Dropout: 98.9% → Dropout wins (marginal advantage)
    #
    #    Large Networks (100K-1M parameters):
    #      L2: Helps but insufficient
    #      Dropout: Essential for good performance (clear advantage)
    #
    #    Very Large Networks (>1M parameters):
    #      L2: Alone is insufficient
    #      Dropout: Critical component (mandatory)
    #
    # 5. COMPUTATIONAL COMPARISON
    #
    #    L2 Regularization: ~0% overhead (negligible)
    #    Dropout:          ~2-5% overhead (minimal)
    #    
    #    Both are computationally cheap!
    #    Choose based on effectiveness, not speed.
    #
    # REAL-WORLD APPLICATIONS:
    #
    # AlexNet (2012):
    #   • Used dropout (p=0.5) in FC layers
    #   • Won ImageNet competition
    #   • Without dropout: 10% worse accuracy!
    #
    # VGGNet (2014):
    #   • Dropout (p=0.5) + L2 (λ=5e-4)
    #   • Standard architecture pattern
    #
    # Modern Transformers (BERT, GPT):
    #   • Dropout everywhere (attention, FC, embeddings)
    #   • Typically p=0.1-0.2
    #   • Can't train without dropout!
    #
    # PRACTICAL RECOMMENDATIONS:
    #
    # Small Networks (<10K parameters):
    #   → Use L2 (λ=0.1 to 5.0) - simpler, works well
    #
    # Medium Networks (10K-100K parameters):
    #   → Use dropout (p=0.5) on FC layers
    #   → Combine with light L2 (λ=0.1)
    #
    # Large Networks (>100K parameters):
    #   → Use dropout (p=0.5) on FC layers
    #   → Add light L2 (λ=0.01-0.1)
    #
    # Default Recipe (when in doubt):
    #   → Dropout p=0.5 on all FC layers
    #   → L2 λ=0.1 on all weights
    #   → This works 90% of the time!
    #
    # WHAT YOU'VE LEARNED:
    #   ✓ How L2 and dropout differ mechanically
    #   ✓ Why dropout is stronger for large networks
    #   ✓ How co-adaptation hurts generalization
    #   ✓ Why dropout's ensemble effect is powerful
    #   ✓ When to use L2 vs dropout vs both
    #   ✓ How to choose regularization for your network
    #
    # PRACTICAL TAKEAWAY:
    #   For most cases: Dropout (p=0.5) > L2 (λ=0.1) for large networks
    #   But combining both is even better! (see next experiment)
    
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print("Next: network_3.2.3.py to combine BOTH for maximum performance!")

if __name__ == "__main__":
    main()

