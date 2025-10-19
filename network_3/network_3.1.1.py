"""
ReLU vs Tanh: Activation Function Comparison
=============================================

PREREQUISITES:
- Complete network_3.1.0.py (ReLU vs sigmoid)
- Understand: vanishing gradients, saturation, activation functions

THIS EXPERIMENT:
Compares ReLU vs tanh (hyperbolic tangent) activation functions.
Tanh is BETTER than sigmoid but still saturates. ReLU remains superior.

THE THREE ACTIVATION FUNCTIONS:

Sigmoid: σ(z) = 1/(1+e^(-z))
  • Output range: (0, 1)
  • Gradient: σ'(z) = σ(z)(1-σ(z))
  • Max gradient: 0.25 (at z=0)
  • Problem: Saturates at both ends, small gradients

Tanh: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
  • Output range: (-1, 1)
  • Gradient: tanh'(z) = 1 - tanh²(z)
  • Max gradient: 1.0 (at z=0)
  • Advantage over sigmoid: Zero-centered outputs, 4× stronger gradient
  • Problem: Still saturates at both ends!

ReLU: ReLU(z) = max(0, z)
  • Output range: [0, ∞)
  • Gradient: 1 if z>0, 0 if z≤0
  • Max gradient: 1.0 (for all z>0)
  • Advantage: Never saturates for positive values!
  • Best choice for deep learning

WHY TANH IS BETTER THAN SIGMOID:

1. STRONGER GRADIENTS
   Sigmoid: max gradient = 0.25
   Tanh: max gradient = 1.0
   Result: Tanh learns 4× faster!

2. ZERO-CENTERED OUTPUT
   Sigmoid: output ∈ (0, 1) → always positive
   Tanh: output ∈ (-1, 1) → zero-centered
   
   Why zero-centered matters:
     • If all inputs to next layer are positive (sigmoid):
       All weights either all increase or all decrease together
       Zig-zag path to optimum (inefficient optimization)
     
     • If inputs are zero-centered (tanh):
       Weights can increase/decrease independently
       Direct path to optimum (efficient optimization)

3. BETTER NUMERICAL PROPERTIES
   Sigmoid: outputs near 0 or 1 → multiplication causes numerical issues
   Tanh: outputs near -1 or 1 → better numerical stability

WHY RELU STILL BEATS TANH:

1. NO SATURATION (for positive values)
   Tanh: saturates at +1 and -1 (gradient → 0)
   ReLU: never saturates for z>0 (gradient = 1)
   
   In deep networks:
     Tanh: gradients still vanish (just slower than sigmoid)
     ReLU: gradients flow perfectly (no vanishing)

2. COMPUTATIONAL EFFICIENCY
   Tanh: requires exp computation (slow)
   ReLU: max(0, x) (trivial, very fast)

3. SPARSE REPRESENTATIONS
   Tanh: all neurons active (output ≠ 0)
   ReLU: ~50% neurons inactive (output = 0)
   Result: ReLU is more efficient

THE EXPERIMENT:

Same CNN architecture, two different activations:

Network A (ReLU):
  Conv (ReLU) → Pool → FC (ReLU) → Softmax
  Expected: Reaches 98% in ~5 epochs

Network B (Tanh):
  Conv (tanh) → Pool → FC (tanh) → Softmax
  Expected: Reaches 98% in ~10 epochs

Result: ReLU converges ~2× faster than tanh!

Expected Results:
- ReLU: Reaches 98% in ~5 epochs, ~98.5% final
- Tanh: Reaches 98% in ~10 epochs, ~98.3% final
- Key lesson: Non-saturating activations (ReLU) win even vs better alternatives

WHEN TO USE EACH:

Sigmoid:
  • Output layer for binary classification
  • LSTM/GRU gates
  • When you need outputs in (0,1)

Tanh:
  • RNN hidden states (zero-centered helps)
  • When you need outputs in (-1,1)
  • Legacy code (historical)

ReLU:
  • DEFAULT CHOICE for all hidden layers
  • CNNs, deep feedforward networks
  • Modern architectures

NEXT STEPS:
- network_3.1.2.py: Deep networks (5 layers) - tanh/sigmoid fail, ReLU succeeds!
- network_3.2.x: Dropout regularization

Run: python network_3.1.1.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU, tanh

def main():
    # ========================================================================
    # EXPERIMENT: ReLU vs Tanh Activation Functions
    # ========================================================================
    print("=" * 75)
    print("ACTIVATION FUNCTION COMPARISON: ReLU vs Tanh")
    print("=" * 75)

    # Load MNIST data
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    # ========================================================================
    # EXPERIMENT A: Network with ReLU Activation
    # ========================================================================
    # ARCHITECTURE: Conv (ReLU) → Pool → FC (ReLU) → Softmax
    #
    # ReLU PROPERTIES:
    #   • f(z) = max(0, z)
    #   • f'(z) = 1 if z>0, else 0
    #   • Never saturates for positive values
    #   • Gradient = 1 for all positive activations
    #   • Fastest convergence
    #
    # Expected: Epoch 1-5 rapid learning, reaches ~98.5% by epoch 15
    
    print("\n[EXPERIMENT A: ReLU Network]")
    
    # Build ReLU network
    layer1_relu = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=ReLU
    )
    
    layer2_relu = FullyConnectedLayer(
        n_in=20*12*12,
        n_out=100,
        activation_fn=ReLU,
        p_dropout=0.5
    )
    
    layer3_relu = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_relu = Network([layer1_relu, layer2_relu, layer3_relu], mini_batch_size)
    
    print("Training ReLU network (15 epochs)...")
    print("-" * 75)
    
    # Train ReLU network
    net_relu.SGD(training_data, 15, mini_batch_size, 0.03,
                 validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # EXPERIMENT B: Network with Tanh Activation
    # ========================================================================
    # ARCHITECTURE: Conv (tanh) → Pool → FC (tanh) → Softmax
    #
    # TANH PROPERTIES:
    #   • f(z) = (e^z - e^(-z))/(e^z + e^(-z))
    #   • f'(z) = 1 - tanh²(z)
    #   • Output range: (-1, 1), max gradient: 1.0 at z=0
    #   • Saturates at both ends (gradient → 0)
    #
    # Expected: Good initial learning, but saturation slows convergence
    #           Reaches ~98.3% by epoch 30 (slower than ReLU)
    
    print("\n[EXPERIMENT B: Tanh Network]")
    
    # Build tanh network
    layer1_tanh = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=tanh
    )
    
    layer2_tanh = FullyConnectedLayer(
        n_in=20*12*12,
        n_out=100,
        activation_fn=tanh,
        p_dropout=0.5
    )
    
    layer3_tanh = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_tanh = Network([layer1_tanh, layer2_tanh, layer3_tanh], mini_batch_size)
    
    print("Training tanh network (30 epochs - more needed due to saturation)...")
    print("-" * 75)
    
    # Train tanh network (30 epochs - needs more due to saturation)
    net_tanh.SGD(training_data, 30, mini_batch_size, 0.03,
                 validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print("""
Expected Results:
  ReLU: Fast convergence (~5 epochs to 98%), ~98.5% final
  Tanh: Moderate convergence (~10 epochs to 98%), ~98.3% final
  Speed-up: ReLU is 2× faster

Key Insight: Non-saturating activations (ReLU) beat even improved saturating
             activations (tanh). Saturation is the enemy!

Gradient Comparison:
  Sigmoid: max gradient 0.25 → saturates easily
  Tanh:    max gradient 1.0 → better, but still saturates at extremes (|z|>3)
  ReLU:    gradient = 1 for all z>0 → no saturation!

Why Tanh is Better than Sigmoid:
  • Zero-centered outputs → better optimization
  • 4× stronger gradient (1.0 vs 0.25)
  • But still saturates at both extremes

Why ReLU Beats Both:
  • No saturation for positive values
  • 5-10× faster to compute
  • Sparse activation (implicit regularization)
  • Enables deep networks (5+ layers)

Activation Function Ranking:
  1. ReLU (no saturation for z>0) - default choice
  2. Tanh (zero-centered, 4× better than sigmoid) - for RNNs
  3. Sigmoid (saturates easily) - only for binary outputs

Next: network_3.1.2.py (deep networks - tanh fails catastrophically!)
""")

if __name__ == "__main__":
    main()

