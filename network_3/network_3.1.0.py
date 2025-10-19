"""
ReLU vs Sigmoid: Activation Function Comparison
================================================

PREREQUISITES:
- Complete network_3.0.x series (understand CNNs)
- Understand: backpropagation, gradient flow, vanishing gradients

THIS EXPERIMENT:
Direct comparison between ReLU (modern) and sigmoid (traditional) activation
functions. Demonstrates why ReLU has become the default choice for deep learning.

THE VANISHING GRADIENT PROBLEM:

Sigmoid Activation:
  σ(z) = 1/(1+e^(-z))
  
  Gradient: σ'(z) = σ(z)(1-σ(z))
  
  Problem:
    • When z is large positive: σ(z) ≈ 1, σ'(z) ≈ 0
    • When z is large negative: σ(z) ≈ 0, σ'(z) ≈ 0
    • Gradient vanishes at both extremes!
    
  In backpropagation:
    gradient = σ'(z₁) × σ'(z₂) × ... × σ'(zₙ)
    
    If σ'(zᵢ) ≈ 0 for multiple layers:
    → gradient ≈ 0 × 0 × ... × 0 ≈ 0
    → neurons stop learning (gradient vanishing)

ReLU Activation:
  ReLU(z) = max(0, z) = { z  if z > 0
                        { 0  if z ≤ 0
  
  Gradient: ReLU'(z) = { 1  if z > 0
                       { 0  if z ≤ 0
  
  Advantages:
    • No saturation for positive values (gradient = 1)
    • Simple computation (just thresholding!)
    • Sparse activation (many neurons output 0)
    • Gradient flows easily through deep networks

THE EXPERIMENT:

We train the SAME CNN architecture with two different activations:

Network A (ReLU):
  Conv (ReLU) → Pool → FC (ReLU) → Softmax
  Expected: 98% accuracy in ~5 epochs

Network B (Sigmoid):
  Conv (sigmoid) → Pool → FC (sigmoid) → Softmax
  Expected: 98% accuracy in ~15+ epochs

Result: ReLU converges 2-3× faster!

Expected Results:
- ReLU: Reaches 98% in ~5 epochs, ~98.5% final
- Sigmoid: Reaches 98% in ~15+ epochs, ~98.0% final
- Key lesson: Non-saturating activations enable fast learning

WHY RELU TRAINS FASTER:

1. GRADIENT MAGNITUDE
   Sigmoid: gradient ≤ 0.25 (max at z=0)
   ReLU: gradient = 1.0 (for z>0)
   Result: ReLU gradients are 4× stronger!

2. NO SATURATION
   Sigmoid: saturates at both extremes
   ReLU: never saturates for positive values
   Result: ReLU neurons keep learning!

3. SPARSE REPRESENTATIONS
   Sigmoid: all neurons always active (output in [0,1])
   ReLU: ~50% neurons inactive (output = 0)
   Result: More efficient representations!

NEXT STEPS:
- network_3.1.1.py: ReLU vs tanh comparison
- network_3.1.2.py: Deep networks (5+ layers) - sigmoid fails completely!
- network_3.2.x: Dropout experiments

Run: python network_3.1.0.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU, sigmoid

def main():
    # ========================================================================
    # EXPERIMENT: ReLU vs Sigmoid Activation Functions
    # ========================================================================
    print("=" * 75)
    print("ACTIVATION FUNCTION COMPARISON: ReLU vs Sigmoid")
    print("=" * 75)

    # Load MNIST data once (shared for both experiments)
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    # ========================================================================
    # EXPERIMENT A: Network with ReLU Activation
    # ========================================================================
    # ARCHITECTURE:
    # Conv layer (ReLU) → Max pooling → FC layer (ReLU) → Softmax
    #
    # WHY RELU WORKS:
    #   • Gradient = 1 for positive activations (no vanishing!)
    #   • No saturation for positive values
    #   • Simple computation: max(0, x)
    #   • Enables fast, efficient learning
    #
    # Expected behavior:
    #   • Epoch 1-3: Rapid learning (gradient flows well)
    #   • Epoch 3-5: Reaches 98% (fast convergence)
    #   • Epoch 5-20: Fine-tuning to ~98.5%
    
    print("\n[EXPERIMENT A: ReLU Network]")
    
    # Build ReLU network: Conv (ReLU) → Pool → FC (ReLU) → Softmax
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
    
    print("Training ReLU network (20 epochs)...")
    print("-" * 75)
    
    # Train ReLU network (20 epochs - converges fast!)
    net_relu.SGD(training_data, 20, mini_batch_size, 0.03, 
                 validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # EXPERIMENT B: Network with Sigmoid Activation
    # ========================================================================
    # ARCHITECTURE:
    # Conv layer (sigmoid) → Max pooling → FC layer (sigmoid) → Softmax
    #
    # WHY SIGMOID IS SLOWER:
    #   • Gradient ≤ 0.25 (max at z=0)
    #   • Saturates at both extremes (gradient ≈ 0)
    #   • Gradients multiply through layers → vanishing
    #   • Learning is much slower
    #
    # Expected behavior:
    #   • Epoch 1-5: Slow initial learning (weak gradients)
    #   • Epoch 5-15: Gradual improvement (fighting saturation)
    #   • Epoch 15-40: Slow convergence to ~98%
    #   • Takes 2-3× longer than ReLU to reach same accuracy
    
    print("\n[EXPERIMENT B: Sigmoid Network]")
    
    # Build sigmoid network: Conv (sigmoid) → Pool → FC (sigmoid) → Softmax
    layer1_sigmoid = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),
        image_shape=(mini_batch_size, 1, 28, 28),
        poolsize=(2, 2),
        activation_fn=sigmoid
    )
    
    layer2_sigmoid = FullyConnectedLayer(
        n_in=20*12*12,
        n_out=100,
        activation_fn=sigmoid,
        p_dropout=0.5
    )
    
    layer3_sigmoid = SoftmaxLayer(
        n_in=100,
        n_out=10,
        p_dropout=0.0
    )
    
    net_sigmoid = Network([layer1_sigmoid, layer2_sigmoid, layer3_sigmoid], mini_batch_size)
    
    print("Training sigmoid network (40 epochs - slower convergence)...")
    print("-" * 75)
    
    # Train sigmoid network (40 epochs - needs more due to slow convergence)
    net_sigmoid.SGD(training_data, 40, mini_batch_size, 0.03,
                    validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print("""
Expected Results:
  ReLU:    Fast convergence (~5 epochs to 98%), ~98.5% final
  Sigmoid: Slow convergence (~15+ epochs to 98%), ~98.0% final
  Speed-up: ReLU is 3× faster

Key Insight: Non-saturating activations (ReLU) enable efficient training
             by preventing vanishing gradients.

Mathematical Explanation:
  Sigmoid: σ'(z) ≤ 0.25 → gradients vanish through layers
  ReLU:    ReLU'(z) = 1 for z>0 → gradients flow perfectly

Next: network_3.1.1.py (ReLU vs tanh), network_3.1.2.py (deep networks)
""")

if __name__ == "__main__":
    main()

