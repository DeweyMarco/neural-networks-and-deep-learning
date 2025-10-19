"""
Deep Network Comparison: Why ReLU Enables Deep Learning
========================================================

PREREQUISITES:
- Complete network_3.1.0.py (ReLU vs sigmoid)
- Complete network_3.1.1.py (ReLU vs tanh)
- Understand: vanishing gradients, saturation

THIS EXPERIMENT:
Demonstrates the CATASTROPHIC FAILURE of sigmoid/tanh in deep networks.
Shows why ReLU enabled the "deep learning revolution."

THE CRITICAL INSIGHT:

Shallow networks (2-3 layers):
  • Sigmoid/tanh work OK (gradients don't vanish too much)
  • ReLU is faster but not essential

Deep networks (5+ layers):
  • Sigmoid/tanh FAIL COMPLETELY (gradients vanish to zero)
  • ReLU SUCCEEDS (gradients flow through all layers)
  • This is why deep learning wasn't possible before ReLU!

THE VANISHING GRADIENT CATASTROPHE:

In a 5-layer network with sigmoid:
  gradient_layer1 = σ'(z₁) × σ'(z₂) × σ'(z₃) × σ'(z₄) × σ'(z₅)
  
  Each σ'(z) ≤ 0.25 (max gradient of sigmoid)
  
  Worst case: 0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.00098
  
  → Gradient is 1000× smaller at layer 1 than at layer 5!
  → Layer 1 essentially stops learning
  → Network can't learn deep representations

With typical saturation (σ'(z) ≈ 0.1):
  0.1 × 0.1 × 0.1 × 0.1 × 0.1 = 0.00001
  
  → Gradient is 100,000× smaller!
  → Layer 1 learns NOTHING
  → Network fails to converge

With ReLU:
  gradient_layer1 = 1 × 1 × 1 × 1 × 1 = 1.0
  
  → Gradient flows perfectly through all layers!
  → All layers learn at similar rates
  → Deep networks work!

THE EXPERIMENT:

We build DEEP (5-layer) networks with different activations:

Network A (ReLU):
  FC (ReLU) → FC (ReLU) → FC (ReLU) → FC (ReLU) → FC (ReLU) → Softmax
  Expected: Learns successfully, ~97.5% accuracy

Network B (Sigmoid):
  FC (sigmoid) → FC (sigmoid) → FC (sigmoid) → FC (sigmoid) → FC (sigmoid) → Softmax
  Expected: FAILS to learn, ~10-50% accuracy (barely better than random!)

Network C (Tanh):
  FC (tanh) → FC (tanh) → FC (tanh) → FC (tanh) → FC (tanh) → Softmax
  Expected: Struggles to learn, ~80-90% accuracy (poor performance)

This is the SMOKING GUN demonstration of why ReLU revolutionized deep learning!

Expected Results:
- ReLU: ~97.5% accuracy (learns successfully)
- Tanh: ~85% accuracy (struggles with depth)
- Sigmoid: ~30-50% accuracy (catastrophic failure!)
- Key lesson: Depth REQUIRES non-saturating activations

WHY THIS MATTERS:

Before ReLU (pre-2012):
  • Deep networks didn't work
  • Limited to 2-3 layer networks
  • Couldn't learn complex representations
  • "Deep learning" was impossible!

After ReLU (2012+):
  • Networks with 10, 50, 100+ layers possible
  • State-of-the-art in computer vision
  • Enabled the deep learning revolution
  • Modern AI as we know it!

NEXT STEPS:
- network_3.2.x: Dropout regularization experiments
- network_3.3.x: Optimized deep CNNs for 99.5%+ accuracy

Run: python network_3.1.2.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, FullyConnectedLayer, SoftmaxLayer, ReLU, sigmoid, tanh

def main():
    # ========================================================================
    # EXPERIMENT: Deep Networks - ReLU vs Sigmoid vs Tanh
    # ========================================================================
    # This experiment demonstrates why sigmoid/tanh FAIL in deep networks
    # while ReLU SUCCEEDS. This is one of the most important insights in
    # modern deep learning!
    
    print("=" * 75)
    print("DEEP NETWORKS: Why ReLU Enabled the Deep Learning Revolution")
    print("=" * 75)

    # Load MNIST data
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10

    # ========================================================================
    # EXPERIMENT A: Deep Network with ReLU (SHOULD SUCCEED)
    # ========================================================================
    # ARCHITECTURE: 5 hidden layers [784]→[100]→[80]→[60]→[40]→[20]→[10]
    #
    # WHY RELU WORKS IN DEEP NETWORKS:
    #   • Gradient = 1 for positive activations
    #   • No gradient multiplication decay
    #   • Layer 1 receives same gradient strength as layer 5
    #   • All layers learn at similar rates
    #
    # Expected: All layers learn, smooth convergence, ~97.5% accuracy
    
    print("\n[EXPERIMENT A: 5-Layer Deep Network with ReLU]")
    
    # Build 5-layer ReLU network (progressively compressing representation)
    
    layer1_relu = FullyConnectedLayer(
        n_in=784, n_out=100,
        activation_fn=ReLU,
        p_dropout=0.1  # Light dropout (deep network needs less)
    )
    
    layer2_relu = FullyConnectedLayer(
        n_in=100, n_out=80,
        activation_fn=ReLU,
        p_dropout=0.1
    )
    
    layer3_relu = FullyConnectedLayer(
        n_in=80, n_out=60,
        activation_fn=ReLU,
        p_dropout=0.1
    )
    
    layer4_relu = FullyConnectedLayer(
        n_in=60, n_out=40,
        activation_fn=ReLU,
        p_dropout=0.1
    )
    
    layer5_relu = FullyConnectedLayer(
        n_in=40, n_out=20,
        activation_fn=ReLU,
        p_dropout=0.1
    )
    
    output_relu = SoftmaxLayer(
        n_in=20, n_out=10,
        p_dropout=0.0
    )
    
    net_relu = Network(
        [layer1_relu, layer2_relu, layer3_relu, layer4_relu, layer5_relu, output_relu],
        mini_batch_size
    )
    
    print("Training deep ReLU network (30 epochs, lr=0.1)...")
    print("-" * 75)
    
    # Train ReLU network (can use higher learning rate)
    net_relu.SGD(training_data, 30, mini_batch_size, 0.1,
                 validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # EXPERIMENT B: Deep Network with Tanh (SHOULD STRUGGLE)
    # ========================================================================
    # ARCHITECTURE: Same 5 hidden layers [784]→[100]→[80]→[60]→[40]→[20]→[10]
    #
    # WHY TANH STRUGGLES IN DEEP NETWORKS:
    #   • Gradient ≤ 1.0, typically much smaller due to saturation
    #   • 5 layers → gradient multiplied 5 times
    #   • If tanh'(z) ≈ 0.2 per layer: 0.2^5 = 0.00032
    #   • Layer 1 receives 3000× weaker gradient than layer 5
    #
    # Expected: Early layers barely learn, slow convergence, ~85% accuracy
    
    print("\n[EXPERIMENT B: 5-Layer Deep Network with Tanh]")
    
    # Build 5-layer tanh network
    
    layer1_tanh = FullyConnectedLayer(
        n_in=784, n_out=100,
        activation_fn=tanh,
        p_dropout=0.1
    )
    
    layer2_tanh = FullyConnectedLayer(
        n_in=100, n_out=80,
        activation_fn=tanh,
        p_dropout=0.1
    )
    
    layer3_tanh = FullyConnectedLayer(
        n_in=80, n_out=60,
        activation_fn=tanh,
        p_dropout=0.1
    )
    
    layer4_tanh = FullyConnectedLayer(
        n_in=60, n_out=40,
        activation_fn=tanh,
        p_dropout=0.1
    )
    
    layer5_tanh = FullyConnectedLayer(
        n_in=40, n_out=20,
        activation_fn=tanh,
        p_dropout=0.1
    )
    
    output_tanh = SoftmaxLayer(
        n_in=20, n_out=10,
        p_dropout=0.0
    )
    
    net_tanh = Network(
        [layer1_tanh, layer2_tanh, layer3_tanh, layer4_tanh, layer5_tanh, output_tanh],
        mini_batch_size
    )
    
    print("Training deep tanh network (30 epochs, lr=0.03)...")
    print("-" * 75)
    
    # Train tanh network (must use lower learning rate)
    net_tanh.SGD(training_data, 30, mini_batch_size, 0.03,
                 validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # EXPERIMENT C: Deep Network with Sigmoid (SHOULD FAIL CATASTROPHICALLY)
    # ========================================================================
    # ARCHITECTURE: Same 5 hidden layers [784]→[100]→[80]→[60]→[40]→[20]→[10]
    #
    # WHY SIGMOID FAILS CATASTROPHICALLY IN DEEP NETWORKS:
    #   • Gradient ≤ 0.25, typically much smaller due to saturation
    #   • 5 layers → gradient multiplied 5 times
    #   • If σ'(z) ≈ 0.15 per layer: 0.15^5 = 0.0000759
    #   • Layer 1 receives 13,000× weaker gradient than layer 5!
    #   • Early layers essentially frozen
    #
    # Expected: Catastrophic failure, ~40-50% accuracy (barely better than random!)
    
    print("\n[EXPERIMENT C: 5-Layer Deep Network with Sigmoid]")
    
    # Build 5-layer sigmoid network
    
    layer1_sigmoid = FullyConnectedLayer(
        n_in=784, n_out=100,
        activation_fn=sigmoid,
        p_dropout=0.1
    )
    
    layer2_sigmoid = FullyConnectedLayer(
        n_in=100, n_out=80,
        activation_fn=sigmoid,
        p_dropout=0.1
    )
    
    layer3_sigmoid = FullyConnectedLayer(
        n_in=80, n_out=60,
        activation_fn=sigmoid,
        p_dropout=0.1
    )
    
    layer4_sigmoid = FullyConnectedLayer(
        n_in=60, n_out=40,
        activation_fn=sigmoid,
        p_dropout=0.1
    )
    
    layer5_sigmoid = FullyConnectedLayer(
        n_in=40, n_out=20,
        activation_fn=sigmoid,
        p_dropout=0.1
    )
    
    output_sigmoid = SoftmaxLayer(
        n_in=20, n_out=10,
        p_dropout=0.0
    )
    
    net_sigmoid = Network(
        [layer1_sigmoid, layer2_sigmoid, layer3_sigmoid, layer4_sigmoid, layer5_sigmoid, output_sigmoid],
        mini_batch_size
    )
    
    print("Training deep sigmoid network (30 epochs, lr=0.03)...")
    print("Expect catastrophic failure due to vanishing gradients!")
    print("-" * 75)
    
    # Train sigmoid network (watch it fail!)
    net_sigmoid.SGD(training_data, 30, mini_batch_size, 0.03,
                    validation_data, test_data, lmbda=0.1)

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print("""
Expected Results (5-layer deep networks):
  ReLU:    ~97.5% accuracy (SUCCESS! All layers learn)
  Tanh:    ~85.0% accuracy (struggles, early layers barely learn)
  Sigmoid: ~45.0% accuracy (CATASTROPHIC FAILURE!)

Performance Gap: 52.5% difference between ReLU and sigmoid!

Why This Matters:
  This demonstrates why deep learning wasn't possible before ReLU.
  Before 2012: Networks limited to 2-3 layers (sigmoid/tanh fail deeper)
  After 2012:  AlexNet with ReLU enabled 8+ layer networks
  Today:       Networks with 100+ layers are standard (ResNet, etc.)

Mathematical Explanation:
  ReLU:    gradient = 1 × 1 × 1 × 1 × 1 = 1.0 (perfect flow!)
  Tanh:    gradient ≈ 0.3^5 = 0.00243 (412× weaker)
  Sigmoid: gradient ≈ 0.15^5 = 0.000076 (13,000× weaker!)

Key Insight:
  Deep networks (5+ layers) REQUIRE non-saturating activations.
  ReLU made modern deep learning possible!

""")

if __name__ == "__main__":
    main()

