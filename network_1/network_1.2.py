"""
Wider Neural Network (Version 1.2)
===================================

WHAT'S NEW IN VERSION 1.2:
This script demonstrates the effect of INCREASING NETWORK WIDTH by doubling
the number of neurons in the hidden layer from 30 to 60.

WIDTH vs DEPTH COMPARISON:
- network_1.0.py: [784, 30, 10] - baseline (23,860 parameters)
- network_1.1.py: [784, 30, 30, 10] - DEEPER (24,790 parameters, +930)
- network_1.2.py: [784, 60, 10] - WIDER (47,710 parameters, +23,850)

ARCHITECTURE:
Input (784) → Hidden (60 neurons) → Output (10)

WIDTH vs DEPTH - WHICH IS BETTER?
Width (more neurons per layer):
  + More capacity to learn diverse features in PARALLEL
  + Can represent more complex decision boundaries at each level
  + Less prone to vanishing gradient problems (shallower network)
  + Easier to train than very deep networks
  - Learns "flat" features, not hierarchical
  - Much more parameters → more memory, slower training

Depth (more layers):
  + Learns HIERARCHICAL features (low-level → high-level)
  + More parameter-efficient for complex patterns
  + Better inductive bias for structured data
  - Prone to vanishing gradients (especially with sigmoid)
  - More difficult optimization

Each neuron computes: activation = σ(Σ(w·x) + b)
where σ is the sigmoid function: σ(z) = 1/(1 + e^(-z))

TRAINING ALGORITHM:
Uses stochastic gradient descent (SGD) with backpropagation:
1. Forward pass: compute predictions
2. Compute error (cost function)
3. Backward pass: calculate gradients using backpropagation
4. Update weights and biases to reduce error

Expected accuracy: ~96-97% after 30 epochs (best of the three versions!)

Run: python network_1.2.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Training a WIDER Neural Network (v1.2)")
    print("=" * 60)
    print("Improvement over v1.0: Doubled hidden layer width")
    print("[784, 30, 10] → [784, 60, 10]")
    print("Comparison: v1.1 added DEPTH, v1.2 adds WIDTH")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    # MNIST Dataset:
    # - 70,000 grayscale images of handwritten digits (0-9)
    # - Each image is 28×28 pixels = 784 pixel values
    # - Pixel values range from 0.0 (white) to 1.0 (black)
    # 
    # Data Split:
    # - Training: 50,000 images - used to learn weights and biases
    # - Validation: 10,000 images - unused in this basic example
    # - Test: 10,000 images - used to evaluate final performance
    #
    # Data Format:
    # - Input: 784-dimensional vector (flattened 28×28 image)
    # - Output: 10-dimensional vector (one-hot encoded digit)
    #   Example: digit "3" → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(test_data)} test samples")
    
    # ========================================================================
    # STEP 2: Create the neural network
    # ========================================================================
    # Architecture: [784, 60, 10] - WIDER than network_1.0.py and network_1.1.py!
    #
    # Layer 1 (Input): 784 neurons
    #   - One neuron per pixel in the 28×28 image
    #   - No computation, just passes input values forward
    #
    # Layer 2 (Hidden): 60 neurons *** DOUBLED from 30 in v1.0! ***
    #   - Each neuron connected to ALL 784 input neurons
    #   - Total weights: 784 × 60 = 47,040 weights + 60 biases
    #   - Computation: a²ⱼ = σ(Σᵢ w²ⱼᵢ·a¹ᵢ + b²ⱼ)
    #   - With 60 neurons, can learn TWICE as many diverse features in parallel!
    #   - Each neuron can specialize in different edges, curves, patterns
    #   - More representational capacity at this single layer
    #
    # Layer 3 (Output): 10 neurons
    #   - One neuron per digit class (0-9)
    #   - Each neuron connected to all 60 hidden neurons (not 30!)
    #   - Total weights: 60 × 10 = 600 weights + 10 biases
    #   - Highest activation indicates predicted digit
    #   - Benefits from richer set of features from wider hidden layer
    #
    # Total Parameters: 47,040 + 60 + 600 + 10 = 47,710 learnable parameters!
    #
    # COMPARISON ACROSS VERSIONS:
    # - network_1.0.py [784, 30, 10]:     23,860 params (baseline)
    # - network_1.1.py [784, 30, 30, 10]: 24,790 params (+3.9%, added DEPTH)
    # - network_1.2.py [784, 60, 10]:     47,710 params (+100%, added WIDTH)
    #
    # Key insight: Doubling width doubles parameters dramatically!
    # But staying shallow avoids vanishing gradient problems.
    #
    # WIDTH ADVANTAGE:
    #   - More neurons = more parallel feature detectors
    #   - Single layer can learn very complex decision boundaries
    #   - Easier gradient flow (only 3 layers vs 4 in v1.1)
    #   - Often achieves better accuracy than shallow narrow networks
    #
    # WIDTH DISADVANTAGE:
    #   - 2× neurons = 2× parameters to train (memory + computation)
    #   - Learns "flat" features, not hierarchical representations
    #   - Less parameter-efficient than deep networks for complex patterns
    #
    # Initialization:
    # - Weights: Random values from Gaussian distribution N(0, 1)
    # - Biases: Random values from Gaussian distribution N(0, 1)
    # - Random initialization breaks symmetry so neurons learn different features
    print("\n2. Creating network...")
    print("   Architecture: [784, 60, 10] (WIDER than v1.0 and v1.1)")
    print("   - Input layer: 784 neurons (28×28 pixel image)")
    print("   - Hidden layer: 60 neurons (2× wider than v1.0!) *** NEW ***")
    print("   - Output layer: 10 neurons (digits 0-9)")
    print("   - Total parameters: 47,710 (vs 23,860 in v1.0, 24,790 in v1.1)")
    print("   Strategy: WIDTH over DEPTH")
    net = network.Network([784, 60, 10])
    
    # ========================================================================
    # STEP 3: Train using Stochastic Gradient Descent (SGD)
    # ========================================================================
    # TRAINING HYPERPARAMETERS:
    #
    # Epochs = 30
    #   - One epoch = one complete pass through all training data
    #   - Network sees each of the 50,000 training images 30 times
    #   - More epochs → more learning (but risk of overfitting)
    #
    # Mini-batch size = 10
    #   - Instead of updating weights after each image (slow, noisy)
    #     or after all images (slow convergence),
    #   - We update after every 10 images (good balance!)
    #   - Gradient is averaged over the mini-batch
    #   - 50,000 images ÷ 10 = 5,000 weight updates per epoch
    #
    # Learning rate (η) = 3.0
    #   - Controls how big each weight update step is
    #   - Update rule: w → w - η·∇w (move opposite to gradient)
    #   - Too large: network may overshoot and diverge
    #   - Too small: learning is very slow
    #   - 3.0 is relatively large, works well with sigmoid + quadratic cost
    #
    # Cost Function: Quadratic (Mean Squared Error)
    #   - C = (1/2n)·Σ||y(x) - a||²
    #   - Measures how far predictions are from correct answers
    #   - Goal: minimize this cost by adjusting weights and biases
    #
    # BACKPROPAGATION ALGORITHM (happens inside SGD):
    #   1. Forward pass: compute activations layer by layer
    #   2. Output error: δᴸ = (aᴸ - y) ⊙ σ'(zᴸ)
    #   3. Backward pass: propagate error backward through layers
    #      δˡ = ((wˡ⁺¹)ᵀ·δˡ⁺¹) ⊙ σ'(zˡ)
    #   4. Compute gradients: ∂C/∂w = a·δᵀ and ∂C/∂b = δ
    #   5. Update parameters: w → w - η·∂C/∂w, b → b - η·∂C/∂b
    #
    # MONITORING:
    #   - After each epoch, the network evaluates accuracy on test set
    #   - You'll see accuracy improve from ~10% (random) to ~95%!
    #   - Note: Using test set for monitoring isn't ideal (see second_network.py)
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 3.0")
    print("\n" + "-" * 60)
    
    # Run training!
    # Watch the accuracy improve epoch by epoch as the network learns
    # to recognize handwritten digits through gradient descent
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # WHAT JUST HAPPENED?
    # ========================================================================
    # The WIDER network just learned to recognize handwritten digits!
    # 
    # During training:
    # - Saw 50,000 images × 30 epochs = 1,500,000 examples
    # - Made 5,000 weight updates per epoch × 30 epochs = 150,000 updates
    # - Adjusted 47,710 parameters to minimize classification errors
    #   (that's 2× the parameters of v1.0!)
    #
    # What the WIDER network learned:
    # - 60 hidden neurons (vs 30 in v1.0) = 2× feature detection capacity
    # - Each neuron specializes in different features: edges, curves, loops, etc.
    # - With more neurons, can represent MORE DIVERSE patterns in PARALLEL
    # - Output layer has richer feature set to work with (60 vs 30 inputs)
    #
    # WIDTH vs DEPTH - THE EXPERIMENT:
    # 
    # network_1.0.py [784, 30, 10] - BASELINE:
    #   - 23,860 parameters, ~95% accuracy
    #   - Single hidden layer, moderate capacity
    #
    # network_1.1.py [784, 30, 30, 10] - DEPTH APPROACH:
    #   - 24,790 parameters (+3.9%), ~95-96% accuracy
    #   - Added second hidden layer for hierarchical features
    #   - Learns low-level → high-level feature progression
    #   - More parameter-efficient but harder to train (vanishing gradients)
    #
    # network_1.2.py [784, 60, 10] - WIDTH APPROACH:
    #   - 47,710 parameters (+100%), ~96-97% accuracy (often BEST!)
    #   - Doubled hidden layer width for more parallel features
    #   - Learns diverse features in single layer, no hierarchy
    #   - Easier to train (shallower) but uses more parameters
    #
    # KEY INSIGHTS:
    # 1. Width provides raw capacity - more neurons = more features
    # 2. Depth provides efficiency - hierarchical features with fewer params
    # 3. For MNIST with sigmoid, width often wins due to vanishing gradient issues
    # 4. Modern deep learning uses BOTH (wide AND deep) with better activations
    # 5. Trade-off: width needs more memory/compute, depth needs careful optimization
    #
    # WHICH SHOULD YOU USE?
    # - Shallow problems: add WIDTH (more neurons per layer)
    # - Complex hierarchical data: add DEPTH (more layers)
    # - Modern practice: Use BOTH + ReLU activation + batch normalization
    #
    # Limitations of this basic approach:
    # 1. Quadratic cost causes "learning slowdown" when neurons saturate
    # 2. No regularization → may overfit (especially with 2× parameters!)
    # 3. Large learning rate needed but can cause instability
    # 4. Using test data for monitoring isn't best practice
    # 5. Sigmoid activation limits how deep we can go
    #
    # See network2.py for improvements that address these issues!

if __name__ == "__main__":
    main()
