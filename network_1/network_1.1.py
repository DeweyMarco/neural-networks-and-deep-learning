"""
Deeper Neural Network (Version 1.1)
====================================

WHAT'S NEW IN VERSION 1.1:
This script builds upon network_1.0.py by adding a SECOND HIDDEN LAYER,
demonstrating how deeper architectures can learn more complex representations.

KEY DIFFERENCE FROM 1.0:
- network_1.0.py: [784, 30, 10] - ONE hidden layer (3 layers total)
- network_1.1.py: [784, 30, 30, 10] - TWO hidden layers (4 layers total)

ARCHITECTURE:
Input (784) → Hidden 1 (30) → Hidden 2 (30) → Output (10)

WHY ADD ANOTHER LAYER?
- First hidden layer: learns low-level features (edges, curves)
- Second hidden layer: combines features into higher-level patterns
- Deeper networks can represent more complex functions
- Trade-off: more parameters to train, more computation, risk of vanishing gradients

Each neuron computes: activation = σ(Σ(w·x) + b)
where σ is the sigmoid function: σ(z) = 1/(1 + e^(-z))

TRAINING ALGORITHM:
Uses stochastic gradient descent (SGD) with backpropagation:
1. Forward pass: compute predictions through all layers
2. Compute error (cost function)
3. Backward pass: calculate gradients using backpropagation (now through more layers!)
4. Update weights and biases to reduce error

Expected accuracy: ~95-96% after 30 epochs (slightly better than 1.0)

Run: python network_1.1.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Training Your First DEEPER Neural Network (v1.1)")
    print("=" * 60)
    print("Improvement over v1.0: Added second hidden layer")
    print("[784, 30, 10] → [784, 30, 30, 10]")
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
    # Architecture: [784, 30, 30, 10] - DEEPER than network_1.0.py!
    #
    # Layer 1 (Input): 784 neurons
    #   - One neuron per pixel in the 28×28 image
    #   - No computation, just passes input values forward
    #
    # Layer 2 (Hidden 1): 30 neurons
    #   - Each neuron connected to ALL 784 input neurons
    #   - Total weights: 784 × 30 = 23,520 weights + 30 biases
    #   - Computation: a²ⱼ = σ(Σᵢ w²ⱼᵢ·a¹ᵢ + b²ⱼ)
    #   - Learns LOW-LEVEL features (edges, curves, simple shapes)
    #
    # Layer 3 (Hidden 2): 30 neurons *** NEW IN VERSION 1.1! ***
    #   - Each neuron connected to all 30 neurons from Hidden 1
    #   - Total weights: 30 × 30 = 900 weights + 30 biases
    #   - Computation: a³ⱼ = σ(Σᵢ w³ⱼᵢ·a²ᵢ + b³ⱼ)
    #   - Learns HIGH-LEVEL features by combining low-level features
    #   - Can detect more complex patterns (loops, junctions, digit parts)
    #
    # Layer 4 (Output): 10 neurons
    #   - One neuron per digit class (0-9)
    #   - Each neuron connected to all 30 Hidden 2 neurons
    #   - Total weights: 30 × 10 = 300 weights + 10 biases
    #   - Highest activation indicates predicted digit
    #
    # Total Parameters: 23,520 + 30 + 900 + 30 + 300 + 10 = 24,790 parameters
    # Comparison to network_1.0.py: 24,790 vs 23,860 = +930 parameters (+3.9%)
    #
    # Benefits of extra layer:
    #   + Can learn hierarchical features (low-level → high-level)
    #   + Better representational power for complex patterns
    #   + Often achieves slightly better accuracy
    #
    # Drawbacks of extra layer:
    #   - More parameters to train (slower training)
    #   - Deeper gradient path (risk of vanishing gradients with sigmoid)
    #   - Needs more careful hyperparameter tuning
    #
    # Initialization:
    # - Weights: Random values from Gaussian distribution N(0, 1)
    # - Biases: Random values from Gaussian distribution N(0, 1)
    # - Random initialization breaks symmetry so neurons learn different features
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 30, 10] (DEEPER than version 1.0)")
    print("   - Input layer: 784 neurons (28×28 pixel image)")
    print("   - Hidden layer 1: 30 neurons (low-level features)")
    print("   - Hidden layer 2: 30 neurons (high-level features) *** NEW ***")
    print("   - Output layer: 10 neurons (digits 0-9)")
    print("   - Total parameters: 24,790 (vs 23,860 in version 1.0)")
    net = network.Network([784, 30, 30, 10])
    
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
    # The network just learned to recognize handwritten digits using a DEEPER architecture!
    # 
    # During training:
    # - Saw 50,000 images × 30 epochs = 1,500,000 examples
    # - Made 5,000 weight updates per epoch × 30 epochs = 150,000 updates
    # - Adjusted 24,790 parameters to minimize classification errors
    #
    # What the DEEPER network learned (hierarchical representation):
    # - Hidden layer 1: detects low-level features (edges, curves, corners)
    # - Hidden layer 2: combines into high-level patterns (loops, digit parts)
    # - Output layer: combines high-level patterns to recognize complete digits
    # - The extra layer enables more complex feature hierarchies!
    #
    # Comparison to network_1.0.py (shallow network):
    # - network_1.0.py [784, 30, 10]: One-step feature extraction → classification
    # - network_1.1.py [784, 30, 30, 10]: Two-step hierarchical feature learning
    # - Deeper network often achieves 0.5-1% better accuracy
    # - Trade-off: +930 parameters, slightly slower training per epoch
    #
    # Key insight about depth:
    # - Depth allows hierarchical feature learning (similar to visual cortex)
    # - However, with sigmoid activation, very deep networks struggle (vanishing gradients)
    # - Modern deep networks use ReLU and other techniques (see network2.py, network3.py)
    #
    # Limitations of this approach:
    # 1. Quadratic cost causes "learning slowdown" when neurons saturate
    # 2. No regularization → may overfit to training data
    # 3. Large learning rate needed but can cause instability
    # 4. Using test data for monitoring isn't best practice
    # 5. Sigmoid + depth = vanishing gradient problems
    #
    # See network2.py for improvements that address these issues!

if __name__ == "__main__":
    main()
