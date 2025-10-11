"""
Your First Neural Network
==========================

This script introduces the FUNDAMENTALS of neural network training using
a simple 3-layer network on the MNIST handwritten digit dataset.

WHAT YOU'LL LEARN:
- How neural networks are structured (layers, neurons, weights, biases)
- How backpropagation learns from examples
- How stochastic gradient descent (SGD) optimizes the network
- What happens during training epochs

ARCHITECTURE:
Input Layer (784 neurons) → Hidden Layer (30 neurons) → Output Layer (10 neurons)

Each neuron computes: activation = σ(Σ(w·x) + b)
where σ is the sigmoid function: σ(z) = 1/(1 + e^(-z))

TRAINING ALGORITHM:
Uses stochastic gradient descent (SGD) with backpropagation:
1. Forward pass: compute predictions
2. Compute error (cost function)
3. Backward pass: calculate gradients using backpropagation
4. Update weights and biases to reduce error

Expected accuracy: ~95% after 30 epochs

WHAT'S NEXT:
After understanding this baseline, explore how different hyperparameters affect learning:
- network_1.0.1.py and network_1.0.2.py: Different learning rates
- network_1.0.3.py and network_1.0.4.py: Different batch sizes
- network_1.0.5.py and network_1.0.6.py: Different epoch counts
- network_1.1.0.py, network_1.2.0.py, network_1.3.0.py: Different architectures

Run: python network_1.0.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Training Your First Neural Network")
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
    # Architecture: [784, 30, 10]
    #
    # Layer 1 (Input): 784 neurons
    #   - One neuron per pixel in the 28×28 image
    #   - No computation, just passes input values forward
    #
    # Layer 2 (Hidden): 30 neurons
    #   - Each neuron connected to ALL 784 input neurons
    #   - Total weights: 784 × 30 = 23,520 weights + 30 biases
    #   - Computation: a²ⱼ = σ(Σᵢ w²ⱼᵢ·a¹ᵢ + b²ⱼ)
    #   - This layer learns to detect features (edges, curves, etc.)
    #
    # Layer 3 (Output): 10 neurons
    #   - One neuron per digit class (0-9)
    #   - Each neuron connected to all 30 hidden neurons
    #   - Total weights: 30 × 10 = 300 weights + 10 biases
    #   - Highest activation indicates predicted digit
    #
    # Total Parameters: 23,520 + 30 + 300 + 10 = 23,860 learnable parameters!
    #
    # Initialization:
    # - Weights: Random values from Gaussian distribution N(0, 1)
    # - Biases: Random values from Gaussian distribution N(0, 1)
    # - Random initialization breaks symmetry so neurons learn different features
    print("\n2. Creating network...")
    print("   Architecture: [784, 30, 10]")
    print("   - Input layer: 784 neurons (28×28 pixel image)")
    print("   - Hidden layer: 30 neurons")
    print("   - Output layer: 10 neurons (digits 0-9)")
    net = network.Network([784, 30, 10])
    
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
    # The network just learned to recognize handwritten digits!
    # 
    # During training:
    # - Saw 50,000 images × 30 epochs = 1,500,000 examples
    # - Made 5,000 weight updates per epoch × 30 epochs = 150,000 updates
    # - Adjusted 23,860 parameters to minimize classification errors
    #
    # What the network learned:
    # - Hidden layer neurons detect edges, curves, loops, and other features
    # - Output layer neurons combine these features to recognize digits
    # - The network can now generalize to NEW handwritten digits it's never seen!
    #
    # Limitations of this basic approach:
    # 1. Quadratic cost causes "learning slowdown" when neurons saturate
    # 2. No regularization → may overfit to training data
    # 3. Fixed learning rate doesn't adapt during training
    # 4. Using test data for monitoring isn't best practice
    #
    # Next steps in your learning journey:
    # - Try network_1.0.1.py to see how lower learning rates affect training
    # - See network_1.1.0.py+ for architecture experiments (depth vs width)
    # - For advanced techniques, see network2.py and network3.py

if __name__ == "__main__":
    main()
