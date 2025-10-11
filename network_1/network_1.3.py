"""
Wide AND Deep Neural Network (Version 1.3)
===========================================

WHAT'S NEW IN VERSION 1.3:
This script demonstrates the COMBINED POWER of both WIDTH and DEPTH by using
TWO hidden layers with 60 neurons each - the best of both worlds!

ARCHITECTURE EVOLUTION:
- network_1.0.py: [784, 30, 10] - BASELINE (narrow + shallow)
- network_1.1.py: [784, 30, 30, 10] - DEPTH ONLY (narrow + deep)
- network_1.2.py: [784, 60, 10] - WIDTH ONLY (wide + shallow)
- network_1.3.py: [784, 60, 60, 10] - WIDTH + DEPTH (wide + deep) *** BEST! ***

ARCHITECTURE:
Input (784) → Hidden 1 (60) → Hidden 2 (60) → Output (10)

WHY COMBINE WIDTH AND DEPTH?
Width benefits:
  + 60 neurons per layer = MORE parallel feature detectors at each level
  + Can learn rich, diverse features simultaneously
  + Greater representational capacity

Depth benefits:
  + 2 hidden layers = HIERARCHICAL feature learning
  + Layer 1: learns 60 diverse low-level features (edges, curves, textures)
  + Layer 2: combines them into 60 complex high-level patterns (digit parts, shapes)
  + Most powerful architecture for complex pattern recognition

Trade-offs:
  - Maximum parameters (2× more than v1.2, 25× more than v1.0!)
  - Slower training (more computation per forward/backward pass)
  - Risk of overfitting without regularization
  - Vanishing gradient issues (sigmoid + depth)

Each neuron computes: activation = σ(Σ(w·x) + b)
where σ is the sigmoid function: σ(z) = 1/(1 + e^(-z))

TRAINING ALGORITHM:
Uses stochastic gradient descent (SGD) with backpropagation:
1. Forward pass: compute predictions through ALL layers
2. Compute error (cost function)
3. Backward pass: calculate gradients using backpropagation through deep, wide network
4. Update weights and biases to reduce error

Expected accuracy: ~97-98% after 30 epochs (BEST accuracy of all versions!)

Run: python network_1.3.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Training a WIDE + DEEP Neural Network (v1.3)")
    print("=" * 60)
    print("Combining WIDTH and DEPTH strategies")
    print("[784, 30, 10] → [784, 60, 60, 10]")
    print("v1.1 = DEPTH, v1.2 = WIDTH, v1.3 = BOTH!")
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
    # Architecture: [784, 60, 60, 10] - WIDE + DEEP! Most powerful yet!
    #
    # Layer 1 (Input): 784 neurons
    #   - One neuron per pixel in the 28×28 image
    #   - No computation, just passes input values forward
    #
    # Layer 2 (Hidden 1): 60 neurons *** 2× WIDER than v1.0 and v1.1! ***
    #   - Each neuron connected to ALL 784 input neurons
    #   - Total weights: 784 × 60 = 47,040 weights + 60 biases
    #   - Computation: a²ⱼ = σ(Σᵢ w²ⱼᵢ·a¹ᵢ + b²ⱼ)
    #   - With 60 neurons (vs 30), learns TWICE as many LOW-LEVEL features
    #   - Can detect diverse edges, curves, textures, gradients in parallel
    #   - WIDTH benefit: more representational capacity at first layer
    #
    # Layer 3 (Hidden 2): 60 neurons *** WIDE + DEEP! ***
    #   - Each neuron connected to all 60 neurons from Hidden 1 (not 30!)
    #   - Total weights: 60 × 60 = 3,600 weights + 60 biases
    #   - Computation: a³ⱼ = σ(Σᵢ w³ⱼᵢ·a²ᵢ + b³ⱼ)
    #   - With 60 neurons receiving 60 inputs, can learn complex HIGH-LEVEL features
    #   - Combines 60 low-level features → 60 rich high-level patterns
    #   - Can detect sophisticated patterns: loops, corners, digit parts, shapes
    #   - DEPTH benefit: hierarchical feature composition
    #   - WIDTH benefit: many parallel high-level feature detectors
    #
    # Layer 4 (Output): 10 neurons
    #   - One neuron per digit class (0-9)
    #   - Each neuron connected to all 60 Hidden 2 neurons
    #   - Total weights: 60 × 10 = 600 weights + 10 biases
    #   - Highest activation indicates predicted digit
    #   - Benefits from 60 rich hierarchical features (vs 30 or 60 flat features)
    #
    # Total Parameters: 47,040 + 60 + 3,600 + 60 + 600 + 10 = 51,370 parameters!
    #
    # PARAMETER COMPARISON:
    # - network_1.0.py [784, 30, 10]:     23,860 params (baseline)
    # - network_1.1.py [784, 30, 30, 10]: 24,790 params (+3.9% - depth, efficient)
    # - network_1.2.py [784, 60, 10]:     47,710 params (+100% - width, powerful)
    # - network_1.3.py [784, 60, 60, 10]: 51,370 params (+115% - BOTH, most powerful!)
    #
    # WHY THIS IS THE MOST POWERFUL ARCHITECTURE:
    #   ✓ 60 low-level feature detectors (width at layer 1)
    #   ✓ 60 high-level pattern combiners (width at layer 2)
    #   ✓ Hierarchical learning (depth with 2 hidden layers)
    #   ✓ Maximum representational capacity for MNIST
    #   ✓ Can learn both diverse AND hierarchical features
    #
    # TRADE-OFFS:
    #   - 51,370 parameters (2.15× baseline, 2.07× v1.1, 1.08× v1.2)
    #   - Slower training (more computation per forward/backward pass)
    #   - Higher risk of overfitting (many parameters, no regularization)
    #   - Vanishing gradients (sigmoid + depth still problematic)
    #   - More memory usage
    #
    # This demonstrates modern deep learning: networks are BOTH wide AND deep!
    # (Though modern networks use ReLU, BatchNorm, etc. to address limitations)
    #
    # Initialization:
    # - Weights: Random values from Gaussian distribution N(0, 1)
    # - Biases: Random values from Gaussian distribution N(0, 1)
    # - Random initialization breaks symmetry so neurons learn different features
    print("\n2. Creating network...")
    print("   Architecture: [784, 60, 60, 10] (WIDE + DEEP!)")
    print("   - Input layer: 784 neurons (28×28 pixel image)")
    print("   - Hidden layer 1: 60 neurons (wide, low-level features)")
    print("   - Hidden layer 2: 60 neurons (wide, high-level features)")
    print("   - Output layer: 10 neurons (digits 0-9)")
    print("   - Total parameters: 51,370 (most powerful architecture yet!)")
    print("   Strategy: Both WIDTH and DEPTH combined")
    net = network.Network([784, 60, 60, 10])
    
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
    # The WIDE + DEEP network just learned to recognize handwritten digits!
    # 
    # During training:
    # - Saw 50,000 images × 30 epochs = 1,500,000 examples
    # - Made 5,000 weight updates per epoch × 30 epochs = 150,000 updates
    # - Adjusted 51,370 parameters to minimize classification errors
    #   (that's 2.15× the baseline parameters!)
    #
    # What this WIDE + DEEP network learned:
    # - Hidden layer 1 (60 neurons): detects 60 DIVERSE low-level features
    #   → edges, curves, corners, textures, gradients in ALL orientations
    #   → WIDTH allows learning many features in parallel
    #
    # - Hidden layer 2 (60 neurons): combines into 60 COMPLEX high-level patterns
    #   → loops, junctions, strokes, digit parts, shapes
    #   → DEPTH enables hierarchical composition of features
    #   → WIDTH allows many high-level patterns simultaneously
    #
    # - Output layer: combines rich hierarchical features to classify digits
    #   → Benefits from 60 sophisticated features (vs 30 in v1.0/v1.1)
    #   → Features are both diverse (width) and hierarchical (depth)
    #
    # ========================================================================
    # COMPREHENSIVE COMPARISON - WIDTH vs DEPTH vs BOTH
    # ========================================================================
    #
    # network_1.0.py [784, 30, 10] - BASELINE:
    #   Parameters: 23,860 (baseline)
    #   Strategy: Narrow + Shallow (moderate capacity)
    #   Accuracy: ~95%
    #   Features: 30 flat features → direct classification
    #   Pros: Fast, simple, good baseline
    #   Cons: Limited capacity, no hierarchy
    #
    # network_1.1.py [784, 30, 30, 10] - DEPTH STRATEGY:
    #   Parameters: 24,790 (+3.9%)
    #   Strategy: Narrow + Deep (hierarchical learning)
    #   Accuracy: ~95-96%
    #   Features: 30 low → 30 high (hierarchical)
    #   Pros: Parameter-efficient, learns hierarchies
    #   Cons: Vanishing gradients, limited width per layer
    #
    # network_1.2.py [784, 60, 10] - WIDTH STRATEGY:
    #   Parameters: 47,710 (+100%)
    #   Strategy: Wide + Shallow (parallel learning)
    #   Accuracy: ~96-97%
    #   Features: 60 flat features → direct classification
    #   Pros: Rich features, easy to train, avoids vanishing gradients
    #   Cons: No hierarchy, many parameters, flat representation
    #
    # network_1.3.py [784, 60, 60, 10] - COMBINED STRATEGY:
    #   Parameters: 51,370 (+115%)
    #   Strategy: Wide + Deep (best of both worlds!)
    #   Accuracy: ~97-98% (BEST!)
    #   Features: 60 low → 60 high (hierarchical + diverse)
    #   Pros: Maximum capacity, hierarchical AND diverse features
    #   Cons: Most parameters, slower, overfitting risk, vanishing gradients
    #
    # ========================================================================
    # KEY INSIGHTS FROM THIS EXPERIMENT
    # ========================================================================
    #
    # 1. WIDTH provides CAPACITY:
    #    - More neurons = more parallel feature detectors
    #    - Can represent more diverse patterns simultaneously
    #    - Better raw performance but costs parameters
    #
    # 2. DEPTH provides HIERARCHY:
    #    - More layers = compositional feature learning
    #    - Low-level → high-level feature progression
    #    - More parameter-efficient but harder to optimize
    #
    # 3. WIDTH + DEPTH provides MAXIMUM POWER:
    #    - Combines benefits: diverse hierarchical features
    #    - Best accuracy but highest computational cost
    #    - This is why modern networks are BOTH wide AND deep!
    #
    # 4. TRADE-OFFS to consider:
    #    - Accuracy vs Parameters: v1.3 is best but uses 2× parameters of v1.0
    #    - Efficiency vs Power: v1.1 is efficient, v1.3 is powerful
    #    - For production: balance accuracy needs with computational budget
    #
    # 5. MODERN DEEP LEARNING uses BOTH:
    #    - ResNet, VGG, Transformers: all wide AND deep
    #    - But with better activations (ReLU), normalization, residual connections
    #    - These techniques address vanishing gradients and training issues
    #
    # ========================================================================
    # WHICH ARCHITECTURE SHOULD YOU USE?
    # ========================================================================
    #
    # - Limited compute/memory? → v1.0 or v1.1 (narrow architectures)
    # - Need efficiency? → v1.1 (depth is parameter-efficient)
    # - Need maximum accuracy? → v1.3 (wide + deep for best performance)
    # - Balanced approach? → v1.2 (wide but shallow, good accuracy/speed trade-off)
    #
    # For MNIST specifically:
    # - v1.0: Good starting point, fast experimentation
    # - v1.1: Learn about depth with minimal cost
    # - v1.2: Best accuracy-to-training-time ratio
    # - v1.3: Best absolute accuracy (if you don't mind slower training)
    #
    # Limitations of ALL these approaches:
    # 1. Quadratic cost causes "learning slowdown" when neurons saturate
    # 2. No regularization → overfitting risk (especially v1.2, v1.3)
    # 3. Large learning rate needed but can cause instability
    # 4. Using test data for monitoring isn't best practice
    # 5. Sigmoid + depth = vanishing gradient problems
    #
    # See network2.py and network3.py for advanced techniques that address these!

if __name__ == "__main__":
    main()
