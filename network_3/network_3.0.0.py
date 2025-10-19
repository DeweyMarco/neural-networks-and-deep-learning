"""
Your First Convolutional Neural Network
=========================================

PREREQUISITES:
- Complete Chapters 1 and 2 (network_1/ and network_2/)
- Understand: backpropagation, SGD, cross-entropy, regularization

THIS EXPERIMENT:
Introduces CONVOLUTIONAL NEURAL NETWORKS (CNNs) - the revolutionary architecture
that dominates computer vision and many other domains.

THE THREE KEY INNOVATIONS:

1. CONVOLUTIONAL LAYERS
   - Instead of fully connected: use small filters that slide across the image
   - Exploit spatial structure: nearby pixels are related
   - Parameter sharing: same filter used across entire image
   - Translation invariance: recognize patterns anywhere

2. MAX POOLING
   - Reduces spatial dimensions (downsampling)
   - Provides translation invariance (small shifts don't matter)
   - Reduces parameters for deeper layers
   - Forces learning of robust features

3. SOFTMAX OUTPUT
   - Outputs proper probability distribution over classes
   - Better than sigmoid for multi-class classification
   - Works perfectly with cross-entropy loss

ARCHITECTURE:
Input (28×28 image) 
  → Convolutional Layer (20 filters, 5×5)
  → ReLU activation
  → Max Pooling (2×2)
  → Fully Connected Layer (100 neurons)
  → Dropout (p=0.5)
  → Softmax Output (10 classes)

WHY THIS WORKS BETTER THAN FULLY CONNECTED:

Fully Connected [784, 30, 10]:
  • Flattens image → loses spatial structure
  • Each neuron sees entire image
  • Parameters: 784×30 = 23,520 weights
  • Accuracy: ~95-97%
  
Convolutional (this network):
  • Preserves 2D structure
  • Each filter sees local patches
  • Parameters: 5×5×20 = 500 for conv layer (47× fewer!)
  • Accuracy: ~98.5% (+1.5-3% improvement!)

THE POWER OF CONVOLUTION:

Consider detecting a "horizontal edge":
  Fully connected: Must learn 784 different weights (one per pixel)
  Convolutional: Learn ONE 5×5 filter, apply everywhere
  
Result: 
  • 150× fewer parameters (5×5 vs 28×28)
  • Works anywhere in image (translation invariance)
  • More robust to variations

Expected Results:
- Validation accuracy: ~98.5% (vs ~96-97% for fully connected)
- Parameters: ~50K (vs ~24K for [784,30,10] but MUCH better!)
- Training time: Slower than FC (but worth it!)
- Key lesson: Specialized architectures beat general ones

COMPARISON TO CHAPTER 2:
Chapter 2 best [784, 100, 100, 10]: 88K params → ~98%
This CNN: ~50K params → ~98.5% (BETTER with fewer parameters!)

WHAT'S NEXT:
- network_3.0.1.py: Fully connected baseline for direct comparison
- network_3.0.2.py: Deep CNN (multiple conv layers)
- network_3.1.x: Activation function improvements (ReLU focus)
- network_3.2.x: Dropout experiments
- network_3.3.x: Optimized architectures (99%+ accuracy)

Run: python network_3.0.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

def main():
    # ========================================================================
    # EXPERIMENT: First Convolutional Neural Network
    # ========================================================================
    print("=" * 75)
    print("YOUR FIRST CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 75)

    # ========================================================================
    # STEP 1: Load MNIST data (Theano shared variables)
    # ========================================================================
    # network3.py uses Theano, which loads data differently than network2.py
    # Data is placed in GPU memory (if available) for fast computation
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = network3.load_data_shared()
    print("   ✓ Loaded 50,000 training samples")
    print("   ✓ Loaded 10,000 validation samples") 
    print("   ✓ Loaded 10,000 test samples")

    # ========================================================================
    # STEP 2: Create the CNN architecture
    # ========================================================================
    # ARCHITECTURE OVERVIEW:
    #
    # LAYER 1: Convolutional Layer
    # ----------------------------
    # Input:  28×28 grayscale image (1 channel)
    # Filters: 20 filters of size 5×5
    # Operation: Slide each filter across image, compute dot product
    # Output: 20 feature maps, each 24×24 (28-5+1 = 24, due to no padding)
    # Activation: ReLU (max(0, x))
    #
    # What it learns:
    #   • 20 different local patterns (edges, curves, textures)
    #   • Filter 1 might detect horizontal edges
    #   • Filter 2 might detect vertical edges
    #   • Filter 3 might detect diagonal lines, etc.
    #
    # Parameters: 5×5×20 + 20 = 520 (weights + biases)
    #   vs Fully Connected [784,30]: 23,520 parameters!
    #   47× fewer parameters for better feature detection!
    #
    # LAYER 2: Max Pooling (2×2)
    # ---------------------------
    # Input:  20 feature maps, each 24×24
    # Operation: Take maximum value in each 2×2 window
    # Output: 20 feature maps, each 12×12 (24/2 = 12)
    #
    # What it does:
    #   • Reduces spatial dimensions (24×24 → 12×12)
    #   • Provides translation invariance (feature at position 5 or 6 → same)
    #   • Reduces computation for next layer
    #   • Forces learning of robust features
    #
    # Parameters: 0 (max pooling has no learnable parameters)
    #
    # LAYER 3: Fully Connected Layer (100 neurons)
    # ---------------------------------------------
    # Input:  20×12×12 = 2,880 neurons (flattened feature maps)
    # Output: 100 neurons
    # Activation: ReLU
    # Dropout: p=0.5 (randomly drop 50% of neurons during training)
    #
    # What it learns:
    #   • High-level combinations of features
    #   • "Does this image have a vertical line (filter 1) AND
    #      a curve at the top (filter 7)?" → Probably a '9'!
    #
    # Parameters: 2,880×100 + 100 = 288,100
    #
    # Dropout regularization:
    #   • Prevents overfitting with so many parameters
    #   • Forces neurons to work independently
    #   • Creates ensemble effect
    #
    # LAYER 4: Softmax Output (10 neurons)
    # -------------------------------------
    # Input:  100 neurons
    # Output: 10 neurons (one per digit 0-9)
    # Activation: Softmax (outputs sum to 1.0)
    #
    # What it does:
    #   • Converts neuron activations to probabilities
    #   • Output: [0.01, 0.02, 0.03, 0.89, 0.01, ...] ← 89% confident it's a '3'
    #   • Better than sigmoid for classification
    #
    # Parameters: 100×10 + 10 = 1,010
    #
    # TOTAL PARAMETERS:
    #   Conv Layer:     520
    #   Pooling Layer:  0
    #   FC Layer:       288,100
    #   Output Layer:   1,010
    #   --------------------------------
    #   TOTAL:          ~289,630 parameters
    #
    # Note: Most parameters are in the FC layer!
    # Later networks will use more conv layers to reduce this.
    
    print("\n2. Creating network layers...")
    
    # MINI-BATCH SIZE:
    # network3.py requires specifying mini_batch_size at network creation
    # We use 10 to match Chapter 2 experiments for fair comparison
    mini_batch_size = 10
    
    # Layer 1: Convolutional + Pooling
    # ConvPoolLayer(filter_shape, image_shape, poolsize, activation_fn)
    #   filter_shape: (num_filters, input_channels, filter_height, filter_width)
    #   image_shape: (mini_batch_size, input_channels, image_height, image_width)
    #   poolsize: (pool_height, pool_width)
    #   activation_fn: ReLU (modern choice, better than sigmoid)
    layer1 = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),     # 20 filters, 1 input channel, 5×5 filter
        image_shape=(mini_batch_size, 1, 28, 28),  # batch of 28×28 grayscale images
        poolsize=(2, 2),                 # 2×2 max pooling
        activation_fn=ReLU
    )
    
    # Layer 2: Fully Connected
    # After conv (28×28 → 24×24) and pooling (24×24 → 12×12):
    # 20 feature maps × 12×12 = 2,880 inputs
    # Output: 100 neurons
    # Dropout: p=0.5 (strong regularization)
    layer2 = FullyConnectedLayer(
        n_in=20*12*12,                   # 2,880 inputs from pooled feature maps
        n_out=100,                       # 100 neurons in hidden layer
        activation_fn=ReLU,              # ReLU activation
        p_dropout=0.5                    # 50% dropout during training
    )
    
    # Layer 3: Softmax Output
    # Input: 100 neurons from previous layer
    # Output: 10 neurons (one per digit)
    # Softmax: Converts to probability distribution
    layer3 = SoftmaxLayer(
        n_in=100,                        # 100 inputs from FC layer
        n_out=10,                        # 10 outputs (digits 0-9)
        p_dropout=0.0                    # No dropout on output layer
    )
    
    # Create the complete network
    net = Network([layer1, layer2, layer3], mini_batch_size)
    
    print("   ✓ Conv layer (20 filters, 5×5) + Max pooling (2×2)")
    print("   ✓ Fully connected (100 neurons, ReLU, dropout 0.5)")
    print("   ✓ Softmax output (10 classes)")

    # ========================================================================
    # STEP 3: Train the network
    # ========================================================================
    # TRAINING PARAMETERS:
    # - epochs: 60 (more than Chapter 2 because CNNs take longer to converge)
    # - eta: 0.03 (learning rate, smaller than Chapter 2's 0.5)
    # - lmbda: 0.1 (L2 regularization, lighter than Chapter 2's 5.0)
    #   (Dropout provides strong regularization, so less L2 needed)
    #
    # WHY CNNS TAKE LONGER TO TRAIN:
    #   • More complex operations (convolution vs matrix multiply)
    #   • More epochs needed for feature detectors to develop
    #   • But result is MUCH better accuracy!
    #
    # WHAT TO WATCH DURING TRAINING:
    #   • Epoch 0-10: Learning basic edges and curves
    #   • Epoch 10-30: Combining features into shapes
    #   • Epoch 30-60: Fine-tuning and achieving high accuracy
    #   • Expect: 96% → 98% → 98.5%+ progression
    
    print("\n3. Training network...")
    print("   Epochs: 60")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.03")
    print("   L2 regularization: 0.1")
    print("   Dropout: 0.5")
    print()
    print("Training in progress...")
    print("-" * 75)
    
    # Train the network
    # SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda)
    net.SGD(
        training_data, 
        60,                    # epochs
        mini_batch_size,       # mini_batch_size (10)
        0.03,                  # eta (learning rate)
        validation_data, 
        test_data,
        lmbda=0.1              # L2 regularization
    )

    # ========================================================================
    # STEP 4: Summary and key takeaways
    # ========================================================================
    # WHY CNNS WIN FOR IMAGES:
    #
    # KEY INSIGHT #1: Exploiting Spatial Structure
    # ----------------------------------------------
    # Fully Connected Networks:
    #   • Flatten image: [28, 28] → [784]
    #   • Lose information: Pixel 0 and pixel 1 are adjacent, but network doesn't know
    #   • Must learn: "Pixels that are near each other are related"
    #   • Inefficient: Relearn same concept for every image position
    #
    # Convolutional Networks:
    #   • Keep 2D structure: [28, 28] stays as [28, 28]
    #   • Convolution naturally exploits locality
    #   • 5×5 filter sees 25 adjacent pixels (natural receptive field)
    #   • Same filter works everywhere (translation invariance)
    #   • Result: Better features with fewer parameters!
    #
    # KEY INSIGHT #2: Translation Invariance
    # ---------------------------------------
    # Problem: A '7' is a '7' whether it's in top-left, center, or shifted
    #
    # Fully Connected Solution:
    #   • Must learn separate detectors for each position
    #   • 'Vertical line at position 5' ≠ 'Vertical line at position 6'
    #   • Inefficient and doesn't generalize well
    #
    # Convolutional Solution:
    #   • Same filter slides across entire image
    #   • 'Vertical line detector' finds vertical lines EVERYWHERE
    #   • Max pooling adds robustness to small shifts
    #   • Result: Naturally translation-invariant!
    #
    # KEY INSIGHT #3: Hierarchical Feature Learning
    # ----------------------------------------------
    # Layer 1 (Convolutional): Low-level features
    #   • Edge detectors (horizontal, vertical, diagonal)
    #   • Curve detectors
    #   • Texture patterns
    #
    # Layer 2 (Fully Connected): High-level features
    #   • Combinations of low-level features
    #   • 'Vertical line + curve at top' → top of '9'
    #   • 'Two horizontal lines + vertical line' → '7'
    #
    # Output (Softmax): Classification
    #   • Combines high-level features
    #   • Maps to digit probabilities
    #
    # Result: Natural hierarchy from simple → complex
    #
    # KEY INSIGHT #4: Parameter Efficiency
    # -------------------------------------
    # To detect 20 different features across a 28×28 image:
    #
    # Fully Connected:
    #   • 20 neurons × 784 weights each = 15,680 weights
    #   • Must relearn feature for each position
    #
    # Convolutional:
    #   • 20 filters × 5×5 weights each = 500 weights
    #   • Same filter used at all positions
    #
    # Ratio: 500 / 15,680 = 3.2%
    # → 31× FEWER PARAMETERS for BETTER feature detection!
    #
    # COMPARISON TO CHAPTER 2:
    # -------------------------
    # Best Chapter 2 [784, 100, 100, 10]:
    #   • Architecture: Fully connected
    #   • Parameters: 88,610
    #   • Techniques: Cross-entropy + L2 regularization
    #   • Accuracy: ~98.0%
    #   • Limitation: Can't exploit spatial structure
    #
    # This CNN:
    #   • Architecture: Conv + Pooling + FC
    #   • Parameters: ~289,630 (mostly in FC layer)
    #   • Techniques: Cross-entropy + L2 + Dropout + ReLU + CNNs
    #   • Accuracy: ~98.5%+ (+0.5% improvement)
    #   • Advantage: Exploits spatial structure of images
    #
    # THE PATH TO 99%+:
    # -----------------
    # This achieves ~98.5% with a simple CNN. To reach 99%+:
    #
    # 1. Deeper CNNs (network_3.0.2.py)
    #    • Multiple convolutional layers
    #    • Hierarchical feature learning
    #    • Expected: ~99.0%
    #
    # 2. Architecture optimization (network_3.3.x)
    #    • Optimal filter sizes and layer depths
    #    • Better regularization strategies
    #    • Expected: ~99.3%
    #
    # 3. Ensemble methods (network_3.3.3.py)
    #    • Train multiple networks
    #    • Average predictions
    #    • Expected: ~99.5%+
    #
    # WHAT YOU'VE LEARNED:
    # --------------------
    # ✓ Why CNNs dominate computer vision
    # ✓ How convolution exploits spatial structure
    # ✓ How max pooling provides translation invariance
    # ✓ Why specialized architectures beat general ones
    # ✓ How to design and train a CNN
    # ✓ ReLU + Dropout + CNNs = modern deep learning
    
    print("\n" + "=" * 75)
    print("Training complete!")
    print("=" * 75)
    print("\n✓ Expected accuracy: ~98.5% (vs ~96-97% for fully connected)")
    print("✓ CNNs exploit spatial structure for better image understanding")
    print("✓ Next: network_3.0.2.py for deeper CNNs achieving ~99%+")
    print()
    print("Congratulations on training your first CNN!")

if __name__ == "__main__":
    main()

