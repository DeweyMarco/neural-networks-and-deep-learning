"""
Deep Convolutional Neural Network - Multiple Conv Layers
=========================================================

PREREQUISITES:
- Complete network_3.0.0.py (first CNN with 1 conv layer)
- Complete network_3.0.1.py (fully connected baseline)
- Understand: why CNNs exploit spatial structure

THIS EXPERIMENT:
Introduces DEEP CONVOLUTIONAL NETWORKS with multiple conv layers.
Demonstrates HIERARCHICAL FEATURE LEARNING - the key to achieving 99%+ accuracy.

THE EVOLUTIONARY LEAP:

network_3.0.1.py (Fully Connected):
  • No spatial structure
  • ~97% accuracy
  • Lesson: Architecture matters

network_3.0.0.py (Shallow CNN):
  • 1 conv layer learns simple features
  • ~98.5% accuracy
  • Lesson: Convolutions help

network_3.0.2.py (Deep CNN - THIS FILE):
  • 2 conv layers learn hierarchical features
  • ~99% accuracy (+0.5% improvement!)
  • Lesson: DEPTH in convolution space = better features

HIERARCHICAL FEATURE LEARNING:

Layer 1 (First Conv): LOW-LEVEL FEATURES
  • Edges (horizontal, vertical, diagonal)
  • Curves and corners
  • Simple textures
  • Example: "There's a vertical edge here"

Layer 2 (Second Conv): MID-LEVEL FEATURES
  • COMBINATIONS of low-level features
  • Shapes (loops, stems, curves)
  • Parts of digits
  • Example: "Vertical edge + curve = top of '9'"

Layer 3 (Fully Connected): HIGH-LEVEL FEATURES
  • COMBINATIONS of mid-level features
  • Complete digit patterns
  • Example: "Loop + stem + base = complete '9'"

Output (Softmax): CLASSIFICATION
  • Maps high-level features to digit probabilities

This hierarchy mirrors how the human visual cortex works!

ARCHITECTURE COMPARISON:

Shallow CNN (network_3.0.0.py):
  Input (28×28) 
    → Conv1 (20 filters, 5×5) + Pool → 12×12×20
    → FC (100 neurons)
    → Softmax (10)
  Problem: Conv layer must learn ALL features (simple AND complex)
  Result: ~98.5%

Deep CNN (THIS FILE):
  Input (28×28)
    → Conv1 (20 filters, 5×5) + Pool → 12×12×20
    → Conv2 (40 filters, 5×5) + Pool → 4×4×40
    → FC (100 neurons)
    → Softmax (10)
  Advantage: Conv1 learns simple features, Conv2 combines them
  Result: ~99.0% (+0.5% improvement!)

WHY DEPTH IN CONVOLUTION SPACE WINS:

1. FEATURE HIERARCHY
   Single Conv Layer:
     • Must learn both "edges" and "edge combinations" in one layer
     • No natural progression from simple to complex
     • Features compete for filter slots
   
   Multiple Conv Layers:
     • Layer 1: Focus on edges (optimized for this task)
     • Layer 2: Combine edges into shapes (builds on layer 1)
     • Natural progression matches vision science
     • Each layer specialized for its level

2. RECEPTIVE FIELD GROWTH
   Single Conv Layer:
     • 5×5 filter sees 25 pixels (5×5 patch)
     • Limited context for understanding
   
   Multiple Conv Layers:
     • Layer 1: 5×5 = sees 5×5 patch (25 pixels)
     • Layer 2: 5×5 on pooled layer 1
       After pooling, each position represents 2×2 original pixels
       So 5×5 filter on layer 2 sees: 5×2 × 5×2 = 10×10 patch (100 pixels!)
     • Effective receptive field: 4× larger!
     • More context = better understanding

3. COMPOSITIONAL EFFICIENCY
   Shallow Network (1 conv layer with 60 filters to match parameters):
     • 60 filters must each learn complete patterns
     • "Loop at top-left" = separate filter
     • "Loop at top-right" = separate filter
     • "Stem on left" = separate filter
     • "Stem on right" = separate filter
     • Exponential growth in needed filters
   
   Deep Network (20 + 40 filters):
     • Layer 1: 20 filters learn basic elements (edges, curves)
     • Layer 2: 40 filters combine elements
     • 20 × 40 = 800 possible combinations!
     • Exponential expressiveness with linear parameters
     • This is why deep networks are so powerful!

Expected Results:
- Validation accuracy: ~99.0% (vs ~98.5% for shallow CNN)
- Parameters: ~150K (more than shallow CNN due to second conv layer)
- Training time: Longer (more computation)
- Key lesson: Depth enables hierarchical feature learning

THE POWER OF DEPTH:
This 0.5% improvement from 98.5% → 99% is HUGE:
  • 98.5% = 150 errors per 10,000 images
  • 99.0% = 100 errors per 10,000 images
  • 50 fewer errors = 33% error reduction!

WHAT'S NEXT:
- network_3.1.x: ReLU vs sigmoid activation experiments
- network_3.2.x: Dropout regularization experiments
- network_3.3.x: Optimized architectures for 99.5%+

Run: python network_3.0.2.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

def main():
    # ========================================================================
    # EXPERIMENT: Deep CNN with Multiple Convolutional Layers
    # ========================================================================
    print("=" * 75)
    print("DEEP CONVOLUTIONAL NEURAL NETWORK - HIERARCHICAL FEATURES")
    print("=" * 75)

    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = network3.load_data_shared()
    print("   ✓ Loaded 50,000 training samples")
    print("   ✓ Loaded 10,000 validation samples") 
    print("   ✓ Loaded 10,000 test samples")

    # ========================================================================
    # STEP 2: Create the DEEP CNN architecture
    # ========================================================================
    # ARCHITECTURE OVERVIEW:
    #
    # LAYER 1: First Convolutional Layer + Pooling
    # ---------------------------------------------
    # Input:  28×28 grayscale image (1 channel)
    # Filters: 20 filters of size 5×5
    # Operation: Convolution (no padding)
    # Output after conv: 20 feature maps, each 24×24 (28-5+1 = 24)
    # Pooling: 2×2 max pooling
    # Output after pool: 20 feature maps, each 12×12 (24/2 = 12)
    # Activation: ReLU
    #
    # What it learns:
    #   • LOW-LEVEL FEATURES (edges, curves, corners)
    #   • Horizontal edges: [ 1  1  1  1  1]
    #   •                   [ 0  0  0  0  0]
    #   •                   [-1 -1 -1 -1 -1]
    #   • Vertical edges, diagonal edges, curves, etc.
    #   • These are the building blocks for higher-level features
    #
    # Parameters: 5×5×1×20 + 20 = 520 (weights + biases)
    #
    # Receptive field: 5×5 = 25 pixels
    #
    # LAYER 2: Second Convolutional Layer + Pooling
    # ----------------------------------------------
    # Input:  20 feature maps, each 12×12 (output of layer 1 pooling)
    # Filters: 40 filters of size 5×5
    #   • Each filter looks at ALL 20 input feature maps
    #   • Each filter creates one output feature map
    # Output after conv: 40 feature maps, each 8×8 (12-5+1 = 8)
    # Pooling: 2×2 max pooling
    # Output after pool: 40 feature maps, each 4×4 (8/2 = 4)
    # Activation: ReLU
    #
    # What it learns:
    #   • MID-LEVEL FEATURES (combinations of edges/curves)
    #   • "Horizontal edge + vertical edge" → Corner
    #   • "Curve + curve" → Loop (like in 0, 6, 8, 9)
    #   • "Vertical line + horizontal top" → Top of 7, T-shape
    #   • Parts of digits rather than individual edges
    #
    # Parameters: 5×5×20×40 + 40 = 20,040 (weights + biases)
    #   • Each of 40 filters has 5×5×20 = 500 weights
    #   • Combines information from ALL 20 input feature maps
    #
    # Effective receptive field:
    #   • Layer 2 sees 5×5 region in layer 1 feature maps
    #   • Each position in layer 1 corresponds to 2×2 region in input (due to pooling)
    #   • Each position in layer 2 sees: (5-1)×2 + 5 = 13×13 input patch
    #   • Much larger context than single layer!
    #
    # KEY INSIGHT: Hierarchical Composition
    #   • Layer 1 has 20 different edge/curve detectors
    #   • Layer 2 has 40 filters, each combining these 20 features
    #   • Possible combinations: 2^20 ≈ 1 million patterns!
    #   • But we only need 20 + 40 = 60 filters (not 1 million)
    #   • This compositional efficiency is why deep networks work!
    #
    # LAYER 3: Fully Connected Layer
    # --------------------------------
    # Input:  40×4×4 = 640 neurons (flattened feature maps)
    # Output: 100 neurons
    # Activation: ReLU
    # Dropout: p=0.5
    #
    # What it learns:
    #   • HIGH-LEVEL FEATURES (complete digit patterns)
    #   • "Does this have loop at top + stem below?" → Probably '9'
    #   • "Two loops stacked?" → Probably '8'
    #   • Combines spatial information from all 40 feature maps
    #
    # Parameters: 640×100 + 100 = 64,100
    #
    # LAYER 4: Softmax Output
    # ------------------------
    # Input:  100 neurons
    # Output: 10 neurons (one per digit 0-9)
    # Activation: Softmax
    #
    # Parameters: 100×10 + 10 = 1,010
    #
    # TOTAL PARAMETERS:
    #   Conv Layer 1:   520
    #   Conv Layer 2:   20,040
    #   FC Layer:       64,100
    #   Output Layer:   1,010
    #   --------------------------------
    #   TOTAL:          ~85,670 parameters
    #
    # COMPARISON TO NETWORK_3.0.0.PY:
    #   Shallow CNN:    ~289,630 parameters
    #   Deep CNN:       ~85,670 parameters (3.4× FEWER!)
    #   
    #   Why fewer?
    #   • Shallow CNN: Conv output is 12×12×20 = 2,880 → 100 FC
    #     FC params: 2,880×100 = 288,000
    #   • Deep CNN: Second conv reduces to 4×4×40 = 640 → 100 FC
    #     FC params: 640×100 = 64,000
    #   • More convolution, less fully connected = fewer parameters!
    #   • This is a key principle: push processing into conv layers
    
    print("\n2. Creating network layers...")
    
    mini_batch_size = 10
    
    # Layer 1: First Convolutional + Pooling
    # Input: 28×28 image
    # Output: 20 feature maps of size 12×12
    layer1 = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),              # 20 filters, 1 input channel, 5×5
        image_shape=(mini_batch_size, 1, 28, 28), # Batch of 28×28 images
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    
    # Layer 2: Second Convolutional + Pooling
    # Input: 20 feature maps of size 12×12 (from layer 1)
    # Output: 40 feature maps of size 4×4
    # NOTE: image_shape here describes the OUTPUT of layer 1
    layer2 = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),             # 40 filters, 20 input channels, 5×5
        image_shape=(mini_batch_size, 20, 12, 12), # Output from layer 1 (after pooling)
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    
    # Layer 3: Fully Connected
    # Input: 40×4×4 = 640 (flattened feature maps from layer 2)
    # Output: 100 neurons
    layer3 = FullyConnectedLayer(
        n_in=40*4*4,                             # 640 inputs from pooled feature maps
        n_out=100,                               # 100 neurons
        activation_fn=ReLU,                      # ReLU activation
        p_dropout=0.5                            # 50% dropout
    )
    
    # Layer 4: Softmax Output
    layer4 = SoftmaxLayer(
        n_in=100,                                # 100 inputs from FC layer
        n_out=10,                                # 10 outputs (digits 0-9)
        p_dropout=0.0                            # No dropout on output
    )
    
    # Create the complete network
    net = Network([layer1, layer2, layer3, layer4], mini_batch_size)
    
    print("   ✓ Conv layer 1 (20 filters, 5×5) + Max pooling (2×2)")
    print("   ✓ Conv layer 2 (40 filters, 5×5) + Max pooling (2×2)")
    print("   ✓ Fully connected (100 neurons, ReLU, dropout 0.5)")
    print("   ✓ Softmax output (10 classes)")
    print("   ✓ Total parameters: ~85,670")

    # ========================================================================
    # STEP 3: Train the network
    # ========================================================================
    # TRAINING PARAMETERS:
    # Same as network_3.0.0.py for fair comparison:
    # - epochs: 60
    # - eta: 0.03 (learning rate)
    # - lmbda: 0.1 (L2 regularization)
    #
    # WHAT TO EXPECT DURING TRAINING:
    #   • Epoch 0-15: Layer 1 learns edges, layer 2 starts combining
    #   • Epoch 15-35: Hierarchical features emerge, accuracy climbs
    #   • Epoch 35-60: Fine-tuning, reaching ~99%
    #   • Progression: 97% → 98.5% → 99.0%+
    
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
    # STEP 4: Analysis and key takeaways
    # ========================================================================
    # WHY DEEP CNNS ACHIEVE 99%:
    #
    # KEY INSIGHT #1: Hierarchical Feature Learning
    # ----------------------------------------------
    # Human Vision Works Hierarchically:
    #   Retina → Simple cells (edges) → Complex cells (shapes) → Objects
    #   This is how V1, V2, V4, IT cortex work in neuroscience!
    #
    # Shallow CNN (1 conv layer):
    #   • Must learn "edges" and "shapes" in same layer
    #   • Like asking V1 (primary visual cortex) to do object recognition
    #   • Can work but not optimal
    #
    # Deep CNN (2 conv layers):
    #   • Layer 1: Edges (like V1 simple cells)
    #   • Layer 2: Shapes (like V1 complex cells / V2)
    #   • FC layer: Objects (like IT cortex)
    #   • Matches biological vision hierarchy
    #   • More natural, more effective
    #
    # KEY INSIGHT #2: Receptive Field Growth
    # ---------------------------------------
    # Shallow CNN:
    #   • Conv layer sees 5×5 patch = 25 pixels
    #   • Limited context for digit understanding
    #   • A '1' might be 20+ pixels tall, can't see full digit
    #
    # Deep CNN:
    #   • Layer 1 filter: sees 5×5 = 25 pixels (local edges)
    #   • Layer 2 filter: sees 13×13 ≈ 169 pixels (nearly half the image!)
    #   • Can see much larger patterns
    #   • Can distinguish "narrow 1" from "wide 8" better
    #   • More context = better classification
    #
    # Mathematical explanation:
    #   Layer 1: 5×5 receptive field
    #   Pooling: reduces by 2×
    #   Layer 2: 5×5 on pooled input
    #   Effective receptive field = 5 + (5-1)×2 = 5 + 8 = 13
    #
    # KEY INSIGHT #3: Compositional Efficiency
    # -----------------------------------------
    # Consider detecting "top of 9" (loop + stem connection):
    #
    # Shallow CNN Approach:
    #   • Need specific filter for "top-of-9 at position (x,y)"
    #   • Different positions require different filters
    #   • 20 filters must encode EVERYTHING
    #   • Limited expressiveness
    #
    # Deep CNN Approach:
    #   • Layer 1: Learn "curve" (reusable)
    #   • Layer 1: Learn "vertical line" (reusable)
    #   • Layer 2: Learn "curve + vertical = top of 9"
    #   • Layer 2 can combine ANY of 20 layer-1 features
    #   • 20 × 40 = 800 possible feature combinations
    #   • Exponentially more expressive!
    #
    # This is called COMPOSITIONAL REPRESENTATION:
    #   • Express complex patterns as compositions of simple ones
    #   • Fundamental to deep learning success
    #   • Why "deep" beats "wide"
    #
    # KEY INSIGHT #4: Parameter Efficiency Through Depth
    # ---------------------------------------------------
    # Shallow CNN (network_3.0.0.py):
    #   Conv: 520 params
    #   FC: 288,100 params (2,880→100)
    #   Total: ~289,630 params
    #   Problem: Most parameters in FC layer (fully connected bottleneck)
    #
    # Deep CNN (this network):
    #   Conv1: 520 params
    #   Conv2: 20,040 params
    #   FC: 64,100 params (640→100)
    #   Total: ~85,670 params (3.4× FEWER!)
    #   Benefit: More work in conv layers (efficient), less in FC (expensive)
    #
    # Key Principle: "Push computation into convolutional layers"
    #   • Conv layers: parameter sharing (efficient)
    #   • FC layers: no sharing (expensive)
    #   • More conv layers = more efficiency
    #   • This scales to very deep networks (ResNet-50, ResNet-152)
    #
    # COMPARISON ACROSS ALL THREE NETWORKS:
    # --------------------------------------
    #
    # network_3.0.1.py (Fully Connected):
    #   Architecture: [784→330→100→10]
    #   Parameters: ~293,160
    #   Accuracy: ~97.0%
    #   Errors per 10K: ~300
    #   Limitation: No spatial structure, no hierarchy
    #
    # network_3.0.0.py (Shallow CNN):
    #   Architecture: Conv→Pool→FC→Softmax
    #   Parameters: ~289,630
    #   Accuracy: ~98.5%
    #   Errors per 10K: ~150
    #   Limitation: Single conv layer limits feature hierarchy
    #
    # network_3.0.2.py (Deep CNN - THIS FILE):
    #   Architecture: Conv→Pool→Conv→Pool→FC→Softmax
    #   Parameters: ~85,670 (3.4× fewer than shallow CNN!)
    #   Accuracy: ~99.0%
    #   Errors per 10K: ~100
    #   Advantage: Hierarchical features + larger receptive fields
    #
    # PROGRESSION:
    #   FC → Shallow CNN: +1.5% (exploit spatial structure)
    #   Shallow CNN → Deep CNN: +0.5% (exploit hierarchy)
    #   Total improvement: +2.0% = 200 fewer errors per 10K
    #   Error reduction: 66% (300 → 100 errors)
    #
    # THE 99% BARRIER:
    # ----------------
    # Getting from 98.5% to 99% is harder than 95% to 98.5%:
    #   • Easy examples are already learned
    #   • Remaining errors are hard cases:
    #     - Ambiguous digits (sloppy handwriting)
    #     - Edge cases (unusual styles)
    #     - Actual label errors in MNIST
    #   • Need better features to distinguish hard cases
    #   • Deep CNNs provide these better features
    #
    # VISUALIZING THE LEARNED FEATURES:
    # ----------------------------------
    # If we could visualize the learned filters (advanced topic):
    #
    # Layer 1 filters would show:
    #   • Horizontal edge detector
    #   • Vertical edge detector
    #   • Diagonal edges (/ and \)
    #   • Curve detectors
    #   • Corner detectors
    #   • Texture patterns
    #
    # Layer 2 filters would show:
    #   • Loop shapes (for 0, 6, 8, 9)
    #   • Stem patterns (vertical bars)
    #   • Cross patterns (for 4, 7)
    #   • Curve combinations (S-shapes for 2, 5)
    #
    # This hierarchy is learned automatically through backpropagation!
    # Network discovers optimal features for digit recognition.
    #
    # PATH TO 99.5%+:
    # ---------------
    # This network achieves ~99%. To push further:
    #
    # 1. More conv layers (3+ layers)
    #    • Even more hierarchical features
    #    • Expected: +0.1-0.2%
    #
    # 2. More filters per layer
    #    • Learn more diverse features
    #    • Expected: +0.1-0.2%
    #
    # 3. Better regularization strategies
    #    • Combine dropout + L2 optimally
    #    • Expected: +0.1%
    #
    # 4. Architectural innovations
    #    • Batch normalization
    #    • Residual connections
    #    • Expected: +0.2-0.3%
    #
    # 5. Ensemble methods
    #    • Train multiple networks, average predictions
    #    • Expected: +0.2-0.3%
    #
    # We'll explore these in network_3.3.x series!
    #
    # WHAT YOU'VE LEARNED:
    # --------------------
    # ✓ Why depth matters in CNNs (hierarchical features)
    # ✓ How multiple conv layers create feature hierarchies
    # ✓ Why receptive fields grow with depth
    # ✓ How compositional representation works
    # ✓ Why deeper CNNs can be more parameter efficient
    # ✓ How to design and train a deep CNN
    # ✓ The path from 95% → 97% → 98.5% → 99%
    
    print("\n" + "=" * 75)
    print("Training complete!")
    print("=" * 75)
    print("\n✓ Expected accuracy: ~99.0%")
    print("✓ Improvement: +0.5% over shallow CNN (50% error reduction!)")
    print("✓ Key lesson: Depth enables hierarchical feature learning")
    print()
    print("COMPARISON:")
    print("  Fully connected (3.0.1):  ~97.0%  (no spatial structure)")
    print("  Shallow CNN (3.0.0):      ~98.5%  (spatial structure)")
    print("  Deep CNN (3.0.2):         ~99.0%  (hierarchical features!)")
    print()
    print("✓ Next: network_3.1.x for activation function experiments (ReLU focus)")

if __name__ == "__main__":
    main()


