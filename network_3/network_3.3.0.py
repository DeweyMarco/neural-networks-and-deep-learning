"""
Standard CNN Architecture Pattern
==================================

PREREQUISITES:
- Complete network_3.0.x series (understand CNNs)
- Complete network_3.1.x series (understand ReLU)
- Complete network_3.2.x series (understand dropout and combined regularization)

THIS EXPERIMENT:
Introduces the INDUSTRY-STANDARD CNN ARCHITECTURE PATTERN that generalizes
to virtually all computer vision tasks. This pattern has proven successful
across ImageNet, COCO, medical imaging, and countless other domains.

THE STANDARD PATTERN:

Multiple Conv-Pool Blocks → Multiple FC Layers → Softmax Output

Specifically:
  [Conv + Pool] × N → [Fully Connected + Dropout] × M → Softmax

This experiment uses N=2, M=2 for MNIST:
  Conv Block 1 → Conv Block 2 → FC Layer 1 → FC Layer 2 → Softmax

WHY THIS PATTERN IS "STANDARD":

1. PROVEN ACROSS DOMAINS
   • AlexNet (2012): Conv×5 → FC×3 → Won ImageNet
   • VGGNet (2014): Conv×13 → FC×3 → 7.3% ImageNet error
   • Early ResNets: Conv blocks → FC → State-of-the-art
   • Medical imaging: Same pattern for X-rays, MRIs, CT scans

2. HIERARCHICAL FEATURE EXTRACTION
   • Early conv layers: Low-level features (edges, textures)
   • Middle conv layers: Mid-level features (shapes, parts)
   • Late conv layers: High-level features (object parts)
   • FC layers: Global reasoning and classification

3. BALANCED CAPACITY
   • Conv layers: Efficient parameter sharing
   • Pool layers: Translation invariance + dimension reduction
   • FC layers: High-level reasoning with controlled parameters
   • Dropout: Regularization where overfitting risk is highest

4. SCALABILITY
   • Easy to make deeper: Add more conv blocks
   • Easy to make wider: Add more filters per conv layer
   • Easy to tune: Clear hyperparameter roles
   • Proven scaling laws: More compute → better accuracy

ARCHITECTURE DETAILS:

Input (28×28 image)
  ↓
┌────────────────────────────────────┐
│ CONV BLOCK 1                       │
│ • Conv Layer (20 filters, 5×5)     │
│ • ReLU activation                  │
│ • Max Pooling (2×2)                │
│ Output: 20 × 12×12                 │
└────────────────────────────────────┘
  ↓
┌────────────────────────────────────┐
│ CONV BLOCK 2                       │
│ • Conv Layer (40 filters, 5×5)     │
│ • ReLU activation                  │
│ • Max Pooling (2×2)                │
│ Output: 40 × 4×4 = 640 features    │
└────────────────────────────────────┘
  ↓
┌────────────────────────────────────┐
│ FC BLOCK 1                         │
│ • Fully Connected (150 neurons)    │
│ • ReLU activation                  │
│ • Dropout (p=0.5)                  │
└────────────────────────────────────┘
  ↓
┌────────────────────────────────────┐
│ FC BLOCK 2                         │
│ • Fully Connected (100 neurons)    │
│ • ReLU activation                  │
│ • Dropout (p=0.5)                  │
└────────────────────────────────────┘
  ↓
┌────────────────────────────────────┐
│ OUTPUT                             │
│ • Softmax (10 classes)             │
│ • Cross-entropy loss               │
└────────────────────────────────────┘

Expected Results:
- Validation accuracy: ~99.3%
- Test accuracy: ~99.3%
- Parameters: ~110K
- Training time: Moderate (40 epochs)
- Key lesson: This pattern generalizes to ALL vision tasks!

COMPARISON TO PREVIOUS EXPERIMENTS:

Simple CNN (3.0.0):
  • 1 conv block → 1 FC layer
  • ~98.5% accuracy
  • Good starting point

Deep CNN (3.0.2):
  • 2 conv blocks → 1 FC layer
  • ~99.0% accuracy
  • Better hierarchical features

Standard CNN (THIS FILE):
  • 2 conv blocks → 2 FC layers
  • ~99.3% accuracy (+0.3%)
  • Industry-standard pattern
  • More FC capacity for reasoning

DESIGN PRINCIPLES:

1. CONV LAYERS: Spatial feature extraction
   • Start with fewer filters (20)
   • Increase depth gradually (40)
   • Always use ReLU activation
   • Always pool after conv

2. FC LAYERS: High-level reasoning
   • First FC: Larger (150 neurons) - learns feature combinations
   • Second FC: Smaller (100 neurons) - refines representations
   • Always use dropout (p=0.5) - prevents overfitting
   • Decreasing size: 640 → 150 → 100 → 10

3. REGULARIZATION: Multi-layered defense
   • L2 on all weights (λ=0.1)
   • Dropout on all FC layers (p=0.5)
   • Max pooling (implicit regularization)
   • Combined effect: excellent generalization

WHY TWO FC LAYERS HELP:

One FC Layer [640→100]:
  • Must learn combinations AND refine in one step
  • All 640 features directly influence 100 outputs
  • Less specialization
  • Result: ~99.0%

Two FC Layers [640→150→100]:
  • Layer 1 [640→150]: Learn useful feature combinations
  • Layer 2 [150→100]: Refine and select best combinations
  • More specialization and abstraction
  • Result: ~99.3% (+0.3% improvement!)

This is analogous to deep conv layers: more depth = better abstraction!

NEXT STEPS:
- network_3.3.1.py: Even deeper CNN (3 conv blocks) for ~99.4%
- network_3.3.2.py: Optimized architecture for ~99.5%+
- network_3.3.3.py: Ensemble methods for ~99.6%+

Run: python network_3.3.0.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

def main():
    # ========================================================================
    # EXPERIMENT: Standard Industry CNN Architecture
    # ========================================================================
    print("=" * 75)
    print("STANDARD CNN ARCHITECTURE - Industry Pattern")
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
    # STEP 2: Create the STANDARD CNN architecture
    # ========================================================================
    # ARCHITECTURE OVERVIEW:
    #
    # This is the STANDARD PATTERN used across computer vision:
    # Conv Blocks → FC Blocks → Output
    #
    # CONV BLOCK 1: Early Feature Extraction
    # ---------------------------------------
    # Input:  28×28 grayscale image (1 channel)
    # Conv:   20 filters of size 5×5
    #   • Learns 20 different low-level features
    #   • Edges, curves, corners, textures
    #   • Output after conv: 20×24×24
    # Pool:   2×2 max pooling
    #   • Downsamples by factor of 2
    #   • Provides translation invariance
    #   • Output after pool: 20×12×12
    # Activation: ReLU (fast, non-saturating)
    #
    # Parameters: 5×5×1×20 + 20 = 520
    #
    # CONV BLOCK 2: Mid-Level Feature Extraction
    # -------------------------------------------
    # Input:  20×12×12 feature maps from block 1
    # Conv:   40 filters of size 5×5
    #   • Learns 40 mid-level features
    #   • Combinations of edges → shapes
    #   • Loops, stems, digit parts
    #   • Output after conv: 40×8×8
    # Pool:   2×2 max pooling
    #   • Further downsampling
    #   • Larger effective receptive field
    #   • Output after pool: 40×4×4 = 640 features
    # Activation: ReLU
    #
    # Parameters: 5×5×20×40 + 40 = 20,040
    #
    # FC BLOCK 1: Feature Combination Layer
    # --------------------------------------
    # Input:  640 features (flattened from conv block 2)
    # Output: 150 neurons
    # 
    # What it does:
    #   • Learns combinations of spatial features
    #   • "Does this have a loop (feature 5) AND a stem (feature 23)?"
    #   • Creates high-level abstract representations
    #   • More neurons = more expressiveness
    #
    # Activation: ReLU
    # Dropout: p=0.5 (critical for preventing overfitting!)
    #
    # Parameters: 640×150 + 150 = 96,150
    #
    # Why 150 neurons?
    #   • More than output (10) to allow expressiveness
    #   • Less than input (640) to force compression/abstraction
    #   • Middle ground: enough capacity without overfitting
    #
    # FC BLOCK 2: Refinement Layer
    # -----------------------------
    # Input:  150 neurons from FC block 1
    # Output: 100 neurons
    #
    # What it does:
    #   • Refines the 150 feature combinations from FC1
    #   • Selects most discriminative features
    #   • Further abstraction layer
    #   • Prepares for final classification
    #
    # Activation: ReLU
    # Dropout: p=0.5
    #
    # Parameters: 150×100 + 100 = 15,100
    #
    # Why another FC layer?
    #   • Depth in FC space helps (like depth in conv space!)
    #   • FC1: Raw combinations, FC2: Refined selections
    #   • Analogous to: Conv1=edges, Conv2=shapes
    #   • Result: Better abstractions, higher accuracy
    #
    # OUTPUT LAYER: Classification
    # -----------------------------
    # Input:  100 neurons from FC block 2
    # Output: 10 neurons (digits 0-9)
    # Activation: Softmax (probability distribution)
    #
    # Parameters: 100×10 + 10 = 1,010
    #
    # TOTAL PARAMETERS:
    #   Conv Block 1:     520
    #   Conv Block 2:     20,040
    #   FC Block 1:       96,150
    #   FC Block 2:       15,100
    #   Output:           1,010
    #   --------------------------------
    #   TOTAL:            ~132,820 parameters
    #
    # PARAMETER DISTRIBUTION:
    #   Conv layers: 15% (20,560 / 132,820)
    #   FC layers:   84% (112,260 / 132,820)
    #   Note: Most parameters in FC layers, but convs do most work!
    #   This is typical and fine with dropout regularization.
    
    print("\n2. Creating network architecture...")
    
    mini_batch_size = 10
    
    # ========================================================================
    # CONV BLOCK 1: Low-level features
    # ========================================================================
    layer1 = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),              # 20 filters, 1 input channel, 5×5
        image_shape=(mini_batch_size, 1, 28, 28), # Batch of 28×28 images
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    print("   ✓ Conv Block 1: 20 filters (5×5) + pooling → 20×12×12")
    
    # ========================================================================
    # CONV BLOCK 2: Mid-level features
    # ========================================================================
    layer2 = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),             # 40 filters, 20 input channels, 5×5
        image_shape=(mini_batch_size, 20, 12, 12), # Output from layer 1
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    print("   ✓ Conv Block 2: 40 filters (5×5) + pooling → 40×4×4")
    
    # ========================================================================
    # FC BLOCK 1: Feature combination layer (150 neurons)
    # ========================================================================
    layer3 = FullyConnectedLayer(
        n_in=40*4*4,                             # 640 inputs from conv blocks
        n_out=150,                               # 150 neurons (first FC layer)
        activation_fn=ReLU,                      # ReLU activation
        p_dropout=0.5                            # 50% dropout
    )
    print("   ✓ FC Block 1: 640 → 150 neurons (ReLU, dropout 0.5)")
    
    # ========================================================================
    # FC BLOCK 2: Refinement layer (100 neurons)
    # ========================================================================
    layer4 = FullyConnectedLayer(
        n_in=150,                                # 150 inputs from FC block 1
        n_out=100,                               # 100 neurons (second FC layer)
        activation_fn=ReLU,                      # ReLU activation
        p_dropout=0.5                            # 50% dropout
    )
    print("   ✓ FC Block 2: 150 → 100 neurons (ReLU, dropout 0.5)")
    
    # ========================================================================
    # OUTPUT LAYER: Softmax classification
    # ========================================================================
    layer5 = SoftmaxLayer(
        n_in=100,                                # 100 inputs from FC block 2
        n_out=10,                                # 10 outputs (digits 0-9)
        p_dropout=0.0                            # No dropout on output
    )
    print("   ✓ Output: 100 → 10 classes (Softmax)")
    
    # Create the complete network
    net = Network([layer1, layer2, layer3, layer4, layer5], mini_batch_size)
    
    print("   ✓ Total parameters: ~132,820")
    print("\n   Architecture: 2 Conv Blocks → 2 FC Blocks → Softmax")
    print("   This is the INDUSTRY-STANDARD pattern! ⭐")

    # ========================================================================
    # STEP 3: Train the network
    # ========================================================================
    # TRAINING HYPERPARAMETERS:
    # 
    # Epochs: 40
    #   • More than simple CNN (needs time for 2 FC layers to converge)
    #   • Standard for this architecture size
    #
    # Learning rate: 0.03
    #   • Conservative rate for stable training
    #   • Works well with Adam/SGD for this network size
    #
    # L2 regularization: λ=0.1
    #   • Moderate weight decay
    #   • Works well with dropout
    #   • Combined regularization strategy
    #
    # Dropout: p=0.5
    #   • Standard rate for FC layers
    #   • Strong regularization for 100K+ parameters
    #
    # WHAT TO EXPECT DURING TRAINING:
    #   Epoch 0-10:   Learning basic features, ~97% → 98.5%
    #   Epoch 10-25:  Refining features, 98.5% → 99%
    #   Epoch 25-40:  Fine-tuning, 99% → 99.3%
    #   
    #   The two FC layers need time to learn good abstractions!
    
    print("\n3. Training network...")
    print("   Epochs: 40")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.03")
    print("   L2 regularization: 0.1")
    print("   Dropout: 0.5 (on both FC layers)")
    print()
    print("   Training in progress...")
    print("-" * 75)
    
    # Train the network
    net.SGD(
        training_data, 
        40,                    # epochs
        mini_batch_size,       # mini_batch_size (10)
        0.03,                  # eta (learning rate)
        validation_data, 
        test_data,
        lmbda=0.1              # L2 regularization
    )

    # ========================================================================
    # STEP 4: Analysis and Key Takeaways
    # ========================================================================
    # PERFORMANCE COMPARISON:
    # -----------------------
    # network_3.0.0 (1 conv + 1 FC):     ~98.5%
    # network_3.0.2 (2 conv + 1 FC):     ~99.0%
    # network_3.3.0 (2 conv + 2 FC):     ~99.3% ← THIS FILE
    #
    # Improvement: +0.3% from adding second FC layer
    # Error reduction: 100 → 70 errors per 10K (30% fewer!)
    #
    # WHY THIS ARCHITECTURE PATTERN WORKS:
    # -------------------------------------
    #
    # 1. HIERARCHICAL PROCESSING
    #    Conv Blocks: Spatial features (local → global)
    #    FC Blocks: Abstract reasoning (concrete → abstract)
    #    Output: Classification decision
    #    
    #    This matches how information should flow!
    #
    # 2. EFFICIENT PARAMETER USAGE
    #    Conv layers: Few parameters, lots of computation (efficient!)
    #    FC layers: Many parameters, less computation (controlled with dropout)
    #    
    #    Conv does heavy lifting with parameter sharing.
    #    FC does reasoning with regularization.
    #
    # 3. PROVEN SCALABILITY
    #    Want better accuracy? Two clear paths:
    #    • Make conv blocks deeper (more layers)
    #    • Make conv blocks wider (more filters)
    #    
    #    Clear scaling strategy used in VGG, ResNet, EfficientNet!
    #
    # 4. EASY TO UNDERSTAND AND DEBUG
    #    Clear separation of concerns:
    #    • Conv blocks: Feature extraction
    #    • FC blocks: Feature combination and reasoning
    #    • Output: Classification
    #    
    #    Easy to visualize, interpret, and improve!
    #
    # 5. WORKS ACROSS DOMAINS
    #    This exact pattern succeeds in:
    #    • Image classification (ImageNet)
    #    • Object detection (COCO)
    #    • Medical imaging (X-rays, MRIs)
    #    • Satellite imagery
    #    • Document analysis
    #    • Face recognition
    #    
    #    Universal pattern for spatial data!
    #
    # PRACTICAL DESIGN GUIDELINES:
    # -----------------------------
    # For your own CNN architectures:
    #
    # CONV BLOCKS:
    #   • Start with 16-32 filters in first conv layer
    #   • Double filters after each pooling: 32 → 64 → 128
    #   • Use 3×3 or 5×5 filters (3×3 more common now)
    #   • Always: Conv → ReLU → Pool
    #   • Repeat until spatial size is 4×4 to 8×8
    #
    # FC BLOCKS:
    #   • First FC: Larger (more neurons than output)
    #   • Second FC: Smaller (approach output size)
    #   • Always use dropout (p=0.5) on FC layers
    #   • Use 1-3 FC layers (2 is most common)
    #
    # REGULARIZATION:
    #   • L2: λ=0.1 (default starting point)
    #   • Dropout: p=0.5 on FC, p=0.0-0.2 on conv (optional)
    #   • Combine both for best results
    #
    # TRAINING:
    #   • Learning rate: 0.01-0.1 (start with 0.03)
    #   • Epochs: 30-60 for MNIST-sized tasks
    #   • Mini-batch: 10-128 (larger for bigger datasets)
    #   • Monitor validation accuracy, stop if overfitting
    #
    # KEY TAKEAWAYS:
    # --------------
    # ✓ This 2 conv + 2 FC pattern is industry-standard
    # ✓ Achieves ~99.3% on MNIST (top 5% performance)
    # ✓ Two FC layers provide better abstraction than one
    # ✓ This pattern generalizes to virtually ALL vision tasks
    # ✓ Easy to scale: add conv blocks or increase filter counts
    # ✓ Combined regularization (L2 + dropout) is essential
    # ✓ Clear separation: Conv extracts, FC reasons, Output classifies
    #
    # This architecture could be deployed in production!
    
    print("\n" + "=" * 75)
    print("Training complete!")
    print("=" * 75)
    print("\n✓ Expected accuracy: ~99.3%")
    print("✓ This 2 conv + 2 FC pattern is the industry standard")
    print("✓ Next: network_3.3.1.py for deeper CNN (~99.4%)")
    print()

if __name__ == "__main__":
    main()

