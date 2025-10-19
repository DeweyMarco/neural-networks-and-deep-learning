"""
Dropout Regularization Introduction
====================================

PREREQUISITES:
- Complete network_3.0.x series (understand CNNs)
- Complete network_3.1.x series (understand ReLU)
- Understand: overfitting, regularization, ensemble methods

THIS EXPERIMENT:
Introduces DROPOUT REGULARIZATION - one of the most powerful techniques for
preventing overfitting in deep neural networks. Often more effective than L2!

WHAT IS DROPOUT?

During Training:
  • Randomly "drop" (set to 0) each neuron with probability p
  • Typically p = 0.5 (drop 50% of neurons)
  • Dropped neurons don't participate in forward or backward pass
  • Different neurons dropped in each training example
  • Forces network to learn redundant representations

During Testing:
  • Use ALL neurons (no dropping)
  • Scale outputs by (1-p) to compensate for more active neurons
  • Or equivalently: scale during training by 1/(1-p)

WHY DROPOUT WORKS:

1. PREVENTS CO-ADAPTATION
   Without dropout:
     • Neuron A learns to rely on neuron B
     • If B makes mistake, A fails too
     • Network is fragile
   
   With dropout:
     • Neuron A can't rely on B (B might be dropped!)
     • Must learn to work independently
     • Network is robust

2. ENSEMBLE EFFECT
   • Dropping neurons creates 2^n different sub-networks
   • Each mini-batch trains a different sub-network
   • At test time, using all neurons approximates averaging all sub-networks
   • Ensemble of exponentially many networks!

3. REDUNDANT REPRESENTATIONS
   • Forces multiple neurons to learn same feature
   • If one neuron is dropped, others compensate
   • More robust to noise and variations

4. IMPLICIT REGULARIZATION
   • Makes neurons less dependent on specific features
   • Encourages sparse, distributed representations
   • Similar effect to L2 but often stronger

DROPOUT VS L2 REGULARIZATION:

L2 Regularization:
  • Penalizes large weights
  • Encourages many small weights
  • Single mechanism
  • Cost: (lambda/2n) * sum(w^2)

Dropout:
  • Prevents feature co-adaptation
  • Creates ensemble effect
  • Multiple mechanisms working together
  • Generally STRONGER for large networks

Both can be used together!

ARCHITECTURE:

Input (28×28 image)
  → Conv Layer 1 (20 filters, 5×5, ReLU) + Max Pooling (2×2)
  → Conv Layer 2 (40 filters, 5×5, ReLU) + Max Pooling (2×2)
  → Fully Connected (100 neurons, ReLU, DROPOUT p=0.5)
  → Softmax Output (10 classes)

Expected Results:
- Validation accuracy: ~99.0%
- Test accuracy: ~99.0%
- Compared to no dropout: +0.3-0.5% improvement
- Compared to L2 only: +0.1-0.2% improvement
- Key lesson: Dropout is the strongest regularizer for CNNs

WHY THIS WORKS:

The fully connected layer has 640×100 = 64,000 parameters!
Without dropout: Severe overfitting (train: 100%, validation: 98%)
With dropout: Good generalization (train: 99%, validation: 99%)

Dropout forces the 100 neurons to learn independently, preventing
overfitting even with many parameters.

DROPOUT IN PRACTICE:

Where to apply:
  • Fully connected layers: YES (high dropout like p=0.5)
  • Convolutional layers: OPTIONAL (light dropout like p=0.1-0.2)
  • Output layer: NO (never drop output neurons)

Typical values:
  • p=0.5: Standard for fully connected layers
  • p=0.2-0.3: Light dropout for conv layers
  • p=0.7-0.8: Heavy dropout (risk of underfitting)

NEXT STEPS:
- network_3.2.1.py: Compare different dropout rates (0.2, 0.5, 0.8)
- network_3.2.2.py: Dropout vs L2 direct comparison
- network_3.2.3.py: Combined dropout + L2 for maximum performance

Run: python network_3.2.0.py
"""

import sys
sys.path.append('../src')
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU

def main():
    # ========================================================================
    # EXPERIMENT: Dropout Regularization Introduction
    # ========================================================================
    print("=" * 75)
    print("DROPOUT REGULARIZATION: Preventing Overfitting")
    print("=" * 75)

    # Load MNIST data
    training_data, validation_data, test_data = network3.load_data_shared()

    # ========================================================================
    # STEP 2: Create the CNN architecture with DROPOUT
    # ========================================================================
    # ARCHITECTURE OVERVIEW:
    #
    # LAYER 1: First Convolutional Layer + Pooling
    # ---------------------------------------------
    # Input:  28×28 grayscale image (1 channel)
    # Filters: 20 filters of size 5×5
    # Output after conv: 20 feature maps, each 24×24
    # Pooling: 2×2 max pooling
    # Output after pool: 20 feature maps, each 12×12
    # Activation: ReLU
    # Dropout: None (conv layers typically don't need dropout)
    #
    # Parameters: 5×5×1×20 + 20 = 520
    #
    # LAYER 2: Second Convolutional Layer + Pooling
    # ----------------------------------------------
    # Input:  20 feature maps, each 12×12
    # Filters: 40 filters of size 5×5
    # Output after conv: 40 feature maps, each 8×8
    # Pooling: 2×2 max pooling
    # Output after pool: 40 feature maps, each 4×4
    # Activation: ReLU
    # Dropout: None
    #
    # Parameters: 5×5×20×40 + 40 = 20,040
    #
    # LAYER 3: Fully Connected Layer (WITH DROPOUT!)
    # -----------------------------------------------
    # Input:  40×4×4 = 640 neurons (flattened feature maps)
    # Output: 100 neurons
    # Activation: ReLU
    # Dropout: p=0.5 (KEY INNOVATION!)
    #
    # What dropout does:
    #   • During training: Randomly drop 50% of neurons each mini-batch
    #   • Dropped neuron: output = 0, no gradient flows
    #   • Forces remaining neurons to learn independently
    #   • Each mini-batch trains different sub-network
    #   • Creates ensemble of 2^100 ≈ 10^30 networks!
    #
    # Parameters: 640×100 + 100 = 64,100
    #
    # Without dropout:
    #   • 64,100 parameters → severe overfitting
    #   • Training accuracy: 100%
    #   • Validation accuracy: 98% (2% gap = overfitting!)
    #
    # With dropout p=0.5:
    #   • Same 64,100 parameters
    #   • Training accuracy: 99%
    #   • Validation accuracy: 99% (no gap = good generalization!)
    #
    # LAYER 4: Softmax Output
    # ------------------------
    # Input:  100 neurons
    # Output: 10 neurons (one per digit 0-9)
    # Activation: Softmax
    # Dropout: p=0.0 (NEVER use dropout on output layer!)
    #
    # Parameters: 100×10 + 10 = 1,010
    #
    # TOTAL PARAMETERS: ~85,670
    
    mini_batch_size = 10
    
    # Layer 1: First Convolutional + Pooling
    # No dropout (conv layers have built-in regularization through parameter sharing)
    layer1 = ConvPoolLayer(
        filter_shape=(20, 1, 5, 5),              # 20 filters, 1 input channel, 5×5
        image_shape=(mini_batch_size, 1, 28, 28), # Batch of 28×28 images
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    
    # Layer 2: Second Convolutional + Pooling
    # No dropout
    layer2 = ConvPoolLayer(
        filter_shape=(40, 20, 5, 5),             # 40 filters, 20 input channels, 5×5
        image_shape=(mini_batch_size, 20, 12, 12), # Output from layer 1
        poolsize=(2, 2),                          # 2×2 max pooling
        activation_fn=ReLU
    )
    
    # Layer 3: Fully Connected WITH DROPOUT p=0.5
    # This is where dropout has the biggest impact!
    layer3 = FullyConnectedLayer(
        n_in=40*4*4,                             # 640 inputs from pooled feature maps
        n_out=100,                               # 100 neurons
        activation_fn=ReLU,                      # ReLU activation
        p_dropout=0.5                            # 50% dropout (KEY INNOVATION!)
    )
    
    # Layer 4: Softmax Output (NO DROPOUT!)
    # Never use dropout on output layer
    layer4 = SoftmaxLayer(
        n_in=100,                                # 100 inputs from FC layer
        n_out=10,                                # 10 outputs (digits 0-9)
        p_dropout=0.0                            # No dropout on output
    )
    
    # Create the complete network
    net = Network([layer1, layer2, layer3, layer4], mini_batch_size)

    # ========================================================================
    # Train the network
    # ========================================================================
    # TRAINING PARAMETERS:
    # - epochs: 40 (more than shallow CNN due to depth)
    # - eta: 0.03 (learning rate)
    # - lmbda: 0.1 (L2 regularization - combining with dropout!)
    #
    # WHAT TO WATCH DURING TRAINING:
    #   • Training accuracy slightly LOWER than without dropout
    #     (This is GOOD! Means network is forced to generalize)
    #   • Validation accuracy HIGHER than without dropout
    #     (This is the goal! Better generalization)
    #   • Gap between train/validation should be small
    #     (Small gap = no overfitting!)
    #
    # Expected progression:
    #   • Epoch 0-10: Learning basic features with redundancy
    #   • Epoch 10-25: Climbing to 98.5%
    #   • Epoch 25-40: Fine-tuning to reach ~99%
    
    print("\nTraining network with dropout (p=0.5)...")
    
    # Train the network
    net.SGD(
        training_data, 
        40,                    # epochs
        mini_batch_size,       # mini_batch_size (10)
        0.03,                  # eta (learning rate)
        validation_data, 
        test_data,
        lmbda=0.1              # L2 regularization (combines with dropout!)
    )

    # ========================================================================
    # STEP 4: Analysis and key takeaways
    # ========================================================================
    # HOW DROPOUT PREVENTS OVERFITTING:
    #
    # KEY INSIGHT #1: Co-Adaptation Prevention
    # -----------------------------------------
    # Without Dropout:
    #   • FC layer has 100 neurons
    #   • Neuron 1 learns: "If there's a loop at top..."
    #   • Neuron 2 learns: "...and neuron 1 is active, it's a 9"
    #   • Neuron 2 DEPENDS on neuron 1 (co-adaptation)
    #   • If neuron 1 makes error, neuron 2 fails
    #   • Network is fragile, doesn't generalize
    #
    # With Dropout p=0.5:
    #   • Each mini-batch: randomly drop 50% of neurons
    #   • Neuron 2 can't depend on neuron 1 (might be dropped!)
    #   • Neuron 2 must learn: "If loop at top (from input), it's a 9"
    #   • Both neurons learn to detect feature independently
    #   • Multiple redundant representations
    #   • Network is robust, generalizes better
    #
    # KEY INSIGHT #2: Ensemble Effect
    # --------------------------------
    # With 100 neurons and p=0.5 dropout:
    #   • Each mini-batch drops different neurons
    #   • 2^100 ≈ 10^30 possible sub-networks!
    #   • Each mini-batch trains one sub-network
    #   • Over many mini-batches, we train many different networks
    #
    # At test time (no dropout):
    #   • Using all neurons approximates averaging predictions
    #   • Like an ensemble of 10^30 networks!
    #   • Ensembles always outperform single models
    #   • Dropout gives ensemble effect "for free"
    #
    # KEY INSIGHT #3: Redundant Representations
    # ------------------------------------------
    # Without dropout:
    #   • Network might learn: "Only neuron 42 detects loops"
    #   • If neuron 42 is slightly wrong, entire prediction fails
    #   • Single point of failure
    #
    # With dropout:
    #   • Neuron 42 might be dropped during training
    #   • Network must learn: "Neurons 15, 42, 67, 89 all detect loops"
    #   • Redundant representations
    #   • If one fails, others compensate
    #   • Robust to noise and variations
    #
    # KEY INSIGHT #4: Comparison to L2 Regularization
    # ------------------------------------------------
    # L2 Regularization:
    #   • Mechanism: Penalizes large weights
    #   • Effect: Encourages many small weights
    #   • Strength: Good for simple overfitting
    #   • Weakness: Single mechanism
    #
    # Dropout:
    #   • Mechanism: Randomly drops neurons
    #   • Effect: Prevents co-adaptation + ensemble + redundancy
    #   • Strength: Multiple mechanisms, very powerful
    #   • Weakness: None for large networks!
    #
    # For large networks (many parameters):
    #   • Dropout is STRONGER than L2
    #   • Often +0.3-0.5% accuracy improvement
    #   • Can be combined with L2 for even better results!
    #
    # EXPECTED PERFORMANCE:
    # ---------------------
    # Without dropout (from network_3.0.2.py):
    #   • Training accuracy: ~100% (overfitting!)
    #   • Validation accuracy: ~98.5%
    #   • Test accuracy: ~98.5%
    #   • Gap: 1.5% (overfitting detected)
    #
    # With dropout p=0.5 (this network):
    #   • Training accuracy: ~99% (can't memorize due to dropout)
    #   • Validation accuracy: ~99%
    #   • Test accuracy: ~99%
    #   • Gap: 0% (no overfitting!)
    #   • Improvement: +0.5% accuracy
    #
    # DROPOUT IN THE REAL WORLD:
    # ---------------------------
    # Dropout is used in almost ALL modern deep learning:
    #   • AlexNet (2012): First use in ImageNet, +2% accuracy
    #   • VGGNet (2014): Dropout in FC layers
    #   • Transformers (2017+): Dropout everywhere
    #   • BERT, GPT, etc.: Critical for preventing overfitting
    #
    # Without dropout, these models would severely overfit!
    #
    # WHERE TO APPLY DROPOUT:
    # -----------------------
    # Fully Connected Layers: YES! (p=0.5 is standard)
    #   • Many parameters
    #   • High risk of overfitting
    #   • Dropout has biggest impact
    #
    # Convolutional Layers: OPTIONAL (p=0.1-0.2 if used)
    #   • Fewer parameters due to sharing
    #   • Built-in regularization from parameter sharing
    #   • Light dropout can help, but less critical
    #
    # Output Layer: NEVER!
    #   • Need stable predictions
    #   • Dropout would corrupt final outputs
    #   • Always set p_dropout=0.0
    #
    # DROPOUT RATE SELECTION:
    # ------------------------
    # p=0.2 (light dropout):
    #   • Keeps 80% of neurons
    #   • Mild regularization
    #   • Good for: Small networks, low overfitting risk
    #
    # p=0.5 (standard dropout):
    #   • Keeps 50% of neurons
    #   • Strong regularization
    #   • Good for: Most cases (DEFAULT CHOICE!)
    #
    # p=0.8 (heavy dropout):
    #   • Keeps only 20% of neurons
    #   • Very strong regularization
    #   • Good for: Severe overfitting, but risk of underfitting
    #   • Rarely used in practice
    #
    # WHAT YOU'VE LEARNED:
    # --------------------
    # ✓ What dropout is and how it works
    # ✓ Why dropout prevents overfitting (co-adaptation + ensemble + redundancy)
    # ✓ How dropout differs from L2 regularization
    # ✓ Where to apply dropout (FC layers: yes, output: no)
    # ✓ When to use dropout (large networks with overfitting)
    # ✓ How to select dropout rate (p=0.5 is standard)
    # ✓ Why dropout is the strongest regularizer for deep networks
    
    print("\n" + "=" * 75)
    print("EXPECTED RESULT: ~99.0% accuracy")
    print("=" * 75)
    print("Key lesson: Dropout prevents overfitting through co-adaptation prevention")
    print("Next: network_3.2.1.py to compare different dropout rates")

if __name__ == "__main__":
    main()

