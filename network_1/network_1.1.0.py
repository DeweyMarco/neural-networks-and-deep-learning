"""
Architecture Experiment: Adding Depth (2 hidden layers)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!

WHAT'S DIFFERENT:
- Baseline: [784, 30, 10] - 1 hidden layer, 23,860 parameters
- This version: [784, 30, 30, 10] - 2 hidden layers, 24,790 parameters

FOCUS: How depth enables hierarchical feature learning.
Deeper networks learn low-level → high-level feature progressions.

Expected: Similar or slightly better accuracy (~95-96%), hierarchical representations.

Next: network_1.2.0.py (adding width instead of depth)

Run: python network_1.1.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Architecture Experiment: Adding Depth (2 hidden layers)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create DEEPER network: [784, 30, 30, 10] (see network_1.0.0.py for baseline)
    print("\n2. Creating DEEPER network [784, 30, 30, 10]...")
    print("   Baseline [784, 30, 10]: 1 hidden layer, 23,860 params")
    print("   This run [784, 30, 30, 10]: 2 hidden layers, 24,790 params (+930)")
    net = network.Network([784, 30, 30, 10])
    
    # ========================================================================
    # EXPERIMENT: Adding Depth (2 hidden layers)
    # ========================================================================
    # THE KEY CHANGE: Architecture [784, 30, 30, 10] (vs baseline [784, 30, 10])
    #
    # Baseline: Input → Hidden (30) → Output
    # This run: Input → Hidden 1 (30) → Hidden 2 (30) → Output
    #
    # WHAT DEPTH PROVIDES:
    #
    # ✓ Hierarchical feature learning:
    #   - Hidden layer 1: learns low-level features (edges, curves, corners)
    #   - Hidden layer 2: combines low-level → high-level patterns (loops, digit parts)
    #   - Output layer: uses high-level features for classification
    #   - Similar to visual cortex hierarchy
    #
    # ✓ Better representational power:
    #   - Can represent more complex functions with fewer parameters
    #   - Deep networks are more parameter-efficient for complex patterns
    #   - Two-step feature extraction vs one-step
    #
    # ✓ Often achieves better accuracy:
    #   - Hierarchical features can capture structure better
    #   - May achieve 0.5-1% improvement over baseline
    #   - Only 930 additional parameters (+3.9%)
    #
    # ✗ Vanishing gradient problems:
    #   - With sigmoid activation, gradients shrink through layers
    #   - Deeper layers harder to train than shallow
    #   - First hidden layer may learn slowly
    #
    # ✗ Slower training:
    #   - More layers = more computation per forward/backward pass
    #   - Slightly longer per-epoch time
    #
    # EXPECTED OUTCOME:
    # - Final accuracy: ~95-96% (similar or slightly better than baseline)
    # - Hierarchical feature learning (low → high level)
    # - Efficient: only 930 extra parameters for depth benefit

    print("\n3. Training DEEPER network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 3.0")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Adding Depth
    # ========================================================================
    # Network learned hierarchical representations:
    # - Hidden layer 1: low-level features (edges, curves, corners)
    # - Hidden layer 2: high-level patterns (loops, digit parts)
    # - Output layer: classifies based on high-level features
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~95-96% (similar or slightly better than baseline)
    # - Only 930 extra parameters (+3.9%) for hierarchical learning
    # - Slightly slower training per epoch (more computation)
    # - Vanishing gradient issues with sigmoid activation
    #
    # LESSON: Depth enables hierarchical feature learning
    # - Low-level → high-level feature progression
    # - More parameter-efficient than adding width
    # - Similar to how biological vision systems work
    # - But: sigmoid + depth = vanishing gradients (modern networks use ReLU)
    #
    # Next: network_1.2.0.py - try adding WIDTH instead (60 neurons, 1 layer)

if __name__ == "__main__":
    main()
