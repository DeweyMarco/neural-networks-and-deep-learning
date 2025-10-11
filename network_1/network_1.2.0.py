"""
Architecture Experiment: Adding Width (60 neurons)
========================================

PREREQUISITE: Read network_1.0.0.py for baseline!
ALSO SEE: network_1.1.0.py (adding depth) for comparison.

WHAT'S DIFFERENT:
- Baseline: [784, 30, 10] - 1 hidden layer, 30 neurons, 23,860 parameters
- Depth approach: [784, 30, 30, 10] - 2 hidden layers, 24,790 parameters
- This version: [784, 60, 10] - 1 hidden layer, 60 neurons, 47,710 parameters

FOCUS: How width enables learning diverse features in parallel.
Wider networks have more capacity but learn flat (not hierarchical) features.

Expected: Better accuracy (~96-97%), but 2× more parameters than depth approach.

Next: network_1.3.0.py (combining width AND depth)

Run: python network_1.2.0.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

def main():
    print("=" * 60)
    print("Architecture Experiment: Adding Width (60 neurons)")
    print("=" * 60)

    # Load MNIST data (see network_1.0.0.py for details)
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   Loaded {len(training_data)} training samples")

    # Create WIDER network: [784, 60, 10] (see network_1.0.0.py for baseline)
    print("\n2. Creating WIDER network [784, 60, 10]...")
    print("   Baseline [784, 30, 10]: 1 layer, 30 neurons, 23,860 params")
    print("   Depth [784, 30, 30, 10]: 2 layers, 30 neurons each, 24,790 params")
    print("   This run [784, 60, 10]: 1 layer, 60 neurons, 47,710 params (+100%)")
    net = network.Network([784, 60, 10])
    
    # ========================================================================
    # EXPERIMENT: Adding Width (60 neurons)
    # ========================================================================
    # THE KEY CHANGE: Architecture [784, 60, 10] (vs baseline [784, 30, 10])
    #
    # Baseline: Input → Hidden (30 neurons) → Output
    # This run: Input → Hidden (60 neurons) → Output
    #
    # WHAT WIDTH PROVIDES:
    #
    # ✓ More parallel feature capacity:
    #   - 60 neurons = 60 different feature detectors (vs 30 in baseline)
    #   - Can learn 2× more diverse patterns simultaneously
    #   - Each neuron specializes: edges, curves, textures, etc.
    #   - Richer feature set for output layer to use
    #
    # ✓ Better accuracy:
    #   - More capacity = better performance (~96-97% vs ~95%)
    #   - Can represent more complex decision boundaries
    #   - Often best accuracy for this problem with sigmoid
    #
    # ✓ Avoids vanishing gradients:
    #   - Shallow network (only 3 layers) = easier gradient flow
    #   - Easier to train than deep networks with sigmoid
    #   - All neurons learn at reasonable rates
    #
    # ✗ Many more parameters:
    #   - 47,710 params vs 23,860 (baseline) or 24,790 (depth approach)
    #   - 2× parameters vs baseline, ~2× vs depth
    #   - More memory, more computation per epoch
    #
    # ✗ No hierarchical features:
    #   - Learns "flat" features, not low→high hierarchy
    #   - Less parameter-efficient than depth for complex patterns
    #   - Single-step feature extraction
    #
    # EXPECTED OUTCOME:
    # - Final accuracy: ~96-97% (often best with sigmoid)
    # - 2× more parameters but avoids depth problems
    # - Good choice when vanishing gradients are an issue

    print("\n3. Training WIDER network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 3.0")
    print("\n" + "-" * 60)

    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
    
    # ========================================================================
    # RESULTS: Adding Width
    # ========================================================================
    # Network learned diverse parallel features:
    # - 60 neurons each specializing in different patterns
    # - No hierarchy, but rich diverse feature set
    #
    # KEY OBSERVATIONS:
    # - Final accuracy: ~96-97% (often best for this problem!)
    # - 47,710 parameters (2× baseline, ~2× depth approach)
    # - Easier to train than depth (no vanishing gradients)
    # - Learns flat features, not hierarchical
    #
    # LESSON: Width provides raw capacity
    # - More neurons = more parallel feature detectors
    # - Better accuracy but more parameters
    # - Easier gradient flow (shallow network)
    # - Less parameter-efficient than depth for complex patterns
    #
    # WIDTH vs DEPTH COMPARISON:
    # - Baseline [784, 30, 10]: 23,860 params, ~95%, simple
    # - Depth [784, 30, 30, 10]: 24,790 params, ~95-96%, hierarchical, efficient
    # - Width [784, 60, 10]: 47,710 params, ~96-97%, parallel, high capacity
    #
    # Next: network_1.3.0.py - combine WIDTH + DEPTH for maximum power!

if __name__ == "__main__":
    main()
