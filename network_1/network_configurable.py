"""
Configurable Neural Network
============================

This script allows you to easily experiment with different neural network
configurations and hyperparameters on the MNIST dataset.

CONFIGURABLE PARAMETERS:
- Number of epochs
- Mini-batch size
- Learning rate
- Number of hidden layers
- Size of each hidden layer

Modify the CONFIGURATION section below to experiment with different settings.

Run: python network_configurable.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS TO EXPERIMENT
# ============================================================================

# Training hyperparameters
EPOCHS = 30                 # Number of complete passes through training data
MINI_BATCH_SIZE = 10        # Number of samples per gradient update
LEARNING_RATE = 3.0         # Step size for weight updates (eta)

# Network architecture
# Define the size of each hidden layer in a list
# Examples:
#   [30]           → One hidden layer with 30 neurons
#   [100]          → One hidden layer with 100 neurons
#   [30, 20]       → Two hidden layers: 30 neurons, then 20 neurons
#   [100, 50, 25]  → Three hidden layers: 100, 50, and 25 neurons
HIDDEN_LAYERS = [30]

# ============================================================================
# END CONFIGURATION
# ============================================================================


def build_architecture(hidden_layers):
    """
    Builds the full network architecture including input and output layers.
    
    Args:
        hidden_layers: List of integers specifying size of each hidden layer
        
    Returns:
        List representing full architecture [input, hidden1, hidden2, ..., output]
    """
    # MNIST has 784 input features (28×28 pixels) and 10 output classes (digits 0-9)
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    
    architecture = [INPUT_SIZE] + hidden_layers + [OUTPUT_SIZE]
    return architecture


def print_configuration(architecture):
    """Print the current configuration in a readable format."""
    print("=" * 70)
    print("Configurable Neural Network")
    print("=" * 70)
    
    print("\nTRAINING HYPERPARAMETERS:")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Mini-batch size:  {MINI_BATCH_SIZE}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    
    print("\nNETWORK ARCHITECTURE:")
    print(f"  Full architecture: {architecture}")
    print(f"  - Input layer:     {architecture[0]} neurons (28×28 MNIST images)")
    
    for i, size in enumerate(architecture[1:-1], 1):
        print(f"  - Hidden layer {i}:  {size} neurons")
    
    print(f"  - Output layer:    {architecture[-1]} neurons (digits 0-9)")
    
    # Calculate total parameters
    total_params = 0
    for i in range(len(architecture) - 1):
        weights = architecture[i] * architecture[i+1]
        biases = architecture[i+1]
        total_params += weights + biases
        print(f"\n  Layer {i} → {i+1}: {weights:,} weights + {biases} biases = {weights + biases:,} parameters")
    
    print(f"\n  TOTAL PARAMETERS: {total_params:,}")


def main():
    # Build the full network architecture
    architecture = build_architecture(HIDDEN_LAYERS)
    
    # Print configuration
    print_configuration(architecture)
    
    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    print("\n" + "-" * 70)
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"✓ Loaded {len(training_data)} training samples")
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # ========================================================================
    # STEP 2: Create the neural network
    # ========================================================================
    print("\nCreating network...")
    net = network.Network(architecture)
    print("✓ Network initialized with random weights and biases")
    
    # ========================================================================
    # STEP 3: Train using Stochastic Gradient Descent (SGD)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Training network...")
    print("-" * 70)
    
    # Calculate training statistics
    num_batches_per_epoch = len(training_data) // MINI_BATCH_SIZE
    total_updates = num_batches_per_epoch * EPOCHS
    total_samples_seen = len(training_data) * EPOCHS
    
    print(f"• {num_batches_per_epoch:,} mini-batches per epoch")
    print(f"• {total_updates:,} total weight updates")
    print(f"• {total_samples_seen:,} total training samples processed")
    print()
    
    # Run training
    net.SGD(
        training_data,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        eta=LEARNING_RATE,
        test_data=test_data
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print("\nTIPS FOR EXPERIMENTATION:")
    print("• Increase hidden layer size (e.g., [100]) for more capacity")
    print("• Add more hidden layers (e.g., [100, 50]) for deeper networks")
    print("• Adjust learning rate if training is unstable or too slow")
    print("• Increase epochs for potentially better accuracy")
    print("• Try different mini-batch sizes (common: 10, 32, 64, 128)")
    print("\nSee README.md for more guidance on hyperparameter tuning!")


if __name__ == "__main__":
    main()

