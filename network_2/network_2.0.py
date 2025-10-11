"""
Improved Neural Network with Regularization
============================================

This script demonstrates THREE KEY IMPROVEMENTS over first_network.py:

1. CROSS-ENTROPY COST (instead of quadratic cost)
   - Solves the "learning slowdown" problem when neurons are confidently wrong
   - Gradients remain large when the network makes big mistakes
   - Results in FASTER initial learning

2. L2 REGULARIZATION (weight decay)
   - Penalizes large weights to prevent overfitting
   - Encourages the network to learn simpler, more generalizable patterns
   - Improves accuracy on unseen data (validation/test sets)

3. VALIDATION DATA MONITORING (instead of test data)
   - Uses separate validation set to tune hyperparameters
   - Prevents "peeking" at the test set during development
   - Test set remains truly unseen until final evaluation

Expected accuracy: ~96-97% (vs ~95% with first_network.py)

Run: python second_network.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network2

def main():
    print("=" * 60)
    print("Training Improved Neural Network")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: Load MNIST data
    # ========================================================================
    # Note: We now use THREE datasets (training, validation, test)
    # - Training: Used to learn weights through backpropagation
    # - Validation: Used to monitor progress and tune hyperparameters
    # - Test: Kept completely separate for final evaluation
    print("\n1. Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Loaded {len(training_data)} training samples")
    print(f"   ✓ Loaded {len(validation_data)} validation samples")
    
    # ========================================================================
    # STEP 2: Create improved network with cross-entropy cost
    # ========================================================================
    # KEY IMPROVEMENT #1: Cross-Entropy Cost Function
    # 
    # Why it's better than quadratic cost (used in first_network.py):
    # 
    # Quadratic cost derivative: (a - y) * σ'(z)
    #   Problem: When σ'(z) is small (neuron saturated), learning is SLOW
    #   even when the output is very wrong!
    #
    # Cross-entropy cost derivative: (a - y)
    #   Advantage: No σ'(z) term! Learning speed depends ONLY on the error
    #   When the network is very wrong, gradients are large → fast learning
    #   When the network is nearly correct, gradients are small → fine-tuning
    #
    # Result: Network learns much faster in early epochs
    print("\n2. Creating improved network...")
    print("   Architecture: [784, 30, 10]")
    print("   Cost function: Cross-Entropy (faster learning)")
    print("   Regularization: L2 (λ=5.0)")
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    
    # ========================================================================
    # STEP 3: Train with L2 regularization and monitoring
    # ========================================================================
    # KEY IMPROVEMENT #2: L2 Regularization (Weight Decay)
    #
    # The regularization term adds (λ/2n) * Σw² to the cost function
    # This modifies the weight update rule:
    #   w → w - η*∇C - (η*λ/n)*w
    #   w → (1 - η*λ/n)*w - η*∇C
    #
    # Effect: Weights decay toward zero unless gradient keeps them large
    # 
    # Why this helps:
    # - Prevents any single weight from becoming too influential
    # - Forces network to use many small weights instead of few large ones
    # - Makes the network more robust and less likely to overfit to training data
    # - Improves generalization to new, unseen examples
    #
    # λ (lambda) = 5.0 is the regularization parameter
    #   - Larger λ: More regularization, simpler model (may underfit)
    #   - Smaller λ: Less regularization, more complex model (may overfit)
    #   - λ = 0: No regularization (like first_network.py)
    #
    # KEY IMPROVEMENT #3: Lower Learning Rate
    # 
    # η = 0.5 (instead of 3.0 in first_network.py)
    # Cross-entropy already speeds up learning, so we don't need
    # such a large learning rate. This provides more stable training.
    print("\n3. Training network...")
    print("   Epochs: 30")
    print("   Mini-batch size: 10")
    print("   Learning rate: 0.5")
    print("   Regularization λ: 5.0")
    print("\n" + "-" * 60)
    
    # Run stochastic gradient descent with monitoring enabled
    # Returns: cost and accuracy on both evaluation and training sets
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5,
                lmbda=5.0,  # L2 regularization strength
                evaluation_data=validation_data,  # Monitor validation, not test!
                monitor_evaluation_accuracy=True,  # Track validation accuracy
                monitor_training_accuracy=True)    # Track training accuracy
    
    # Why monitor both training and validation accuracy?
    # - Training accuracy shows if the network CAN learn the patterns
    # - Validation accuracy shows if the network GENERALIZES well
    # - If training >> validation: overfitting (regularization helps!)
    # - If both are low: underfitting (need more capacity or training)
    
    # ========================================================================
    # STEP 4: Save the trained model
    # ========================================================================
    # Network2 includes functionality to serialize weights and biases to JSON
    # This allows us to:
    # - Reuse the trained network without retraining
    # - Share trained models with others
    # - Deploy the model in production
    print("\n4. Saving model...")
    net.save("trained_network.json")
    print("   ✓ Model saved to 'trained_network.json'")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nFinal validation accuracy: {evaluation_accuracy[-1]} / {len(validation_data)}")
    print(f"Percentage: {100.0 * evaluation_accuracy[-1] / len(validation_data):.2f}%")

if __name__ == "__main__":
    main()

