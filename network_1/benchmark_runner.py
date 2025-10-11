"""
Network Architecture Benchmark Runner
======================================

This script runs comparative benchmarks across all 4 network architectures:
- network_1.0.0.py: [784, 30, 10] - Baseline (narrow + shallow)
- network_1.1.0.py: [784, 30, 30, 10] - Depth strategy
- network_1.2.0.py: [784, 60, 10] - Width strategy
- network_1.3.0.py: [784, 60, 60, 10] - Width + Depth strategy

For each architecture, runs multiple trials and records:
- Final test accuracy
- Training time per epoch
- Total training time
- Accuracy progression over epochs

Results are aggregated with mean, std dev, min, and max statistics,
and displayed in comprehensive comparison tables on screen.

CONFIGURATION:
- Adjust NUM_TRIALS, EPOCHS, MINI_BATCH_SIZE, and LEARNING_RATE below
- All output and timing estimates adapt automatically

PRESETS:
- Quick test: NUM_TRIALS=3, EPOCHS=10 (~10-15 minutes)
- Full benchmark: NUM_TRIALS=10, EPOCHS=30 (~85-90 minutes)

Run: python benchmark_runner.py
"""

import sys
sys.path.append('../src')
import mnist_loader
import network
import time
import numpy as np

# Network configurations to benchmark
NETWORK_CONFIGS = [
    {
        'name': 'network_1.0.0',
        'architecture': [784, 30, 10],
        'description': 'Baseline (narrow + shallow)',
        'params': 23860,
        'strategy': 'Narrow + Shallow'
    },
    {
        'name': 'network_1.1.0',
        'architecture': [784, 30, 30, 10],
        'description': 'Depth strategy',
        'params': 24790,
        'strategy': 'Depth (narrow + deep)'
    },
    {
        'name': 'network_1.2.0',
        'architecture': [784, 60, 10],
        'description': 'Width strategy',
        'params': 47710,
        'strategy': 'Width (wide + shallow)'
    },
    {
        'name': 'network_1.3.0',
        'architecture': [784, 60, 60, 10],
        'description': 'Width + Depth strategy',
        'params': 51370,
        'strategy': 'Width + Depth (wide + deep)'
    }
]

# ==============================================================================
# ⚙️  CONFIGURATION - CHANGE THESE 4 VARIABLES TO CUSTOMIZE THE BENCHMARK
# ==============================================================================
# 
# Simply modify the values below and run the script.
# All output, timing estimates, and statistics will adapt automatically.
#
# ------------------------------------------------------------------------------

# Number of training runs per network architecture
NUM_TRIALS = 3          # More trials = better statistics, but longer runtime

# Number of training epochs per run
EPOCHS = 10             # More epochs = higher accuracy, but longer runtime

# Mini-batch size for stochastic gradient descent
MINI_BATCH_SIZE = 10    # Typical: 10 (smaller = noisier but faster convergence)

# Learning rate for stochastic gradient descent
LEARNING_RATE = 5.0     # Typical: 3.0 for sigmoid networks (higher = faster but riskier)

# ==============================================================================
# 📋 PRESET CONFIGURATIONS - Copy & paste one to use:
# ==============================================================================
#
# Ultra Quick (~5 minutes):
#   NUM_TRIALS = 1
#   EPOCHS = 5
#
# Quick Test (~10-15 minutes):
#   NUM_TRIALS = 3
#   EPOCHS = 10
#
# Standard Benchmark (~30-40 minutes):
#   NUM_TRIALS = 5
#   EPOCHS = 20
#
# Full Benchmark (~85-90 minutes):
#   NUM_TRIALS = 10
#   EPOCHS = 30
#
# Production-Quality Benchmark (~3 hours):
#   NUM_TRIALS = 20
#   EPOCHS = 30
#
# ==============================================================================


class BenchmarkNetwork(network.Network):
    """Extended Network class that tracks training metrics"""
    
    def __init__(self, sizes):
        super().__init__(sizes)
        self.epoch_times = []
        self.epoch_accuracies = []
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Modified SGD that tracks time and accuracy per epoch"""
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle and create mini-batches
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Train on mini-batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Evaluate accuracy
            if test_data:
                accuracy = self.evaluate(test_data)
                accuracy_pct = (accuracy / n_test) * 100
                self.epoch_accuracies.append(accuracy_pct)


def run_single_trial(config, trial_num, training_data, test_data):
    """Run a single training trial for a given configuration"""
    print(f"  Trial {trial_num + 1}/{NUM_TRIALS}...", end=' ', flush=True)
    
    # Create network
    net = BenchmarkNetwork(config['architecture'])
    
    # Train network
    start_time = time.time()
    net.SGD(training_data, epochs=EPOCHS, mini_batch_size=MINI_BATCH_SIZE, 
            eta=LEARNING_RATE, test_data=test_data)
    total_time = time.time() - start_time
    
    # Get results
    final_accuracy = net.epoch_accuracies[-1]
    avg_epoch_time = np.mean(net.epoch_times)
    
    print(f"Done! Accuracy: {final_accuracy:.2f}%, Time: {total_time:.1f}s")
    
    return {
        'final_accuracy': final_accuracy,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': net.epoch_times,
        'epoch_accuracies': net.epoch_accuracies
    }


def run_benchmark(config, training_data, test_data):
    """Run multiple trials for a given configuration"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {config['name']} - {config['description']}")
    print(f"Architecture: {config['architecture']}")
    print(f"Parameters: {config['params']:,}")
    print(f"{'='*70}")
    
    results = []
    for trial in range(NUM_TRIALS):
        trial_result = run_single_trial(config, trial, training_data, test_data)
        results.append(trial_result)
    
    # Aggregate statistics
    accuracies = [r['final_accuracy'] for r in results]
    total_times = [r['total_time'] for r in results]
    avg_epoch_times = [r['avg_epoch_time'] for r in results]
    
    stats = {
        'config': config,
        'accuracy': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'all': accuracies
        },
        'total_time': {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'min': np.min(total_times),
            'max': np.max(total_times),
            'all': total_times
        },
        'avg_epoch_time': {
            'mean': np.mean(avg_epoch_times),
            'std': np.std(avg_epoch_times),
            'min': np.min(avg_epoch_times),
            'max': np.max(avg_epoch_times),
            'all': avg_epoch_times
        },
        'raw_results': results
    }
    
    print(f"\n  Final Accuracy: {stats['accuracy']['mean']:.2f}% ± {stats['accuracy']['std']:.2f}%")
    print(f"  (min: {stats['accuracy']['min']:.2f}%, max: {stats['accuracy']['max']:.2f}%)")
    print(f"\n  Total Training Time: {stats['total_time']['mean']:.1f}s ± {stats['total_time']['std']:.1f}s")
    print(f"  Avg Time per Epoch: {stats['avg_epoch_time']['mean']:.2f}s ± {stats['avg_epoch_time']['std']:.2f}s")
    
    return stats


def print_comparison_table(all_stats):
    """Print a comparison table of all network configurations"""
    print("\n" + "="*90)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("="*90)
    
    print("\n" + "-"*90)
    print(f"{'Network':<15} {'Architecture':<20} {'Params':<10} {'Accuracy':<20} {'Speed (s/epoch)':<20}")
    print("-"*90)
    
    for stats in all_stats:
        config = stats['config']
        name = config['name']
        arch = str(config['architecture'])
        params = f"{config['params']:,}"
        
        acc_mean = stats['accuracy']['mean']
        acc_std = stats['accuracy']['std']
        accuracy = f"{acc_mean:.2f}% ± {acc_std:.2f}%"
        
        time_mean = stats['avg_epoch_time']['mean']
        time_std = stats['avg_epoch_time']['std']
        speed = f"{time_mean:.2f}s ± {time_std:.2f}s"
        
        print(f"{name:<15} {arch:<20} {params:<10} {accuracy:<20} {speed:<20}")
    
    print("-"*90)
    
    # Print rankings
    print("\n" + "="*90)
    print("RANKINGS")
    print("="*90)
    
    # Rank by accuracy
    print("\nBy Final Accuracy (highest to lowest):")
    sorted_by_acc = sorted(all_stats, key=lambda x: x['accuracy']['mean'], reverse=True)
    for i, stats in enumerate(sorted_by_acc, 1):
        config = stats['config']
        acc = stats['accuracy']['mean']
        print(f"  {i}. {config['name']:<15} {acc:.2f}%  ({config['strategy']})")
    
    # Rank by speed (fastest to slowest)
    print("\nBy Training Speed (fastest to slowest):")
    sorted_by_speed = sorted(all_stats, key=lambda x: x['avg_epoch_time']['mean'])
    for i, stats in enumerate(sorted_by_speed, 1):
        config = stats['config']
        speed = stats['avg_epoch_time']['mean']
        print(f"  {i}. {config['name']:<15} {speed:.2f}s/epoch  ({config['strategy']})")
    
    # Parameter efficiency
    print("\nBy Parameter Efficiency (accuracy per 1000 params):")
    efficiency_stats = []
    for stats in all_stats:
        config = stats['config']
        acc = stats['accuracy']['mean']
        params = config['params']
        efficiency = (acc / params) * 1000  # accuracy per 1000 params
        efficiency_stats.append((config['name'], efficiency, config['strategy'], acc, params))
    
    sorted_by_efficiency = sorted(efficiency_stats, key=lambda x: x[1], reverse=True)
    for i, (name, eff, strategy, acc, params) in enumerate(sorted_by_efficiency, 1):
        print(f"  {i}. {name:<15} {eff:.3f}  ({acc:.2f}% with {params:,} params)")
    
    # Best overall (balanced score: accuracy - normalized time penalty)
    print("\nBalanced Score (accuracy - time_penalty):")
    max_time = max(s['avg_epoch_time']['mean'] for s in all_stats)
    balanced_scores = []
    for stats in all_stats:
        config = stats['config']
        acc = stats['accuracy']['mean']
        time_norm = (stats['avg_epoch_time']['mean'] / max_time) * 5  # Normalize time to 0-5 scale
        score = acc - time_norm
        balanced_scores.append((config['name'], score, config['strategy'], acc, stats['avg_epoch_time']['mean']))
    
    sorted_by_balanced = sorted(balanced_scores, key=lambda x: x[1], reverse=True)
    for i, (name, score, strategy, acc, time_val) in enumerate(sorted_by_balanced, 1):
        print(f"  {i}. {name:<15} Score: {score:.2f}  (Acc: {acc:.2f}%, Time: {time_val:.2f}s)")
    
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    # Find best accuracy
    best_acc_stats = sorted_by_acc[0]
    best_acc_config = best_acc_stats['config']
    print(f"\n✓ BEST ACCURACY: {best_acc_config['name']} with {best_acc_stats['accuracy']['mean']:.2f}%")
    print(f"  Strategy: {best_acc_config['strategy']}")
    print(f"  Trade-off: {best_acc_config['params']:,} parameters, {best_acc_stats['avg_epoch_time']['mean']:.2f}s per epoch")
    
    # Find fastest
    fastest_stats = sorted_by_speed[0]
    fastest_config = fastest_stats['config']
    print(f"\n✓ FASTEST TRAINING: {fastest_config['name']} with {fastest_stats['avg_epoch_time']['mean']:.2f}s per epoch")
    print(f"  Strategy: {fastest_config['strategy']}")
    print(f"  Trade-off: {fastest_stats['accuracy']['mean']:.2f}% accuracy")
    
    # Find most efficient
    best_eff_name, best_eff_val, best_eff_strategy, _, _ = sorted_by_efficiency[0]
    print(f"\n✓ MOST PARAMETER-EFFICIENT: {best_eff_name}")
    print(f"  Strategy: {best_eff_strategy}")
    print(f"  Achieves {best_eff_val:.3f} accuracy points per 1000 parameters")
    
    # Find balanced
    best_bal_name, best_bal_score, best_bal_strategy, best_bal_acc, best_bal_time = sorted_by_balanced[0]
    print(f"\n✓ BEST BALANCED (accuracy vs speed): {best_bal_name}")
    print(f"  Strategy: {best_bal_strategy}")
    print(f"  Score: {best_bal_score:.2f} ({best_bal_acc:.2f}% accuracy, {best_bal_time:.2f}s per epoch)")
    
    print("\n" + "="*90)


def estimate_benchmark_time():
    """Estimate total benchmark time based on configuration"""
    # Approximate times per epoch for each network (in seconds)
    # Based on typical performance on modern hardware
    epoch_times = {
        'network_1.0.0': 1.4,   # [784, 30, 10]
        'network_1.1.0': 1.6,   # [784, 30, 30, 10]
        'network_1.2.0': 2.3,   # [784, 60, 10]
        'network_1.3.0': 2.7    # [784, 60, 60, 10]
    }
    
    total_seconds = 0
    for config in NETWORK_CONFIGS:
        # Time per network = trials × epochs × time_per_epoch
        estimated_epoch_time = epoch_times[config['name']]
        network_time = NUM_TRIALS * EPOCHS * estimated_epoch_time
        total_seconds += network_time
    
    # Add ~10% overhead for data loading, evaluation, etc.
    total_seconds *= 1.1
    
    minutes = total_seconds / 60
    return minutes


def main():
    print("="*90)
    print("NEURAL NETWORK ARCHITECTURE BENCHMARK")
    print("="*90)
    
    # Estimate benchmark time
    estimated_minutes = estimate_benchmark_time()
    
    print(f"\n CONFIGURATION:")
    print(f"  • Networks to test: {len(NETWORK_CONFIGS)}")
    print(f"  • Trials per network: {NUM_TRIALS}")
    print(f"  • Epochs per trial: {EPOCHS}")
    print(f"  • Mini-batch size: {MINI_BATCH_SIZE}")
    print(f"  • Learning rate: {LEARNING_RATE}")
    print(f"\nTotal training runs: {len(NETWORK_CONFIGS) * NUM_TRIALS} ({len(NETWORK_CONFIGS)} networks × {NUM_TRIALS} trials)")
    print(f"Total epochs: {len(NETWORK_CONFIGS) * NUM_TRIALS * EPOCHS:,} ({len(NETWORK_CONFIGS)} networks × {NUM_TRIALS} trials × {EPOCHS} epochs)")
    
    if estimated_minutes < 1:
        print(f"\nEstimated time: {estimated_minutes*60:.0f} seconds")
    elif estimated_minutes < 60:
        print(f"\nEstimated time: {estimated_minutes:.1f} minutes")
    else:
        hours = estimated_minutes / 60
        print(f"\nEstimated time: {hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    
    print("="*90)
    
    # Load MNIST data once
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    print(f"✓ Loaded {len(training_data)} training samples, {len(test_data)} test samples")
    
    # Run benchmarks for all configurations
    all_stats = []
    overall_start = time.time()
    
    for config in NETWORK_CONFIGS:
        stats = run_benchmark(config, training_data, test_data)
        all_stats.append(stats)
    
    overall_time = time.time() - overall_start
    
    # Print comparison table
    print_comparison_table(all_stats)
    
    # Final summary
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print(f"\nActual time: {overall_time/60:.1f} minutes ({overall_time:.0f} seconds)")
    print(f"Estimated time: {estimated_minutes:.1f} minutes")
    
    accuracy_diff = (overall_time/60) - estimated_minutes
    if abs(accuracy_diff) < 2:
        print(f"✓  Time estimate was accurate!")
    elif accuracy_diff > 0:
        print(f"Took {accuracy_diff:.1f} minutes longer than estimated (system may be slower)")
    else:
        print(f"✓  Completed {abs(accuracy_diff):.1f} minutes faster than estimated!")
    
    print(f"\nConfiguration used:")
    print(f"   • {NUM_TRIALS} trials × {EPOCHS} epochs = {NUM_TRIALS * EPOCHS} runs per network")
    print(f"   • {len(NETWORK_CONFIGS)} networks tested")
    print(f"   • Learning rate: {LEARNING_RATE}, Mini-batch size: {MINI_BATCH_SIZE}")
    
    print("\n" + "="*90)
    print("✓ Benchmark complete! All results displayed above.")
    print("="*90)


if __name__ == "__main__":
    main()

