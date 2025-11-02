# Weights & Biases (wandb) Integration Guide

This guide explains the wandb hyperparameter sweep implementation added to `llm_on_sudoku.ipynb`.

## Overview

I've modified your learning rate experiment to use **Weights & Biases (wandb)** for comprehensive hyperparameter sweeps. The new implementation sweeps over:

- **learning_rate**: [1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3]
- **embedding_dim**: [10, 15, 20, 30]
- **attention_dim**: [9, 18, 27, 36]
- **n_heads**: [1, 3, 9]

## What I Added

### 1. **Markdown Tutorial Cell** (Cell 8)
A comprehensive introduction to wandb covering:
- What wandb is and why it's useful
- Setup instructions (installation, account creation, login)
- Key concepts (init, config, log, finish, sweep, agent)
- Sweep methods (grid, random, Bayesian)

### 2. **Simple Example Cell** (Cell 9)
A didactic single-run example that demonstrates:
- How to initialize a wandb run
- How to log metrics during training
- How to finish and upload results
- Step-by-step comments explaining each part

**Purpose**: Run this first to familiarize yourself with wandb basics before running the full sweep.

### 3. **Full Hyperparameter Sweep Cell** (Cell 10)
The main sweep implementation with:
- Comprehensive sweep configuration
- Training function that wandb calls for each configuration
- Automatic logging of all metrics
- Validation of hyperparameter constraints (attention_dim % n_heads == 0)
- Dataset caching for efficiency

**Key Features**:
- Uses grid search by default (can change to 'random' or 'bayes')
- Trains for 50 epochs per configuration (reduced from 100 for faster sweeps)
- Validates every 5 epochs
- Logs training loss, accuracy, validation loss, validation accuracy, and learning rate

### 4. **Tips and Best Practices Cell** (Cell 11)
Advanced guidance covering:
- Detailed explanation of sweep methods
- Configuration options (discrete values, continuous ranges, categorical)
- Logging best practices
- Advanced features (model artifacts, custom plots, histograms)
- How to analyze sweep results
- Common issues and solutions
- Programmatic analysis using wandb API

## Setup Instructions

### 1. Install wandb
```bash
pip install wandb
```

### 2. Create Account and Login
```bash
# In terminal or notebook cell
wandb login
```
- Go to https://wandb.ai and create a free account
- Copy your API key when prompted

### 3. Run the Simple Example (Cell 9)
Start with the simple example to learn wandb basics:
- It trains one model with fixed hyperparameters
- Logs metrics to wandb
- Takes about 5-10 minutes

### 4. Run the Full Sweep (Cell 10)
Once comfortable with wandb:
- The full sweep will run many configurations
- For testing, modify the `count` parameter:
  ```python
  wandb.agent(sweep_id, function=train_with_wandb, count=5)  # Only 5 runs
  ```
- For full sweep, use `count=None` (will run 60-80 valid configurations)

## How It Works

### Sweep Configuration
```python
sweep_config = {
    'method': 'grid',  # Try all combinations
    'metric': {
        'name': 'val_accuracy',  # Optimize validation accuracy
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'values': [1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3]},
        'embedding_dim': {'values': [10, 15, 20, 30]},
        'attention_dim': {'values': [9, 18, 27, 36]},
        'n_heads': {'values': [1, 3, 9]},
        'epochs': {'value': 50},  # Fixed
        'batch_size': {'value': 512},  # Fixed
    }
}
```

### Training Function
The `train_with_wandb()` function:
1. Initializes a wandb run with `wandb.init()`
2. Gets hyperparameters from `wandb.config`
3. Validates constraints (attention_dim % n_heads == 0)
4. Creates model with current hyperparameters
5. Trains for specified epochs
6. Logs metrics with `wandb.log()` at each epoch
7. Finishes with `wandb.finish()`

### Sweep Execution
```python
# Create sweep
sweep_id = wandb.sweep(sweep_config, project='sudoku-llm-hyperparameter-sweep')

# Run sweep agent
wandb.agent(sweep_id, function=train_with_wandb, count=None)
```

## Analyzing Results

### In the wandb Dashboard

1. **Table View**
   - Sort runs by validation accuracy
   - See all hyperparameters and metrics in one place
   - Export as CSV for further analysis

2. **Charts Panel**
   - Compare training curves across all runs
   - See how different hyperparameters affect convergence
   - Overlay multiple runs

3. **Parallel Coordinates Plot**
   - Visualize high-dimensional relationships
   - Color by performance to see patterns
   - Identify which hyperparameters matter most

4. **Parameter Importance**
   - Shows correlation between each hyperparameter and target metric
   - Helps focus on important hyperparameters

### Programmatically

```python
import wandb

api = wandb.Api()
sweep = api.sweep("your-entity/sudoku-llm-hyperparameter-sweep/sweep-id")

# Get best run
runs = sweep.runs
best_run = max(runs, key=lambda r: r.summary.get('val_accuracy', 0))

print(f"Best configuration:")
print(f"  Learning rate: {best_run.config['learning_rate']}")
print(f"  Embedding dim: {best_run.config['embedding_dim']}")
print(f"  Attention dim: {best_run.config['attention_dim']}")
print(f"  N heads: {best_run.config['n_heads']}")
print(f"  Val accuracy: {best_run.summary['val_accuracy']:.4f}")
```

## Key Differences from Original

| Original | New (wandb) |
|----------|-------------|
| Manual loop over learning rates | Automatic sweep over multiple hyperparameters |
| Local storage of results | Cloud storage with web dashboard |
| Manual plotting | Automatic real-time plots |
| Only learning rate varied | Learning rate, embedding_dim, attention_dim, n_heads |
| Results in notebook | Results accessible anywhere via web |
| No comparison tools | Rich comparison and analysis tools |

## Tips for Your First Sweep

1. **Start Small**: Use `count=5` to test the setup
2. **Check Dashboard**: Make sure runs appear at https://wandb.ai
3. **Monitor Progress**: Watch the sweep dashboard in real-time
4. **Adjust Method**: Try 'random' instead of 'grid' for faster exploration
5. **Refine Search**: After initial sweep, create a new sweep focused on promising regions

## Advanced: Bayesian Optimization

For more efficient search, change the method to 'bayes':

```python
sweep_config = {
    'method': 'bayes',  # Intelligent search
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'embedding_dim': {
            'distribution': 'int_uniform',
            'min': 10,
            'max': 50
        },
        # ... other parameters
    }
}
```

Bayesian optimization learns from previous runs and intelligently chooses the next configuration to try.

## Troubleshooting

### "wandb: ERROR Error uploading"
- Check internet connection
- Try `wandb.finish()` again
- Check wandb status at https://status.wandb.ai

### Sweep creates too many runs
- Use `count` parameter to limit: `wandb.agent(sweep_id, function=train, count=10)`
- Change method from 'grid' to 'random'

### Can't see runs in dashboard
- Make sure you're logged into the correct account
- Check project name matches
- Verify API key is correct

### Runs are slow
- Reduce logging frequency
- Log only essential metrics
- Use offline mode for testing: `os.environ['WANDB_MODE'] = 'offline'`

## Next Steps

1. **Run Simple Example**: Familiarize yourself with wandb (Cell 9)
2. **Test Sweep**: Run a small sweep with `count=5` (Cell 10)
3. **Analyze Results**: Explore the wandb dashboard
4. **Full Sweep**: Run the complete sweep (may take hours/days depending on hardware)
5. **Refine**: Create a new sweep focused on best-performing regions
6. **Train Final Model**: Use best hyperparameters for a long training run

## Resources

- **wandb Documentation**: https://docs.wandb.ai
- **Sweeps Guide**: https://docs.wandb.ai/guides/sweeps
- **Examples**: https://github.com/wandb/examples
- **Community**: https://wandb.ai/community

## Questions?

The notebook cells contain extensive comments and explanations. Each step is documented with:
- What the code does
- Why it's important
- How it fits into the overall workflow

Happy sweeping! 🚀

