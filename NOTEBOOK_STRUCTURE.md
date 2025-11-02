# Diffusion Notebook Structure

## Overview
The notebook has been reorganized into separate cells to allow efficient experimentation with different hyperparameters without regenerating the expensive training dataset.

## Cell Structure

### Cell 1: Configuration Parameters
**Purpose**: Set all hyperparameters  
**Run**: Every time you want to change settings  
**Time**: Instant

**Key Parameters**:
- `HIDDEN_DIM`, `NUM_LAYERS`: Model architecture
- `LEARNING_RATE`, `BATCH_SIZE`: Training hyperparameters
- `K_MAX`, `WEIGHT_DECAY`: Training strategy
- `DATASET_SIZE`: Number of sequences to generate

### Cell 2: Load Embedding Model Classes
**Purpose**: Define Sudoku2Vec and related classes  
**Run**: Once per session  
**Time**: Instant

### Cell 3: Define Diffusion Model & Training
**Purpose**: Define `SudokuDiffusionModel` and `train_sudoku_diffusion()`  
**Run**: Once per session  
**Time**: Instant

### Cell 4: GPU Memory Check
**Purpose**: Clear GPU memory and check available space  
**Run**: As needed to free memory  
**Time**: Instant

### Cell 5: Load Embedding Model ⚡
**Purpose**: Load pre-trained Sudoku2Vec embeddings  
**Run**: Once per session  
**Time**: ~1 second

**Output**: `embedding_layer` variable

### Cell 6: Generate Dataset 🐌
**Purpose**: Generate training diffusion sequences  
**Run**: **ONCE** - then reuse for all experiments!  
**Time**: ~4 minutes for 20k sequences

**Output**: 
- `sequences`: Raw diffusion sequences tensor
- `train_dataset`: PyTorch Dataset wrapper

**💡 Key Benefit**: This is the slowest step. Run it once, then experiment freely with Cells 7 & 8!

### Cell 7: Initialize Model ⚡
**Purpose**: Create and compile the diffusion model  
**Run**: Every time you want to try a different model architecture  
**Time**: ~5 seconds

**Rerun to experiment with**:
- `HIDDEN_DIM`: Network width (32, 64, 128, 256)
- `NUM_LAYERS`: Network depth (6, 9, 12)
- `KERNEL_SIZE`: Convolution kernel size
- `NUM_GROUPS`: GroupNorm groups

**Output**: `model` variable (compiled and ready)

### Cell 8: Train Model 🚀
**Purpose**: Run the training loop  
**Run**: Every time you want to try different training settings  
**Time**: Depends on `NUM_EPOCHS` and `BATCH_SIZE`

**Rerun to experiment with**:
- `LEARNING_RATE`: Optimizer learning rate
- `BATCH_SIZE`: Batch size (affects speed and memory)
- `K_MAX`: Multi-step prediction horizon
- `WEIGHT_DECAY`: Regularization strength
- `NUM_EPOCHS`: How long to train
- `LOG_INTERVAL`, `EVAL_INTERVAL`: Logging frequency

**Output**: 
- `model`: Trained model
- `losses`, `accuracies`: Training metrics

## Typical Workflow

### First Time Setup
```
1. Run Cell 1 (set hyperparameters)
2. Run Cell 2 (load classes)
3. Run Cell 3 (define model/training)
4. Run Cell 4 (clear GPU memory)
5. Run Cell 5 (load embeddings) ← ~1 second
6. Run Cell 6 (generate dataset) ← ~4 minutes ⏰
7. Run Cell 7 (initialize model) ← ~5 seconds
8. Run Cell 8 (train model) ← varies
```

### Experiment with Different Model Architectures
```
1. Modify Cell 1: Change HIDDEN_DIM, NUM_LAYERS, etc.
2. Run Cell 7 (initialize new model) ← ~5 seconds
3. Run Cell 8 (train model) ← varies

✓ No need to rerun Cell 6! Dataset is reused.
```

### Experiment with Different Training Settings
```
1. Modify Cell 1: Change LEARNING_RATE, BATCH_SIZE, K_MAX, etc.
2. Run Cell 7 (reinitialize model) ← ~5 seconds
3. Run Cell 8 (train with new settings) ← varies

✓ No need to rerun Cell 6! Dataset is reused.
```

### Try Multiple Experiments
```
# Experiment 1: Small model, high LR
HIDDEN_DIM = 32
LEARNING_RATE = 1e-3
→ Run Cells 7 & 8

# Experiment 2: Large model, low LR
HIDDEN_DIM = 256
LEARNING_RATE = 1e-5
→ Run Cells 7 & 8

# Experiment 3: Medium model, different K_MAX
HIDDEN_DIM = 128
K_MAX = 3
→ Run Cells 7 & 8

✓ All experiments use the same dataset from Cell 6!
✓ Fair comparison between experiments
✓ Save ~4 minutes per experiment
```

## Time Savings

### Before (Single Cell)
Every experiment required:
- Load embeddings: ~1 second
- Generate dataset: ~4 minutes ⏰
- Initialize model: ~5 seconds
- Train model: varies

**Total per experiment**: ~4+ minutes of overhead

### After (Split Cells)
First experiment:
- Cell 5: ~1 second
- Cell 6: ~4 minutes ⏰ (run once!)
- Cell 7: ~5 seconds
- Cell 8: varies

Subsequent experiments:
- Cell 7: ~5 seconds
- Cell 8: varies

**Time saved per additional experiment**: ~4 minutes! 🎉

## Tips

1. **Generate a large dataset once**: Set `DATASET_SIZE=20000` in Cell 1, run Cell 6 once, then experiment freely

2. **Compare fairly**: Use the same dataset (from Cell 6) for all experiments to ensure fair comparison

3. **Save checkpoints**: Consider saving the model after training:
   ```python
   torch.save(model.state_dict(), 'model_checkpoint.pt')
   ```

4. **Monitor GPU memory**: Run Cell 4 between experiments if you encounter OOM errors

5. **Document experiments**: Keep notes on which hyperparameters you tried and their results

## Common Questions

**Q: Do I need to rerun Cell 6 if I change DATASET_SIZE?**  
A: Yes, but only if you want a different dataset size. Otherwise, reuse the existing dataset.

**Q: Can I modify hyperparameters in Cell 1 and just rerun Cell 8?**  
A: For training hyperparameters (LR, BATCH_SIZE, K_MAX), yes! But you should also rerun Cell 7 to reinitialize the model.

**Q: What if I want to try a completely different model architecture?**  
A: Modify Cell 3 to change the model definition, then rerun Cells 3, 7, and 8.

**Q: How do I resume training from a checkpoint?**  
A: Load the model state in Cell 7 after initialization:
```python
model.load_state_dict(torch.load('checkpoint.pt'))
```

**Q: Can I generate multiple datasets?**  
A: Yes! Save different datasets with different names:
```python
sequences_small = generator.generate_diffusion_sequence(size=5000)
sequences_large = generator.generate_diffusion_sequence(size=20000)
```

