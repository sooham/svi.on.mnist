#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sudoku import SudokuGenerator, Sudoku
import math
import subprocess
import gc


# In[ ]:


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Modify these parameters to experiment with different model configurations

# --- Model Architecture Parameters ---
HIDDEN_DIM = 128              # Network width (default: 256, reduced for memory)
NUM_LAYERS = 9                # Number of residual conv blocks (default: 6, reduced for memory)
KERNEL_SIZE = 3               # Conv kernel size
NUM_GROUPS = 8                # GroupNorm groups (must divide HIDDEN_DIM evenly)
NUM_TIMESTEPS = 81            # Fixed at 81 for Sudoku (9x9 grid)

# --- Embedding Parameters ---
USE_LEARNED_EMBEDDINGS = True  # Use learned embeddings from LLM model
EMBEDDING_MODEL_PATH = './sudoku2vec_trained_model.pt'  # Path to saved embedding model
EMBEDDING_DIM = 15            # Dimension of learned embeddings (from LLM model)

# --- Training Hyperparameters ---
DATASET_SIZE = 20000          # Number of diffusion sequences to pre-generate
NUM_EPOCHS = 4000             # Number of training epochs
BATCH_SIZE = 1024              # Batch size (reduced from 1024 for memory)
LEARNING_RATE = 1e-3          # Optimizer learning rate
WEIGHT_DECAY = 1e-4           # AdamW weight decay for regularization
GRAD_CLIP_MAX_NORM = 1.0      # Gradient clipping threshold

# --- Logging & Evaluation ---
LOG_INTERVAL = 10             # Log metrics every N epochs
EVAL_INTERVAL = 50           # Evaluate and sample every N epochs

# --- Diffusion Parameters ---
K_MAX = 6                     # Maximum number of forward steps for multi-step prediction loss (reduced from 10)

# --- Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- Sudoku2Vec ---
ATTENTION_DIM = 9
N_HEADS = 9

print("Configuration loaded:")
print(f"  Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}, kernel_size={KERNEL_SIZE}")
print(f"  Embeddings: use_learned={USE_LEARNED_EMBEDDINGS}, embedding_dim={EMBEDDING_DIM}")
print(f"  Training: dataset_size={DATASET_SIZE}, epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
print(f"  Diffusion: k_max={K_MAX}")
print(f"  Device: {DEVICE}")
print("\n⚠️  Memory optimizations applied:")
print(f"  - Reduced BATCH_SIZE to {BATCH_SIZE} (from 1024)")
print(f"  - Reduced K_MAX to {K_MAX} (from 10)")
print(f"  - Reduced DATASET_SIZE to {DATASET_SIZE} (from 10000)")
print("  - Added memory cleanup in training loop")
print("\n⚡ Performance optimizations enabled:")
print("  - torch.compile() for JIT compilation (2-3x speedup)")
print("  - Mixed precision training (AMP) (1.5-2x speedup)")
print("  - Fused AdamW optimizer")
print("  - Optimized compute_loss() (removed unnecessary clones/ops)")
print("  - Expected combined speedup: 3-6x faster training")
print("\n💡 If you still get OOM errors, restart the kernel first!")


# In[11]:


# ============================================================================
# LOAD EMBEDDING MODEL (from llm_on_sudoku.ipynb)
# ============================================================================
# We need the Sudoku2Vec class definition to load the trained model


class PositionalEncoding(nn.Module):
    """Positional encoding on unit circle for 9x9 Sudoku grid"""
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        # Create a grid of positions (0-8 for both x and y)
        x_coords = torch.arange(0, 9).unsqueeze(0).repeat(9, 1)
        y_coords = torch.arange(0, 9).unsqueeze(1).repeat(1, 9)

        # Convert grid positions to linear indices (0-80)
        linear_indices = y_coords * 9 + x_coords  # shape: (9, 9)

        # Convert linear indices to angles on unit circle
        angles = 2 * math.pi * linear_indices / 81  # shape: (9, 9)

        # Compute x, y coordinates on unit circle
        x_circle = torch.cos(angles)
        y_circle = torch.sin(angles)

        # Stack and add batch dimension
        pos_encoding = torch.stack([x_circle, y_circle], dim=-1).unsqueeze(0)  # shape: (1, 9, 9, 2)
        self.register_buffer('pos_encoding', pos_encoding)

    def get_embedding_for_position(self, pos):
        # input (batch, 2) where pos[:, 0] is x and pos[:, 1] is y
        linear_indices = pos[:, 1] * 9 + pos[:, 0]  # shape: (batch,)
        angles = 2 * math.pi * linear_indices / 81  # shape: (batch,)
        x_circle = torch.cos(angles).unsqueeze(1)  # shape: (batch, 1)
        y_circle = torch.sin(angles).unsqueeze(1)  # shape: (batch, 1)
        return torch.cat([x_circle, y_circle], dim=1)  # shape: (batch, 2)

    def forward(self, x):
        # x is a (batch, 9, 9, embedding_dim) grid
        # output (batch, 9, 9, embedding_dim + 2) grid by adding pos_encoding to x
        batch_size = x.shape[0]
        pos_expanded = self.pos_encoding.repeat(batch_size, 1, 1, 1)
        return torch.cat([x, pos_expanded], dim=-1)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # seperate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3) # [batch, head, seqlen, dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0,2,1,3) # [batch, seqlen, head, dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values) # [batch, seq_length, 81]

        if return_attention:
            return o, attention
        else:
            return o

class Sudoku2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, attention_dim=ATTENTION_DIM, num_heads=N_HEADS, device='cpu'):
        super(Sudoku2Vec, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.pe = PositionalEncoding()
        self.embed = nn.Embedding(vocab_size, embedding_dim) # this will provide the key queries and values
        self.total_dim = self.embedding_dim + 2

        self.mha = MultiheadAttention(
            input_dim=self.total_dim,
            embed_dim=attention_dim,
            num_heads=num_heads
        )

        # Move model to device
        self.to(device)

    def get_embeddings(self):
        """
        Returns the learned token embeddings.

        Returns:
            torch.Tensor: Embedding weight matrix of shape [vocab_size, embedding_dim]
        """
        return self.embed.weight.detach()

    def get_embedding_for_token(self, token):
        """
        Get the embedding vector for a specific token.

        Args:
            token: Integer token ID or tensor of token IDs

        Returns:
            torch.Tensor: Embedding vector(s) for the given token(s)
        """
        if isinstance(token, int):
            token = torch.tensor([token], device=self.device)
        elif not isinstance(token, torch.Tensor):
            token = torch.tensor(token, device=self.device)
        return self.embed(token).detach()

    def save_model(self, filepath):
        """
        Save the model in a portable format that can be easily loaded.
        This saves the model architecture and weights in a single file.

        Args:
            filepath: Path where to save the model (should end with .pt or .pth)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.embed.num_embeddings,
                'embedding_dim': self.embedding_dim,
                'attention_dim': self.mha.embed_dim,
                'num_heads': self.num_heads,
            },
            'model_class': 'Sudoku2Vec',
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """
        Load a saved Sudoku2Vec model from file.

        Args:
            filepath: Path to the saved model file
            device: Device to load the model on ('cpu', 'cuda', 'mps')

        Returns:
            Sudoku2Vec: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Extract configuration
        config = checkpoint['model_config']

        # Create model instance
        model = cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            attention_dim=config['attention_dim'],
            num_heads=config['num_heads'],
            device=device
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode by default

        print(f"Model loaded from {filepath}")
        print(f"Configuration: vocab_size={config['vocab_size']}, "
              f"embedding_dim={config['embedding_dim']}, "
              f"attention_dim={config['attention_dim']}, "
              f"num_heads={config['num_heads']}")

        return model

    def forward(self, target, position, sudoku_grid, mask=True):
        # target - the token in the target blank space we try to predict shape [batch] i.e [0, 3, 3, 5, 1, ...]
        # position - the (x, y) position of the target shape [batch, 2] - [[1, 1], [0, 3], [7,7], ...]
        # sudoku_grid - the sudoku grid for the problem with target we want to predict shape [batch, 9, 9]
        batch_size = target.shape[0]

        target_token_embeddings = self.embed(target) # shape [batch, embedding_dim]
        target_position_vectors = self.pe.get_embedding_for_position(position) # [batch, 2]
        target_token_with_position = torch.cat([target_token_embeddings, target_position_vectors], dim=-1)  # shape [batch, total_dim]

        # mask the target in the grid
        sudoku_grid_masked = sudoku_grid
        if mask:
            batch_indices = torch.arange(sudoku_grid.shape[0], device=self.device)
            sudoku_grid_masked = sudoku_grid.clone()
            sudoku_grid_masked[batch_indices, position[:, 1], position[:, 0]] = 0 # 0 is a mask token aka blank

        masked_sudoku_grid_embeddings = self.embed(sudoku_grid_masked)
        masked_sudoku_grid_with_position = self.pe(masked_sudoku_grid_embeddings) # shape [batch, 9, 9, total_dim]
        # Reshape grid to sequence: [batch, 81, total_dim]
        masked_grid_seq = masked_sudoku_grid_with_position.view(batch_size, 81, self.total_dim)

        grid_seq_embeddings = self.embed(sudoku_grid)
        grid_seq_embeddings = grid_seq_embeddings.view(batch_size, 81, self.embedding_dim) 

        # Query from target token: [batch, 1, total_dim]
        # query = target_token_with_position.unsqueeze(1)

        output, attention = self.mha(masked_grid_seq, return_attention=True)
        # output is shape [batch, 81, total_dim]

        return output, attention, target_token_with_position, grid_seq_embeddings



# In[12]:


class SudokuDiffusionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for pre-generated diffusion sequences.

    Each item is a diffusion sequence of shape (82, 9, 9) where:
    - Index 0: completely masked grid (all zeros)
    - Index 81: completely solved grid
    - Indices 1-80: intermediate states with progressively more cells revealed
    """
    def __init__(self, sequences):
        """
        Args:
            sequences: Tensor of shape (dataset_size, 82, 9, 9) containing diffusion sequences
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SudokuDiffusionModel(nn.Module):
    """
    Diffusion model for Sudoku puzzles inspired by DDPM.

    Forward process: Progressively mask cells from a complete sudoku (T=81) to empty grid (T=0)
    Reverse process: Learn to predict which cells to reveal to go from T to T+1

    The model learns to reverse the masking process, predicting which cell should be revealed
    at each timestep given the current partially revealed grid.
    """
    def __init__(self, hidden_dim=256, num_layers=6, kernel_size=3, num_groups=8, 
                 embedding_layer=None, device='cuda'):
        super().__init__()
        self.device = device
        self.num_timesteps = 81  # 81 cells in a sudoku grid
        self.embedding_layer = embedding_layer

        # Determine input channels based on whether we use embeddings
        if embedding_layer is not None:
            # Using learned embeddings: embedding_dim channels
            input_channels = embedding_layer.embedding_dim
            self.use_embeddings = True
        else:
            # Using simple normalization: 1 channel
            input_channels = 1
            self.use_embeddings = False

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Convolutional layers for processing the sudoku grid
        self.conv_in = nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])

        # Output: dual heads for position and value prediction
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)

        # Position head: which cell to reveal (81 possibilities)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim * 9 * 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 81)  # Logits for 81 cells
        )

        # Value head: what value to place (10 classes: 0-9)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 9 * 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 10)  # Logits for 10 classes
        )

    def forward(self, x, t):
        """
        Predict which cell should be revealed next and what value to place.

        Args:
            x: (batch, 9, 9) sudoku grids at timestep t (0 = masked, 1-9 = revealed)
            t: (batch,) timesteps (0 to 80)

        Returns:
            position_logits: (batch, 81) logits for which cell should be revealed next
            value_logits: (batch, 10) logits for what value (0-9) to place
        """
        batch_size = x.shape[0]

        # Process input: either use embeddings or simple normalization
        if self.use_embeddings:
            # Use learned embeddings: (batch, 9, 9) -> (batch, 9, 9, embedding_dim)
            x_embedded = self.embedding_layer(x.long())  # (batch, 9, 9, embedding_dim)
            x_norm = x_embedded.permute(0, 3, 1, 2)  # (batch, embedding_dim, 9, 9)
        else:
            # Simple normalization to [-1, 1] range
            x_norm = (x / 4.5) - 1.0
            x_norm = x_norm.unsqueeze(1)  # (batch, 1, 9, 9)

        # Time embedding
        t_norm = t.float().unsqueeze(1) / self.num_timesteps  # (batch, 1)
        t_emb = self.time_embed(t_norm)  # (batch, hidden_dim)

        # Process through conv layers
        h = self.conv_in(x_norm)  # (batch, hidden_dim, 9, 9)

        # Add time embedding to spatial features
        t_emb_spatial = t_emb.view(batch_size, -1, 1, 1).expand(-1, -1, 9, 9)
        h = h + t_emb_spatial

        # Apply conv blocks with residual connections
        for block in self.conv_blocks:
            h = h + block(h)

        # Output processing
        h = self.conv_out(h)  # (batch, hidden_dim, 9, 9)
        h_flat = h.reshape(batch_size, -1)  # (batch, hidden_dim * 81)

        # Dual predictions
        position_logits = self.position_head(h_flat)  # (batch, 81)
        value_logits = self.value_head(h_flat)  # (batch, 10)

        return position_logits, value_logits

    def compute_loss(self, sequences, k_max=10):
        """
        Compute the diffusion loss for training using K-step iterative prediction.

        The forward diffusion process (from sudoku.py) goes from empty (T=0) to complete (T=81).
        We learn to predict K steps ahead by iteratively applying the model.

        Args:
            sequences: (batch, 82, 9, 9) diffusion sequences where:
                      - sequences[:, 0] is completely masked (all zeros)
                      - sequences[:, 81] is completely solved
            k_max: Maximum number of forward steps for multi-step prediction

        Returns:
            loss: scalar combined loss (position + value)
            accuracy: prediction accuracy for logging
        """
        batch_size = sequences.shape[0]

        # Sample random starting timestep B from [0, 81-k_max]
        max_start = max(1, self.num_timesteps - k_max)
        B = torch.randint(0, max_start, (batch_size,), device=self.device)

        # Sample random K from [1, k_max]
        K = torch.randint(1, k_max + 1, (batch_size,), device=self.device)

        # Get starting grids at timestep B (remove unnecessary clone)
        x_current = sequences[torch.arange(batch_size), B].float()  # (batch, 9, 9)

        # Track losses and accuracies across all K steps
        total_position_loss = 0.0
        total_value_loss = 0.0
        total_position_acc = 0.0
        total_value_acc = 0.0

        # Pre-allocate batch_indices outside loop
        batch_indices = torch.arange(batch_size, device=self.device)

        # Iteratively predict K steps
        for step in range(k_max):
            # Current timestep for each batch element
            t_current = B + step

            # Only compute loss for elements where step < K[i] and t_current < 81
            active_mask = (step < K) & (t_current < self.num_timesteps)

            if not active_mask.any():
                break

            # Get target grid at next timestep
            t_next = torch.clamp(t_current + 1, max=self.num_timesteps)
            x_target = sequences[batch_indices, t_next].float()  # (batch, 9, 9)

            # Find which cell was revealed (difference between current and target)
            diff = (x_target != x_current).view(batch_size, 81)  # (batch, 81)

            # Check if there's actually a difference (cell was revealed)
            has_diff = diff.any(dim=1)  # (batch,)
            active_mask = active_mask & has_diff  # Only process if there's a change

            if not active_mask.any():
                break

            target_position = diff.float().argmax(dim=1)  # (batch,)

            # Get target values at revealed positions (vectorized for efficiency)
            rows = target_position // 9
            cols = target_position % 9
            target_values = x_target[batch_indices, rows, cols].long()

            # Predict position and value
            position_logits, value_logits = self.forward(x_current, t_current)  # (batch, 81), (batch, 10)

            # Mask out already revealed cells in position prediction (in-place operation)
            already_revealed = (x_current.view(batch_size, 81) != 0)  # (batch, 81)
            position_logits.masked_fill_(already_revealed, float('-inf'))

            # Compute losses only for active batch elements
            if active_mask.any():
                position_loss = F.cross_entropy(position_logits[active_mask], target_position[active_mask], reduction='sum')
                value_loss = F.cross_entropy(value_logits[active_mask], target_values[active_mask], reduction='sum')

                # Accumulate losses (keep in computation graph for backprop)
                total_position_loss = total_position_loss + position_loss
                total_value_loss = total_value_loss + value_loss

                # Compute accuracy for logging
                with torch.no_grad():
                    pred_position = position_logits[active_mask].argmax(dim=1)
                    pred_value = value_logits[active_mask].argmax(dim=1)
                    total_position_acc += (pred_position == target_position[active_mask]).float().sum()
                    total_value_acc += (pred_value == target_values[active_mask]).float().sum()

            # Update x_current with ground truth for next iteration (remove unnecessary clone)
            x_current = x_target

        # Average losses over all active predictions
        num_predictions = K.float().sum()

        # Avoid division by zero
        if num_predictions == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        total_position_loss = total_position_loss / num_predictions
        total_value_loss = total_value_loss / num_predictions

        # Combine losses
        loss = total_position_loss + total_value_loss

        # Average accuracy
        accuracy = (total_position_acc + total_value_acc) / (2 * num_predictions)

        return loss, accuracy

    @torch.no_grad()
    def sample(self, batch_size=1, return_trajectory=False):
        """
        Generate sudoku puzzles by running the reverse diffusion process.
        Start from empty grid (T=0) and progressively reveal cells to T=81.

        Args:
            batch_size: number of puzzles to generate
            return_trajectory: if True, return full trajectory of generation

        Returns:
            samples: (batch_size, 9, 9) generated sudoku grids
            trajectory: (batch_size, 82, 9, 9) if return_trajectory=True
        """
        # Start from completely masked grid (T=0)
        x = torch.zeros(batch_size, 9, 9, device=self.device)

        if return_trajectory:
            trajectory = torch.zeros(batch_size, 82, 9, 9, device=self.device)
            trajectory[:, 0] = x

        # Progressively reveal cells using model predictions
        for t in tqdm(range(self.num_timesteps), desc='Sampling'):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict position and value
            position_logits, value_logits = self.forward(x, t_batch)

            # Mask out already revealed cells
            already_revealed = (x.view(batch_size, 81) != 0)
            position_logits = position_logits.masked_fill(already_revealed, float('-inf'))

            # Sample or take argmax for position
            position_probs = F.softmax(position_logits, dim=-1)
            cell_idx = torch.multinomial(position_probs, 1).squeeze(-1)  # (batch,)

            # Take argmax for value (deterministic)
            value_probs = F.softmax(value_logits, dim=-1)
            values = torch.argmax(value_probs, dim=-1)  # (batch,)

            # Update grid with predicted values
            for b in range(batch_size):
                idx = cell_idx[b].item()
                row = idx // 9
                col = idx % 9
                x[b, row, col] = values[b]

            if return_trajectory:
                trajectory[:, t + 1] = x

        if return_trajectory:
            return x.long(), trajectory.long()
        return x.long()


def train_sudoku_diffusion(model, dataset, num_epochs=1000, batch_size=32, lr=1e-4, 
                           weight_decay=1e-4, grad_clip_max_norm=1.0,
                           log_interval=10, eval_interval=100, k_max=10, device='cuda'):
    """
    Train the sudoku diffusion model with proper logging and performance optimizations.

    Args:
        model: SudokuDiffusionModel instance
        dataset: SudokuDiffusionDataset instance with pre-generated sequences
        num_epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        weight_decay: weight decay for AdamW optimizer
        grad_clip_max_norm: max norm for gradient clipping
        log_interval: log metrics every N epochs
        eval_interval: evaluate and sample every N epochs
        k_max: maximum number of forward steps for multi-step prediction
        device: device to train on
    """
    # Create DataLoader for batching and shuffling with optimizations
    # Note: pin_memory should be False when data is already on GPU
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for GPU tensors
        pin_memory=False,  # Data is already on GPU, no need to pin
        persistent_workers=False  # No workers, so this doesn't apply
    )

    # Use fused optimizer for faster updates (requires CUDA)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        fused=True if device == 'cuda' else False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # Training metrics
    train_losses = []
    train_accuracies = []
    epoch_losses = []
    epoch_accuracies = []

    model.train()
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, Batches per epoch: {len(dataloader)}")
    print(f"Learning rate: {lr}")
    print(f"Using mixed precision: {scaler is not None}")
    print("-" * 60)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        # Iterate through batches in the dataset
        for batch_sequences in dataloader:
            # Mixed precision training
            if scaler is not None:
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    loss, accuracy = model.compute_loss(batch_sequences, k_max=k_max)

                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training (CPU or non-CUDA)
                loss, accuracy = model.compute_loss(batch_sequences, k_max=k_max)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            num_batches += 1

        # Average metrics over all batches in the epoch
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_acc = epoch_acc / num_batches

        # Record metrics
        epoch_losses.append(avg_epoch_loss)
        epoch_accuracies.append(avg_epoch_acc)

        # Update learning rate scheduler
        scheduler.step()

        # Logging
        if (epoch + 1) % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1:4d}/{num_epochs} | "
                  f"Loss: {avg_epoch_loss:.4f} | "
                  f"Acc: {avg_epoch_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Evaluation and sampling
        if (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at epoch {epoch + 1}")
            print(f"{'='*60}")

            model.eval()
            with torch.no_grad():
                # Generate a sample
                sample = model.sample(batch_size=1)
                print("\nGenerated Sudoku:")
                print(sample[0].cpu().numpy())

                # Check if valid
                sudoku_obj = Sudoku(sample[0].cpu().numpy(), backend='numpy')
                is_valid = sudoku_obj.is_valid()
                print(f"\nIs valid: {is_valid}")

            model.train()
            print(f"{'='*60}\n")

    print("\nTraining completed!")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epoch_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    ax2.plot(epoch_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return model, epoch_losses, epoch_accuracies


# In[13]:


# Check GPU information and clear memory

if torch.cuda.is_available():
    print("GPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print(f"\nInitial GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Aggressively clear GPU memory
    print("\n⚠️  Clearing GPU memory...")

    # Delete all variables in the current namespace that might hold GPU tensors
    if 'model' in dir():
        del model
    if 'generator' in dir():
        del generator
    if 'sequences' in dir():
        del sequences

    # Force garbage collection
    gc.collect()

    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    print(f"\nAfter clearing:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")

    # If still not enough memory, suggest kernel restart
    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
    if free_memory < 1.0:
        print("\n⚠️  WARNING: Very little GPU memory available!")
        print("   Consider: Kernel -> Restart Kernel to fully clear GPU memory")
else:
    print("CUDA not available, will use CPU")


# In[14]:


# ============================================================================
# CELL 5: LOAD EMBEDDING MODEL
# ============================================================================
# Run this cell once per session to load the pre-trained Sudoku2Vec embeddings.
# This is fast (~1 second) and only needs to run once.

print(f"Using device: {DEVICE}")

# Clear GPU memory if using CUDA
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    print(f"\nGPU Memory before setup:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")

# Load embedding model if configured
embedding_layer = None
if USE_LEARNED_EMBEDDINGS:
    print(f"\n{'='*60}")
    print("Loading learned embeddings from LLM model...")
    print(f"{'='*60}")
    try:
        sudoku2vec_model = Sudoku2Vec.load_model(EMBEDDING_MODEL_PATH, device=DEVICE)
        embedding_layer = sudoku2vec_model.embed
        print(f"✓ Successfully loaded embedding layer with {embedding_layer.num_embeddings} tokens")
        print(f"  Embedding dimension: {embedding_layer.embedding_dim}")
    except FileNotFoundError:
        print(f"⚠️  WARNING: Embedding model not found at {EMBEDDING_MODEL_PATH}")
        print("   Falling back to simple normalization")
        USE_LEARNED_EMBEDDINGS = False
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load embedding model: {e}")
        print("   Falling back to simple normalization")
        USE_LEARNED_EMBEDDINGS = False
else:
    print("\nUsing simple normalization (no learned embeddings)")

print("\n✓ Embedding layer ready!")


# In[15]:


# ============================================================================
# CELL 6: GENERATE TRAINING DATASET
# ============================================================================
# Run this cell ONCE to generate the training dataset.
# This is SLOW (~4 minutes for 20k sequences) but you only need to run it once!
# 
# After running this cell, you can:
# - Rerun Cell 7 to try different model architectures
# - Rerun Cell 8 to try different training hyperparameters
# - All without regenerating this expensive dataset!

print(f"\n{'='*60}")
print(f"Pre-generating training dataset...")
print(f"{'='*60}")

# Initialize generator
generator = SudokuGenerator(backend='torch', device=DEVICE)

# Generate diffusion sequences
print(f"Generating {DATASET_SIZE} diffusion sequences...")
import time
start_time = time.time()

sequences = generator.generate_diffusion_sequence(size=DATASET_SIZE)

generation_time = time.time() - start_time
print(f"✓ Dataset generated in {generation_time:.2f} seconds ({generation_time/DATASET_SIZE*1000:.2f} ms per sequence)")

# Report dataset statistics
sequence_shape = sequences.shape
memory_mb = sequences.element_size() * sequences.nelement() / (1024 ** 2)
print(f"\nDataset statistics:")
print(f"  Shape: {sequence_shape}")
print(f"  Memory: {memory_mb:.2f} MB")
print(f"  Device: {sequences.device}")

# Create PyTorch Dataset
train_dataset = SudokuDiffusionDataset(sequences)
print(f"✓ Created SudokuDiffusionDataset with {len(train_dataset)} sequences")
print(f"\n💡 Dataset is ready! You can now rerun Cells 7 & 8 with different hyperparameters.")


# In[16]:


# ============================================================================
# CELL 7: INITIALIZE DIFFUSION MODEL
# ============================================================================
# Run this cell to create and compile the diffusion model.
# This is FAST (~5 seconds) and you can rerun it to experiment with:
# - Different model sizes (HIDDEN_DIM, NUM_LAYERS)
# - Different architectures (KERNEL_SIZE, NUM_GROUPS)
# 
# The dataset from Cell 6 will be reused!

print(f"\n{'='*60}")
print("Initializing Diffusion Model...")
print(f"{'='*60}")

model = SudokuDiffusionModel(
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    kernel_size=KERNEL_SIZE,
    num_groups=NUM_GROUPS,
    embedding_layer=embedding_layer,
    device=DEVICE
).to(DEVICE)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model initialized successfully")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Using embeddings: {model.use_embeddings}")

# Compile model for faster training (PyTorch 2.0+)
if DEVICE == 'cuda':
    print(f"\n⚡ Compiling model with torch.compile for faster training...")
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print(f"✓ Model compiled successfully")
    except Exception as e:
        print(f"⚠️  Could not compile model: {e}")
        print(f"   Continuing without compilation")

if DEVICE == 'cuda':
    print(f"\nGPU Memory after model loading:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print(f"\n✓ Model is ready for training!")


# In[ ]:


# ============================================================================
# CELL 8: TRAIN THE MODEL
# ============================================================================
# Run this cell to train the model with the current hyperparameters.
# You can rerun this cell to experiment with different:
# - Learning rates (LEARNING_RATE)
# - Batch sizes (BATCH_SIZE)
# - Training strategies (K_MAX, WEIGHT_DECAY, GRAD_CLIP_MAX_NORM)
# - Logging intervals (LOG_INTERVAL, EVAL_INTERVAL)
# 
# The model from Cell 7 and dataset from Cell 6 will be used!

print(f"\n{'='*60}")
print("Starting Training...")
print(f"{'='*60}")

model, losses, accuracies = train_sudoku_diffusion(
    model=model,
    dataset=train_dataset,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
    log_interval=LOG_INTERVAL,
    eval_interval=EVAL_INTERVAL,
    k_max=K_MAX,
    device=DEVICE
)


# In[ ]:




