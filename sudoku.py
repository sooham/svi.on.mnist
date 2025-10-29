import numpy as np
import torch
import matplotlib.pyplot as plt
import random

class Sudoku:
    def __init__(self, grid, solution=None, number_tokens=' 123456789', backend='numpy'):
        """
        grid: np.ndarray of numbers from 0 to 9
        solution: similart to grid, but the solution grid
        number_tokens: array of numbers representing 1 to 9, the 0th element is the blank token
        """
        self.grid = grid
        self.solution = solution
        self.backend = backend
        self.num_to_token = {i: v for i, v in enumerate(number_tokens)}
        self.token_to_num = {v: i for v, i in self.num_to_token.items()}
        self.num_to_str = {i: str(v) for i, v in self.num_to_token.items()}
    
    def __str__(self):
        """Return a string representation of the sudoku grid."""
        lines = []
        for i in range(9):
            if i % 3 == 0 and i != 0:
                lines.append("------+-------+------")
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                val = self.grid[i,j]
                if self.backend == 'torch':
                    val = val.item()

                row_str += self.num_to_str[val] + " "
            lines.append(row_str.rstrip())
        return f"Backend: {self.backend}\n" + "\n".join(lines)
    
    def __repr__(self):
        return str(self)
    
    def plot(self):
        """Plot the sudoku grid using matplotlib."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw the grid
        for i in range(10):
            linewidth = 2 if i % 3 == 0 else 0.5
            ax.plot([0, 9], [i, i], 'k-', linewidth=linewidth)
            ax.plot([i, i], [0, 9], 'k-', linewidth=linewidth)
        
        # Add numbers to the grid
        for i in range(9):
            for j in range(9):
                if self.grid[i, j] != 0:
                    val = self.grid[i,j]
                    if self.backend == 'torch':
                        val = val.item()
                    ax.text(j + 0.5, 8.5 - i, self.num_to_str[val], ha='center', va='center', fontsize=20)
        
        # Set axis properties
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _is_unit_valid(self, unit):
        """Check if a unit (row, column, or box) has no duplicate non-blank values."""
        # Filter out blanks
        filled = [val for val in unit if val != 0]
        # Check for duplicates
        return len(filled) == len(set(filled))
    
    def is_valid(self):
        """Check if the sudoku grid is valid."""
        # Check all rows
        for i in range(9):
            row = self.grid[i, :]
            if not self._is_unit_valid(row):
                return False
        
        # Check all columns
        for j in range(9):
            col = self.grid[:, j]
            if not self._is_unit_valid(col):
                return False
        
        # Check all 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = self.grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
                if not self._is_unit_valid(box):
                    return False
        
        return True
    

"""
Question: how many constraints will it take to make a sudoku uniquely identifiable?
"""

class SudokuGenerator:
    def __init__(self, backend='numpy', bit_width='full', device='cpu'):
        """
        Initialize a SudokuGenerator with specified backend and bit width.
        
        Args:
            backend: 'numpy' or 'torch' - which library to use for array operations
            bit_width: 'full' or '4bit' - whether to use full precision or 4-bit representation
            device: 'cpu', 'cuda', or 'mps' - device to use for torch tensors (only applicable when backend='torch')
        """
        self.backend = backend
        self.bit_width = bit_width
        self.device = torch.device(device) if backend == 'torch' else None
    
    def generate_valid_sudoku(self) -> Sudoku:
        """
        Generate a valid and complete sudoku Grid that is random.
        """
        
        # Start with an empty grid
        if self.backend == 'torch':
            grid = torch.zeros((9, 9), dtype=torch.int8 if self.bit_width == '4bit' else torch.int32)
            grid = grid.numpy()  # Convert to numpy for processing
        else:
            grid = np.zeros((9, 9), dtype=np.int8 if self.bit_width == '4bit' else np.int32)
        
        def is_valid_placement(grid, row, col, num):
            """Check if placing num at (row, col) is valid."""
            # Check row
            if num in grid[row, :]:
                return False
            
            # Check column
            if num in grid[:, col]:
                return False
            
            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            if num in grid[box_row:box_row+3, box_col:box_col+3]:
                return False
            
            return True
        
        def fill_grid(grid):
            """Recursively fill the grid using backtracking."""
            for i in range(9):
                for j in range(9):
                    if grid[i, j] == 0:
                        # Try numbers in random order
                        numbers = list(range(1, 10))
                        random.shuffle(numbers)
                        
                        for num in numbers:
                            if is_valid_placement(grid, i, j, num):
                                grid[i, j] = num
                                
                                if fill_grid(grid):
                                    return True
                                
                                # Backtrack
                                grid[i, j] = 0
                        
                        return False
            return True
        
        fill_grid(grid)
        
        # Convert back to torch if needed
        if self.backend == 'torch':
            grid = torch.tensor(grid, dtype=torch.int8 if self.bit_width == '4bit' else torch.int32, device=self.device)
            
        return Sudoku(grid, backend=self.backend)
    
    def generate_puzzle(self, difficulty='medium') -> Sudoku:
        """
        Generate a sudoku puzzle by removing cells from a complete grid.
        
        Args:
            difficulty: 'easy' (35-40 clues), 'medium' (30-35 clues), 'hard' (25-30 clues)
        
        Returns:
            A Sudoku puzzle with some cells removed
        """
        # First generate a complete valid sudoku
        complete_sudoku = self.generate_valid_sudoku()
        
        # Copy the grid appropriately based on backend
        if self.backend == 'torch':
            puzzle_grid = complete_sudoku.grid.clone()
        else:
            puzzle_grid = complete_sudoku.grid.copy()
        
        # Determine number of cells to remove based on difficulty
        if difficulty == 'easy':
            cells_to_remove = random.randint(41, 46)  # 35-40 clues remaining
        elif difficulty == 'hard':
            cells_to_remove = random.randint(51, 56)  # 25-30 clues remaining
        else:  # medium
            cells_to_remove = random.randint(46, 51)  # 30-35 clues remaining
        
        # Get all cell positions
        all_positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(all_positions)
        
        # Remove cells
        removed = 0
        for row, col in all_positions:
            if removed >= cells_to_remove:
                break
            
            # Save the current value
            backup = puzzle_grid[row, col]
            puzzle_grid[row, col] = 0
            
            # Check if puzzle still has a unique solution (simplified: just remove it)
            # A more sophisticated approach would verify uniqueness
            removed += 1
        
        return Sudoku(puzzle_grid, solution=complete_sudoku.grid, backend=self.backend)

    def generate_batch(self, difficulty='medium', size=1):
        puzzles = []
        for i in range(size):
            puzzle = self.generate_puzzle(difficulty=difficulty)
            puzzles.append(puzzle.solution)
        
        if self.backend == 'torch':
            return torch.stack([torch.tensor(p, dtype=torch.int8 if self.bit_width == '4bit' else torch.int32, device=self.device) for p in puzzles])
        else:
            return np.stack(puzzles, dtype=np.int8 if self.bit_width == '4bit' else np.int32)

    def generate_target_context_pairs(self, size, k=1, shuffle=True):
        """
        Generate target-context pairs for training.
        
        Args:
            size: Number of puzzles to generate
            k: Number of target positions to sample per puzzle
            shuffle: Whether to shuffle the outputs along the first dimension
            
        Returns:
            targets: Sampled target values (size * k,)
            position: Sampled positions (size * k, 2)
            puzzles: Puzzle grids repeated k times (size * k, 9, 9)
            puzzle_batch: Original puzzle batch (size, 9, 9)
        """
        puzzle_batch = self.generate_batch(size=size)
        
        # Generate all possible positions once
        if self.backend == 'torch':
            x_coords, y_coords = torch.meshgrid(torch.arange(9, device=self.device), torch.arange(9, device=self.device), indexing='ij')
            all_positions = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)  # Shape: (81, 2)
            
            # Sample k positions for each puzzle without replacement
            sampled_indices = torch.stack([torch.randperm(81, device=self.device)[:k] for _ in range(size)])  # Shape: (size, k)
            
            # Get the sampled positions: (size, k, 2)
            position = all_positions[sampled_indices.flatten()].reshape(size * k, 2)
            
            # Get the sampled targets from the puzzles
            # Convert positions to linear indices for gathering
            linear_indices = sampled_indices.flatten()  # Shape: (size * k,)
            batch_indices = torch.arange(size, device=self.device).repeat_interleave(k)  # Shape: (size * k,)
            
            # Flatten puzzle_batch and gather targets
            flattened_puzzles = puzzle_batch.reshape(size, 81)  # Shape: (size, 81)
            targets = flattened_puzzles[batch_indices, linear_indices]  # Shape: (size * k,)
            
            # Repeat each puzzle k times: (size, 9, 9) -> (size * k, 9, 9)
            puzzles = puzzle_batch.repeat_interleave(k, dim=0)
            
            # Shuffle if requested
            if shuffle:
                shuffle_indices = torch.randperm(size * k, device=self.device)
                targets = targets[shuffle_indices]
                position = position[shuffle_indices]
                puzzles = puzzles[shuffle_indices]
        else:
            x_coords, y_coords = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
            all_positions = np.stack([y_coords.ravel(), x_coords.ravel()], axis=1)  # Shape: (81, 2)
            
            # Sample k positions for each puzzle without replacement
            sampled_indices = np.array([np.random.choice(81, size=k, replace=False) for _ in range(size)])  # Shape: (size, k)
            
            # Get the sampled positions: (size * k, 2)
            position = all_positions[sampled_indices.flatten()].reshape(size * k, 2)
            
            # Get the sampled targets from the puzzles
            linear_indices = sampled_indices.flatten()  # Shape: (size * k,)
            batch_indices = np.arange(size).repeat(k)  # Shape: (size * k,)
            
            # Flatten puzzle_batch and gather targets
            flattened_puzzles = puzzle_batch.reshape(size, 81)  # Shape: (size, 81)
            targets = flattened_puzzles[batch_indices, linear_indices]  # Shape: (size * k,)
            
            # Repeat each puzzle k times: (size, 9, 9) -> (size * k, 9, 9)
            puzzles = np.repeat(puzzle_batch, k, axis=0)
            
            # Shuffle if requested
            if shuffle:
                shuffle_indices = np.random.permutation(size * k)
                targets = targets[shuffle_indices]
                position = position[shuffle_indices]
                puzzles = puzzles[shuffle_indices]
        
        return targets, position, puzzles, puzzle_batch

