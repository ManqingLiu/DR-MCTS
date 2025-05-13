# Doubly Robust Monte Carlo Tree Search (DR-MCTS)

This repository contains the implementation of **Doubly Robust Monte Carlo Tree Search (DR-MCTS)**, a novel algorithm that integrates Doubly Robust (DR) off-policy estimation into Monte Carlo Tree Search (MCTS) to enhance sample efficiency and decision quality in complex environments.

## Overview

DR-MCTS is designed to improve upon standard MCTS by leveraging doubly robust estimation to achieve:
- Higher sample efficiency
- Better decision quality in complex environments
- Superior performance when using smaller language models
- Theoretical guarantees of unbiasedness and variance reduction

The algorithm has been tested on both simple (Tic-Tac-Toe) and complex (SmallGo, VirtualHome) environments, demonstrating consistent improvements over traditional MCTS approaches.

## Key Features

- **Multiple MCTS Variants**: Implements standard MCTS, IS-MCTS (Importance Sampling), and DR-MCTS
- **Game Support**: Includes implementations for Tic-Tac-Toe and SmallGo (5x5 Go)
- **Parallel Simulation**: Supports multi-threaded MCTS simulations for improved performance
- **Comprehensive Experiments**: Includes scripts for reproducing paper results
- **Visualization Tools**: Generates plots and figures for performance analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MCTS_DR
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirement.txt
```

## Project Structure

```
MCTS_DR/
├── src/
│   ├── core/
│   │   ├── Games.py          # Game implementations (TicTacToe, SmallGo)
│   │   ├── MCTS_class.py     # MCTS variants (naive, IS, DR)
│   │   └── MCTS_gridworld.py # GridWorld implementation
│   ├── utils/
│   │   ├── game_helpers.py   # Game playing utilities
│   │   └── seed.py          # Random seed management
│   └── visualizations/
│       ├── plot_helpers.py   # Plotting utilities
│       └── visualize.py      # Visualization scripts
├── experiments/
│   ├── experiments.py        # Main experiment runner
│   ├── figure1.py           # Generate Figure 1 (SmallGo game example)
│   ├── figure2.py           # Generate Figure 2 (DR-MCTS performance)
│   └── figure4.py           # Generate Figure 4 (IS-MCTS performance)
├── DR-MCTS_paper.pdf        # Research paper
├── requirement.txt          # Python dependencies
└── README.md               # This file
```

## Usage

### Running Experiments

To reproduce the experimental results from the paper:

```python
# Run comprehensive experiments
python experiments/experiments.py

# Generate specific figures
python experiments/figure1.py  # SmallGo game visualization
python experiments/figure2.py  # DR-MCTS vs MCTS comparison
python experiments/figure4.py  # IS-MCTS vs MCTS comparison
```

### Basic Example

```python
from src.core.Games import TicTacToe, SmallGo
from src.core.MCTS_class import MCTS_naive, MCTS_DR

# Create a game instance
game = TicTacToe()  # or SmallGo(board_size=5)

# Initialize MCTS algorithms
mcts_naive = MCTS_naive(exploration_weight=1.4)
mcts_dr = MCTS_DR(
    exploration_weight=1.4,
    alpha=0.0,
    beta_base=0.5,
    lambda_param=0.01
)

# Run MCTS search
num_simulations = 100
action, value = mcts_dr.mcts_search(game, num_simulations)

# Make the move
game.make_move(action)
```

### Running on HPC Clusters

For running on HPC clusters (e.g., SLURM), use the provided job script:

```bash
sbatch myjob.sh
```

## Algorithms Implemented

1. **MCTS_naive**: Standard Monte Carlo Tree Search with PUCT selection
2. **MCTS_IS**: MCTS with Importance Sampling estimation
3. **MCTS_DR**: MCTS with Doubly Robust estimation (our contribution)

### Key Parameters

- `exploration_weight`: Controls exploration-exploitation trade-off (default: 1.4)
- `alpha`: Mixing parameter for target policy (DR-MCTS specific)
- `beta_base`: Base value for adaptive beta calculation
- `lambda_param`: Decay rate for adaptive beta
- `num_simulations`: Number of MCTS simulations per move

## Experimental Results

The implementation demonstrates:

1. **Tic-Tac-Toe**: DR-MCTS achieves win rates of 63-88% against standard MCTS
2. **SmallGo**: Superior performance with consistent improvements across different configurations
3. **Sample Efficiency**: DR-MCTS shows better scaling with the number of rollouts

Results are saved in `experiments/results/` directory.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anonymous2024drmcts,
  title={Doubly Robust Monte Carlo Tree Search},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the research paper "Doubly Robust Monte Carlo Tree Search" available in this repository.

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the authors at the email provided in the paper.