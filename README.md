# Uali

## Overview
Uali is an automated game-playing agent designed for Tetris. It uses a simple implementation of the AlphaZero algorithm with the Gumbel root action policy to learn Tetris from scratch.

## Features
- **Parallel Simulator**: Vectorized implementation of Tetris that runs on GPU for efficient simulation of multiple games simultaneously.
- **Learning-from-Scratch**: Implements AlphaZero's MCTS self-play learning algorithm to iteratively improve the policy over time.
- **Gumbel Root Action Policy**: Enhances exploration during self-play by adding noise to action probabilities.
- **Distributed Training**: Utilizes [Ray](https://www.ray.io/) to distribute self-play and training processes across multiple workers.

## Installation

### Prerequisites
- Python 3.9 or later
- Git

### Steps to Install

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vluk/uali2.git
   cd uali2
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To start training the model, execute the following command:

```bash
python muzero.py
```

## Acknowledgments

This project is in part based on prior implementations of the AlphaZero and MuZero algorithms, including:
- https://github.com/suragnair/alpha-zero-general
- https://github.com/werner-duvaud/muzero-general
- https://github.com/google-deepmind/mctx
- https://github.com/opendilab/LightZero