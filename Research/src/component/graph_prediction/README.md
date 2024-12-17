# GCN for Solubility Prediction with ESOL Dataset

This project uses a **Graph Convolutional Network (GCN)** to predict the solubility of molecules using the **ESOL dataset** from MoleculeNet.

## Dataset

- **ESOL dataset**: Contains molecular graphs where nodes represent atoms and edges represent bonds.
- **Target**: The target variable is the log solubility of molecules (regression task).

## Loss Function and Optimizer

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam Optimizer

## Installation

Ensure the following libraries are installed before running the project:

- `torch` (PyTorch)
- `torch-geometric` (PyTorch Geometric)
- `rdkit` (RDKit)
- Additional dependencies for PyTorch Geometric:
  - `torch-scatter`
  - `torch-sparse`
  - `torch-cluster`
  - `torch-spline-conv`
