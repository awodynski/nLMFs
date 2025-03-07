# Neural Network Training for TURBOMOLE-Compatible Local Mixing Functions (n-LMF)

This repository provides a Python script used to train neural networks that serve as local mixing functions (n-LMF) within Density Functional Approximations (DFAs). The script is specifically designed to produce neural-network-derived parameters compatible with the TURBOMOLE quantum chemistry package.

## Project Context

The code presented here was directly utilized in training the neural network local mixing functions (n-LMF) employed in recently developed density functional approximations, including:

- **LH24n** and **LH24n-B95** functionals, as presented in our published work.
- Preliminary studies on neural-network-based **local double hybrid functionals**, investigating novel DFA architectures.

## Overview of the Script

The Python script incorporates the following functionalities:

- **Data Loading and Preprocessing:**
  - Uses custom-defined functions (`collect_data`) to ingest training data from datasets (e.g., W417, BH76), containing computed molecular features such as electron densities, density gradients, and kinetic energy densities (tau).

- **Neural Network Architecture:**
  - Configurable Multi-Layer Perceptron (MLP) with customizable layers, hidden units, and activation functions (e.g., GELU, sigmoid).
  - Incorporates an input transformation method (`SignedLogTransform`) tailored for chemical descriptor data.

- **Training Configuration:**
  - Epochs: 6000
  - Learning rate: 0.01
  - Regularization: Configurable L2 regularization (default set to 0.0)

- **Output and Integration:**
  - Custom callbacks for monitoring training progress.
  - Saving model parameters in a TURBOMOLE-friendly format, facilitating direct integration into quantum chemical calculations.

## Usage Instructions

Run the training script directly from the command line:

```bash
python calc.py
```

This process includes data loading, model training, and exporting weights suitable for integration into TURBOMOLE workflows.

## Citation

If you utilize this script in your research, please cite our relevant publications:

- **LH24n and LH24n-B95 Development Paper:**
J. Chem. Theory Comput. 2025, 21, 2, 762–775
https://doi.org/10.1021/acs.jctc.4c01503

## License

MIT License

## Acknowledgements

Part of this codebase was edited with the assistance of ChatGPT, leveraged to enhance both clarity and consistency.

## Training data

Training data for local double hybrids are available in 10.5281/zenodo.14991207

