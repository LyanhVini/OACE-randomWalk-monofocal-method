# Optimized Assertiveness Cost Evaluation Method

## Overview

This repository contains the implementation of the **Optimized Assertiveness Cost Evaluation (OACE)** method, a multicriteria decision-making (MCDM) approach for selecting and optimizing deep learning models. The OACE method balances assertiveness (e.g., accuracy, precision) and computational cost (e.g., number of parameters, inference time) using a Random Walk algorithm to tune hyperparameters, such as the learning rate (\textit{lr}). The method was developed as part of a research project at PPGEE/UFPA, focusing on computational intelligence and adaptive methods for deep learning optimization, and was evaluated on the CIFAR-10 dataset.

## Installation

### Prerequisites
- Python 3.9+
- PyTorch (assumed based on typical deep learning setups; adjust if using TensorFlow)
- NumPy, Pandas, Matplotlib, Seaborn (for data analysis and visualization)
- Jupyter Notebook (for running `.ipynb` files)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/LyanhVini/OACE-randomWalk-monofocal-method.git
   cd OACE-randomWalk-monofocal-method
2. Install dependencies:
   pip install -r requirements.txt

## Usage

1. **Prepare the Dataset**:
- The method uses the CIFAR-10 dataset. Manually download the dataset and upload it to the `dataset/` directory.
- *Note*: The `dataset/` directory is excluded by `.gitignore` and must be created manually if it does not exist.
- Alternatively, the `datasets.py` script can assist in downloading CIFAR-10, placing it in `base/` if preferred.

2. **Run the Experiments**:
- Experiments were conducted across three distinct scenarios with different priorities: accuracy prioritization (*λ* = 0.75), balanced assertiveness and cost (*λ* = 0.5), and cost optimization (*λ* = 0.25). To replicate, run the `main.py` script or use the `OACE_RandomWalk_monofocal.ipynb` notebook in Jupyter for an interactive setup. Outputs, such as model checkpoints, will be saved in the `results/` directory.

3. **Analyze Outputs**:
- Outputs are saved in the `results/` directory. For a detailed analysis of the results, including visualizations such as box plots and convergence graphs, refer to the Google Colab notebook: [OACE Results Analysis]().

## Repository Structure

- `base/`: Directory for storing datasets downloaded by `datasets.py` (e.g., CIFAR-10).
- `dataset/`: Directory for manually uploaded datasets (excluded by `.gitignore`; create if needed).
- `results/`: Directory to store experiment outputs, including model checkpoints.
- `.gitignore`: Git ignore file for excluding temporary files and the `dataset/` directory.
- `LICENSE`: Apache 2.0 License file.
- `OACE_RandomWalk_monofocal.ipynb`: Jupyter notebook with the complete OACE method implementation and experiments.
- `README.md`: This file.
- `checkpoint.py`: Script for saving and loading model checkpoints during training.
- `datasets.py`: Script for loading and preprocessing datasets.
- `hyperparameter_optimization.py`: Script for hyperparameter tuning using Random Walk.
- `main.py`: Main script to run the OACE evaluation.
- `metrics.py`: Script for computing assertiveness and cost metrics.
- `models.py`: Script defining the deep learning models (EfficientNetB0, MobileNetV2, etc.).
- `requirements.txt`: List of required Python libraries.
- `validation.py`: Script for validating model performance.
- `visualization.ipynb`: Jupyter notebook for generating plots (box plots, convergence graphs).

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.
