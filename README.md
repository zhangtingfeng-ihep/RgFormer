# RgFormer

## Overview

RgFormer is a Transformer-based machine learning model designed for predicting the Radius of Gyration (Rg) of proteins. This repository contains scripts and resources for training the model, performing predictions, conducting ablation studies, benchmarking against other methods, and analyzing performance curves. The project leverages data from the Small Angle Scattering Biological Data Bank (SASBDB) to facilitate accurate predictions of protein structural properties.

## Features

- **Transformer Architecture**: Utilizes advanced Transformer models to capture sequence dependencies and predict Rg values.
- **Training and Prediction Modules**: Dedicated directories for model training and inference.
- **Ablation Studies**: Tools to evaluate the impact of various model components.
- **Benchmarking**: Comparisons with state-of-the-art methods for Rg prediction.
- **Curve Analysis**: Scripts for visualizing learning curves and prediction accuracy.
- **Data Integration**: Includes SASBDB data in XML format for training and validation.

## Requirements
- Nvidia GPU Gencodes SM_70 or higher
- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for curve visualization)

Install the dependencies using the following command:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas scikit-learn matplotlib
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/zhangtingfeng-ihep/RgFormer.git
   cd RgFormer
   ```

2. Install the required packages as listed above.

## Usage

### Training

Navigate to the `Train` directory and execute the training script:

```
python train.py --data sasbdb_data.xml --epochs 50 --batch_size 32
```

Adjust parameters as necessary based on the script's arguments.

### Prediction

In the `Predict` directory, run the prediction script:

```
python predict.py --model_path path/to/trained_model.pth --input_sequence "protein_sequence"
```

This will output the predicted Rg value.

### Ablation Studies

The `ablation` directory contains scripts for model ablation experiments. Execute:

```
python ablation_study.py --config ablation_config.json
```

### Benchmarking

Use the `benchmark` directory to compare RgFormer with other models:

```
python benchmark.py --models list_of_models --dataset test_data.xml
```

### Curve Analysis

In the `curve` directory, generate visualizations:

```
python plot_curves.py --log_file training_log.csv
```

## Data

The `sasbdb_data.xml` file contains curated data from SASBDB, including protein sequences and associated Rg values for training and evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is supported by the Institute of High Energy Physics (IHEP). Contributions and suggestions are welcome through pull requests or issues.
