# RgFormer

## Overview

RgFormer is a Transformer-based machine learning model designed for predicting the Radius of Gyration (Rg) of proteins. This repository contains scripts and resources for training the model, performing predictions, conducting ablation studies, benchmarking against other methods, and analyzing performance curves. The project leverages data from the Small Angle Scattering Biological Data Bank (SASBDB) to facilitate accurate predictions of protein structural properties.

## Features

- **Transformer Architecture**: Utilizes advanced Transformer models to capture sequence dependencies and predict Rg values.
- **Training and Prediction Modules**: Dedicated directories for model training and inference.
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


## Data

The `sasbdb_data.xml` file contains curated data from SASBDB, including biomolecules entry in SASBDB and associated Rg values for training and evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work is supported by the Institute of High Energy Physics (IHEP). Contributions and suggestions are welcome through pull requests or issues.
