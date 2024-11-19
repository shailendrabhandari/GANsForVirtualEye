[![Documentation Status](https://readthedocs.org/projects/gansforvirtualeye/badge/?version=latest)](https://gansforvirtualeye.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/shailendrabhandari/GANsForVirtualEye/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/GANsforVirtualEye)](https://pypi.org/project/GANsforVirtualEye/)
[![PyPI](https://img.shields.io/pypi/v/GANsforVirtualEye)](https://pypi.org/project/GANsforVirtualEye/)
![GitHub watchers](https://img.shields.io/github/watchers/shailendrabhandari/GANsForVirtualEye?style=social)
![GitHub stars](https://img.shields.io/github/stars/shailendrabhandari/GANsForVirtualEye?style=social)


![Gan Architecture](https://raw.githubusercontent.com/shailendrabhandari/GANsForVirtualEye/main/gan_package/results/Class_GAN_Arc.jpg)
# **GAN: Time Series Generation Package**

This package provides an implementation of Generative Adversarial Networks (GANs) for time series generation, with flexible architecture options. Users can select different combinations of generator and discriminator models, including Convolutional Neural Networks (CNN) and Long Short-Term Memory networks (LSTM), to suit their specific needs.

---

## **Table of Contents**

- [Features](#features)
- [Package Structure](#package-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## **Features**

- **Flexible Model Selection**: Choose between CNN and LSTM architectures for both the generator and discriminator.
- **Time Series Generation**: Generate synthetic time series data based on input sequences.
- **Customizable Parameters**: Adjust hyperparameters such as epochs, batch size, and latent dimension.
- **Data Preprocessing**: Includes utilities for loading and preprocessing time series data.
- **Evaluation Metrics**: Calculate and visualize performance metrics like loss and JS divergence.
- **Modular Codebase**: Organized code structure for ease of maintenance and extension.

---

## **Package Structure**

```
GANsForVirtualEye/
├── gan_package/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── models.py
│   ├── train.py
│   ├── testing.py
│   ├── utils.py
├── main.py
├── setup.py
├── requirements.txt
├── README.md
```

- **dataloader.py**: Handles data loading and preprocessing.
- **models.py**: Contains definitions of generator and discriminator models.
- **train.py**: Implements the training loop for the GAN.
- **testing.py**: Contains functions for evaluating and visualizing the results.
- **utils.py**: Provides utility functions used across modules.
- **main.py**: Entry point for running the training and evaluation.
- **setup.py**: Package installation script.
- **requirements.txt**: Lists all package dependencies.

![Velocity Data](https://raw.githubusercontent.com/shailendrabhandari/GANsForVirtualEye/main/gan_package/results/velocity_data.png)


---

## **Installation**

### **Prerequisites**

- Python 3.6 or higher
- `pip` package manager

### **Steps**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shailendrabhandari/GANsForVirtualEye.git 
   cd GANsForVirtualEye
   ```

2. **Install Required Packages**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package**

   ```bash
   pip install .
   ```

---

## **Usage**

The package can be used by running the `main.py` script, which orchestrates the data loading, model training, and evaluation processes.

### **Command-Line Arguments**

You can customize the behavior of the script using the following arguments:

- `--data_path`: Path to your data folder containing the `.txt` files.
- `--save_path`: Path where results and models will be saved.
- `--epochs`: Number of training epochs (default: 500).
- `--batch_size`: Batch size for training (default: 128).
- `--latent_dim`: Dimension of the latent space for the generator (default: 256).
- `--generator_model`: Generator model to use (`CNNGenerator2` or `LSTMGenerator`).
- `--discriminator_model`: Discriminator model to use (`CNNDiscriminator2` or `LSTMDiscriminator`).

### **Example Commands**

#### **1. CNN Generator with CNN Discriminator**

```bash
python main.py \
--data_path '/path/to/your/data' \
--save_path './results' \
--generator_model 'CNNGenerator' \
--discriminator_model 'CNNDiscriminator'
```

#### **2. LSTM Generator with LSTM Discriminator**

```bash
python main.py \
--data_path '/path/to/your/data' \
--save_path './results' \
--generator_model 'LSTMGenerator' \
--discriminator_model 'LSTMDiscriminator'
```

#### **3. CNN Generator with LSTM Discriminator**

```bash
python main.py \
--data_path '/path/to/your/data' \
--save_path './results' \
--generator_model 'CNNGenerator' \
--discriminator_model 'LSTMDiscriminator'
```

#### **4. LSTM Generator with CNN Discriminator**

```bash
python main.py \
--data_path '/path/to/your/data' \
--save_path './results' \
--generator_model 'LSTMGenerator' \
--discriminator_model 'CNNDiscriminator'
```

### **Note**

- Replace `'/path/to/your/data'` with the actual path to your data folder.
- The script automatically detects if a GPU is available and uses it for training if possible.

---

## **Data Preparation**

This package explicitely expects time series data in the form of `.txt` files, each containing sequences of velocity measurements or similar metrics. Modify it depending on the nature of your datasets.  

### **Data Format**

- Each `.txt` file should contain data in columns representing:
  - Time stamps
  - X and Y positions for left and right eye
  - Additional relevant metrics (e.g., saccade indicators) and so 

### **Data Loading**

The `dataloader.py` module handles data loading and preprocessing:

- **Data Cleaning**: Removes the first `n` data points and handles NaN values.
- **Feature Engineering**: Calculates velocities and filters out non-positive values.
- **Normalization**: Normalizes the data using `MinMaxScaler`.
- **Sequence Sampling**: Samples sequences of a specified length for training.

### **Adjusting Parameters**

You can adjust data preprocessing parameters by modifying the `load_and_preprocess_data` and `prepare_datasets` functions in `dataloader.py`.

---

## **Model Architectures**

### **Generators**

#### **1. CNNGenerator**

A convolutional neural network generator that uses transpose convolutional layers to generate sequences.

- **Input**: Latent vector of shape `(batch_size, latent_dim, 1)`
- **Output**: Generated sequence of shape `(batch_size, 1, sequence_length)`

#### **2. LSTMGenerator**

An LSTM-based generator that generates sequences by processing latent vectors at each time step.

- **Input**: Latent vector of shape `(batch_size, sequence_length, latent_dim)`
- **Output**: Generated sequence of shape `(batch_size, sequence_length, output_channels)`

### **Discriminators**

#### **1. CNNDiscriminator**

A CNN-based discriminator that classifies sequences using convolutional layers.

- **Input**: Sequence of shape `(batch_size, 1, sequence_length)`
- **Output**: Probability score indicating real or fake

#### **2. LSTMDiscriminator**

An LSTM-based discriminator that processes sequences and outputs a classification.

- **Input**: Sequence of shape `(batch_size, sequence_length, input_size)`
- **Output**: Probability score indicating real or fake

---

## **Results**

After training, results and models are saved to the specified `--save_path` directory.

- **Model Checkpoints**: Saved as `generator.pt` and `discriminator.pt`.
- **Training Metrics**: Spectral Loss values and divergence scores saved as `.npy` files.
- **Evaluation Plot**: A histogram comparing real and generated data distributions saved as `RealVSGenerated_velGAN.pdf`.

### **Interpreting the Histogram**

The evaluation plot shows the distribution of the log velocities for both real and generated data. A closer alignment indicates better performance of the GAN.

---

## **Dependencies**

- Python 3.6 or higher
- `numpy`
- `torch`
- `torchvision`
- `matplotlib`
- `scipy`
- `sklearn`
- `pandas`
- `progressbar2`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click the 'Fork' button at the top right of the repository page.

2. **Clone Your Fork**

   ```bash
   https://github.com/shailendrabhandari/GANsForVirtualEye.git
   cd GANsForVirtualEye
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your_feature_name
   ```

4. **Make Changes and Commit**

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your_feature_name
   ```

6. **Submit a Pull Request**

   Go to the original repository and click 'New Pull Request'.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](https://github.com/shailendrabhandari/GANsForVirtualEye/blob/main/LICENSE) file for details.

---

## **Contact**

For questions or suggestions, please contact:

- **Shailendra Bhandari**
- **Email**: shailendra.bhandari@oslomet.no
- **GitHub**: [shailendrabhandari](https://github.com/shailendrabhandari)



## **Acknowledgments**

- Thank you to all contributors and the AI lab teams who helped improve this package.
- Inspired by research on GANs for time series generation.

---

# **Frequently Asked Questions (FAQ)**

### **1. What types of data can I use with this package?**

The package is designed for time series data, specifically sequences of numerical values like velocities.

### **2. Can I use this package for other types of data?**

While the package is tailored for time series data, you can extend or modify it to handle other sequential data types with appropriate adjustments to the data loader and models.

### **3. How can I adjust the sequence length or number of sequences?**

You can modify the `sequence_length` and `num_sequences` parameters in the `prepare_datasets` function within `dataloader.py`.

### **4. How do I know if the models are training correctly?**

Monitor the loss values and JS divergence printed during training. Decreasing loss values generally indicate that the models are learning. You can also examine the evaluation plots for visual confirmation.

### **5. Can I add new models to the package?**

Yes! The package is modular, allowing you to add new generator and discriminator models. Ensure they are properly defined in `models.py` and included in the `get_generator` and `get_discriminator` functions.


---

# **Thank You for Using GAN Time Series Generation Package!**

We hope this package helps you in your research or projects involving time series data generation. Your feedback is valuable to us.

