# MLOps Deep Learning Project with Quantization

This project demonstrates an end-to-end MLOps setup for a deep learning model, focusing on quantization. We utilize GitHub Actions for CI/CD, MLflow for model monitoring, and DVC for data & model versioning.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Quantization](#quantization)
  - [Serving](#serving)
- [MLOps Integrations](#mlops-integrations)
  - [GitHub Actions](#github-actions)
  - [MLflow](#mlflow)
  - [DVC](#dvc)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project is organized as follows:

- Training, quantization, and serving scripts are in the `model` directory.
- The Dockerfile is used to containerize the serving script.
- The `.github` directory contains the GitHub Actions CI/CD workflow.
- DVC is set up with a mock configuration for data and model versioning.

## Setup

### Requirements

- Python 3.8+
- Docker
- DVC (with a configured remote, for full functionality)

### Installation

1. Clone the repository:
```bash
git clone [repository_url]
cd my_mlops_project
```

2. Install the Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the neural network model on the MNIST dataset:

```bash
python model/train.py
```

This will produce a `model.pth` file.

### Quantization

Quantize the trained model:

```bash
python model/quantize.py
```

This produces a quantized model saved as `quantized_model.pth`.

### Serving

To serve the quantized model using Flask:

```bash
python model/serve.py
```

This will start a Flask server on port 5000.

## MLOps Integrations

### GitHub Actions

The `.github/workflows/main.yml` file defines a CI/CD pipeline that:

- Trains the model.
- Quantizes the model.
- Builds a Docker image for serving.
- Mocks a DVC push for data and model versioning.
- Starts an MLflow server for model monitoring.

### MLflow

Metrics, parameters, and artifacts from the training and quantization steps are logged to MLflow. To view the MLflow UI:

```bash
mlflow ui
```

### DVC

DVC is set up to version the MNIST dataset and the model files. A mock DVC configuration is provided, which can be adjusted to connect to an actual DVC remote.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
