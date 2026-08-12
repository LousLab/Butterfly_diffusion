# Butterfly Diffusion

An unconditional image generation project using a Denoising Diffusion Probabilistic Model (DDPM) trained on the Smithsonian Butterflies dataset.

The project uses **PyTorch** and the **Hugging Face Diffusers** library to train a U-Net model to progressively denoise random Gaussian noise into new butterfly images.

## Project Overview

Diffusion models learn image generation by learning to reverse a gradual noise-addition process.

In this project:

```text
Butterfly Image
      ↓
Add Gaussian Noise
      ↓
Noisy Image
      ↓
U-Net learns to predict the noise
      ↓
Train the model
      ↓
Start from random noise
      ↓
Iteratively remove noise
      ↓
Generated Butterfly
```

The current version is **unconditional**, meaning the user does not specify a class, color, or text prompt when generating an image.

## Technologies

* Python
* PyTorch
* Hugging Face Diffusers
* Hugging Face Datasets
* Torchvision
* Matplotlib

## Dataset

The model is trained using the **Smithsonian Butterflies** subset available through Hugging Face.

Dataset:

`huggan/smithsonian_butterflies_subset`

More information about the dataset is provided in [`dataset.txt`](dataset.txt).

## Model

The project uses:

* **U-Net (`UNet2DModel`)** for noise prediction
* **DDPMScheduler** for the forward and reverse diffusion process
* **AdamW** optimizer
* **Mean Squared Error (MSE)** loss

Images are resized to **64 × 64 pixels** and normalized to the range **[-1, 1]**.

The diffusion process uses **1000 training timesteps**.

## Training

The model is trained by:

1. Selecting a clean butterfly image.
2. Generating random Gaussian noise.
3. Selecting a random diffusion timestep.
4. Adding noise according to that timestep.
5. Giving the noisy image and timestep to the U-Net.
6. Predicting the noise.
7. Comparing predicted noise with the actual noise using MSE.
8. Updating the model weights.

Both training and validation MSE are recorded during training.

## Evaluation

The current baseline uses:

* Training MSE
* Validation MSE
* Training/validation loss curves
* Qualitative inspection of generated images

Inference speed and image quality are also compared using different numbers of reverse-diffusion steps.

### Inference Step Experiment

Different numbers of inference steps were tested:

* 50
* 100
* 250
* 500
* 1000

The purpose was to observe the trade-off between generation speed and image quality.

## Project Structure

```text
butterfly-diffusion/
│
├── train.py
├── generate.py
├── requirements.txt
├── dataset.txt
├── README.md
├── .gitignore
│
├── models/
│   └── butterfly_ddpm/
│
└── results/
    └── samples/
```

### `train.py`

Trains the unconditional DDPM and saves the trained model and scheduler.

### `generate.py`

Loads the saved model and generates new butterfly images without retraining.

### `dataset.txt`

Contains information about the dataset, its source, and preprocessing.

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `results/`

Contains generated samples and experiment results.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd butterfly-diffusion
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU training, install the appropriate CUDA-enabled PyTorch build according to your system.

## Training

Run:

```bash
python train.py
```

The trained model will be saved under:

```text
models/butterfly_ddpm/
```

The training loss curve will also be generated.

## Image Generation

After training:

```bash
python generate.py
```

The program provides a simple terminal interface for generating butterfly images from random Gaussian noise.

Generated images are saved in:

```text
results/samples/
```

## Current Results

The current baseline demonstrates that the trained DDPM can generate recognizable butterfly-like images from random noise.

The generated images still have limitations such as:

* low resolution
* blurry details
* imperfect wing structure
* occasional color/background artifacts

These results serve as the baseline for further experimentation.

## Future Work

Planned improvements include:

* Experimenting with different noise schedules
* Improving the U-Net architecture
* Increasing image resolution
* Quantitative image-generation evaluation such as FID
* Conditional diffusion
* Color-controlled butterfly generation

## Author

**Rivaldo Lourembam**

B.Tech Artificial Intelligence and Machine Learning
