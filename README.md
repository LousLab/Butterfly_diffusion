# Butterfly Diffusion

An image generation project using a Denoising Diffusion Probabilistic Model (DDPM) trained on the Smithsonian Butterflies dataset.

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
U-Net predicts the noise
      ↓
Train the model
      ↓
Start from random noise
      ↓
Iteratively remove noise
      ↓
Generated Butterfly
```

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
* **DDPMScheduler** for the diffusion process
* **AdamW** optimizer
* **Mean Squared Error (MSE)** loss

Images are resized to **64 × 64 pixels** and normalized to the range **[-1, 1]**.

The diffusion process uses **1000 training timesteps**.

## Training

The model is trained by:

1. Selecting a butterfly image.
2. Generating random Gaussian noise.
3. Selecting a random diffusion timestep.
4. Adding noise according to that timestep.
5. Giving the noisy image and timestep to the U-Net.
6. Predicting the noise.
7. Comparing predicted noise with the actual noise using MSE.
8. Updating the model weights.

Training and validation MSE are recorded during training.

## Evaluation

The model is evaluated using:

* Training MSE
* Validation MSE
* Training/validation loss curves
* Qualitative inspection of generated images

Different numbers of inference steps were also tested to study the trade-off between generation speed and image quality.

### Inference Step Experiment

The following inference step counts were compared:

* 50
* 100
* 250
* 500
* 1000

Higher inference step counts generally produced more refined results but required more computation.

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

For NVIDIA GPU training, install the appropriate CUDA-enabled PyTorch build for your system.

## Training

Run:

```bash
python train.py
```

The trained model is saved under:

```text
models/butterfly_ddpm/
```

The training and validation loss curve is also generated.

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

## Results

The current model is able to generate recognizable butterfly-like images from random noise.

The generated images still have limitations such as:

* Low resolution
* Blurry details
* Imperfect wing structures
* Occasional color and background artifacts

These results provide the baseline for further development.

## Future Work

* Experiment with different noise schedules
* Improve the U-Net architecture
* Increase image resolution
* Add quantitative evaluation such as FID
* Implement conditional diffusion
* Add color-controlled butterfly generation

## Author

**Rivaldo Lourembam**

B.Tech Artificial Intelligence and Machine Learning
