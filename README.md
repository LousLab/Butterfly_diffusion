##Butterfly Diffusion is an image generation project using Denoising Diffusion Probabilistic Models (DDPMs) trained on the Smithsonian Butterflies dataset.

It progresses from unconditional butterfly generation to color-conditioned generation.

The latest model uses Classifier-Free Guidance (CFG) to improve color control.
---
Project Overview
Diffusion models learn image generation by learning to reverse a gradual noise-addition process.
Unconditional DDPM
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
Conditional DDPM
The project was extended to allow generation based on a selected color:
```text
Random Noise + Color Condition
              ↓
       Conditional U-Net
              ↓
      Iterative Denoising
              ↓
       Generated Butterfly
```
Currently supported color conditions:
Brown, Orange, Red, Yellow
---
Technologies
Python
PyTorch
Hugging Face Diffusers
Hugging Face Datasets
Torchvision
OpenCV
NumPy
Matplotlib
scikit-learn
---
Dataset
The project uses the Smithsonian Butterflies subset available through Hugging Face.
Dataset: `huggan/smithsonian_butterflies_subset`
The same dataset is used for both unconditional and conditional diffusion.
For conditional training, images are analyzed and labeled according to their dominant color characteristics.
More information about the dataset is provided in `dataset.txt`.
---
Models
The project uses:
U-Net (`UNet2DModel`) for noise prediction
DDPMScheduler for the diffusion process
AdamW optimizer
Mean Squared Error (MSE) loss
Class embeddings for conditional generation
Classifier-Free Guidance (CFG) for stronger conditional control
Images are resized to 64 × 64 pixels and normalized to [-1, 1].
The diffusion process uses 1000 training timesteps.
---
Color Analysis and Labeling
Before conditional training, the dataset was analyzed using HSV color detection to determine which colors were sufficiently represented.
The analysis considered:
Hue
Saturation
Value
Color coverage within each image
Number of images containing each color
The final conditional classes were:
Color	Approx. Samples
Brown	200
Orange	200
Red	39
Yellow	200
The classes are not perfectly balanced because the dataset contains significantly fewer strongly red examples.
The selected images were labeled and stored in a CSV file for conditional training.
---
Classifier-Free Guidance
The latest conditional model uses Classifier-Free Guidance (CFG).
During training, some images are intentionally trained without a color condition.
During generation, the model produces both conditional and unconditional predictions. These predictions are combined to strengthen the requested condition.
```text
Unconditional Prediction
          +
Conditional Prediction
          ↓
Classifier-Free Guidance
          ↓
Guided Prediction
          ↓
Generated Butterfly
```
The CFG approach produced noticeably stronger color conditioning compared with the earlier conditional models.
---
Installation
Clone the repository:
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Butterfly_diffusion
```
Create and activate a virtual environment:
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts	ctivate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
For NVIDIA GPU training, install the appropriate CUDA-enabled PyTorch build for your system.
---
Training
Unconditional Model
Run:
```bash
python train.py
```
The trained model is saved under:
```text
models/
```
Training and validation MSE are displayed during training, and the training history is recorded for evaluation.
Conditional Model
The conditional training pipeline performs:
Loading the labeled butterfly dataset
Applying color conditions
Training a conditional U-Net
Using balanced sampling to reduce the effect of class imbalance
Training with the DDPM noise-prediction objective
Using Classifier-Free Guidance in the latest version
Run the appropriate conditional training script from the `conditionalDDMP` directory.
---
Image Generation
Unconditional Generation
Run:
```bash
python generate.py
```
The program provides a simple terminal interface for generating butterfly images from random Gaussian noise.
Conditional Generation
Run the conditional generation script from the `conditionalDDMP` directory.
The program provides a terminal interface where a color can be selected before generating butterflies.
Example:
```text
======================================
   CONDITIONAL BUTTERFLY GENERATOR
======================================

1. Brown
2. Orange
3. Red
4. Yellow
5. Exit

Choose a color:
```
Generated images are saved under:
```text
results/
```
---
Evaluation
The models are evaluated using:
Training MSE
Validation MSE
Training/validation loss curves
Qualitative inspection of generated images
Inference-step comparisons
Comparison between unconditional and conditional generation
Color-conditioning consistency
For the unconditional model, different inference step counts were tested:
```text
50
100
400
800
1000
```
Higher inference step counts generally require more computation but can produce more refined results.
For the conditional models, generation quality was additionally evaluated by checking whether the generated butterflies followed the requested color condition.
---
Results
Unconditional DDPM
The unconditional model successfully learned the general distribution of butterfly images and produced recognizable butterfly-like generations.
Conditional DDPM
The initial conditional models showed weak color control.
After introducing Classifier-Free Guidance, the model showed noticeably stronger color conditioning.
The generated samples demonstrate different color characteristics when requesting:
Brown, Orange, Red, and Yellow
However, the generated images remain limited by the 64 × 64 resolution, relatively small conditional dataset, and class imbalance.
---
Current Limitations
The current model is an experimental implementation and is not intended to produce photorealistic butterfly images yet.
Main limitations include:
64 × 64 image resolution
Blurry details
Imperfect wing structures
Occasional visual artifacts
Inconsistent color intensity
Limited conditional training data
Class imbalance, particularly for Red
Overlap between butterfly colors
The color labels represent a dominant color characteristic, not a requirement that every pixel of a butterfly has the selected color.
---
Future Work
Improve image resolution
Increase the conditional training dataset
Improve automatic color labeling
Experiment with different CFG scales
Improve the U-Net architecture
Experiment with different noise schedules
Add quantitative metrics such as FID
Explore additional conditional attributes
Improve overall generation quality
---
Development Progression
The project demonstrates the progression from a basic diffusion model toward controllable image generation:
```text
Unconditional DDPM
        ↓
Dataset Color Analysis
        ↓
Color Labeling
        ↓
Conditional DDPM
        ↓
Improved Training
        ↓
Classifier-Free Guidance
        ↓
Color-Controlled Butterfly Generation
```
---
Author
Rivaldo Lourembam
B.Tech Artificial Intelligence and Machine Learning
