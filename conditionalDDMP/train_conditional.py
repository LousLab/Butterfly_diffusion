import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler


# =========================
# CONFIG
# =========================

DATASET_NAME = "huggan/smithsonian_butterflies_subset"
LABEL_FILE = "data/labels.csv"
MODEL_DIR = "models/conditional_butterfly_ddpm"

IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000

NUM_CLASSES = 4
PATIENCE = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================
# DATASET
# =========================

class ButterflyDataset(Dataset):

    def __init__(self, hf_dataset, labels):

        self.dataset = hf_dataset
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image_id, class_id = self.labels[index]

        image = self.dataset[image_id]["image"].convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(
            class_id,
            dtype=torch.long
        )


# =========================
# LOAD LABELS
# =========================

labels = []

with open(LABEL_FILE, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        labels.append((
            int(row["image_id"]),
            int(row["color_id"])
        ))

print(f"\nLabeled images: {len(labels)}")


# =========================
# LOAD HUGGING FACE DATASET
# =========================

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)


# =========================
# CREATE DATASET
# =========================

full_dataset = ButterflyDataset(
    dataset,
    labels
)


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

validation_size = int(
    len(full_dataset) * 0.2
)

training_size = (
    len(full_dataset) - validation_size
)

train_dataset, validation_dataset = random_split(
    full_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED)
)


# =========================
# BALANCED SAMPLING
# =========================

train_labels = [
    labels[i][1]
    for i in train_dataset.indices
]

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

names = [
    "Brown",
    "Orange",
    "Red",
    "Yellow"
]

for i, name in enumerate(names):
    print(
        f"{name:<8}: "
        f"{class_counts[i]}"
    )


# Give rarer classes larger sampling weights
class_weights = 1.0 / class_counts

sample_weights = [
    class_weights[label]
    for label in train_labels
]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nTraining images: {training_size}")
print(f"Validation images: {validation_size}")


# =========================
# CONDITIONAL UNET
# =========================

model = UNet2DModel(
    sample_size=IMAGE_SIZE,

    in_channels=3,
    out_channels=3,

    layers_per_block=2,

    block_out_channels=(
        64,
        128,
        128,
        256
    ),

    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D"
    ),

    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),

    # THIS makes the model conditional
    num_class_embeds=NUM_CLASSES

).to(device)


# =========================
# DDPM SCHEDULER
# =========================

scheduler = DDPMScheduler(
    num_train_timesteps=NUM_TIMESTEPS
)


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_val_loss = float("inf")
patience_counter = 0

start_time = time.time()

print("\nStarting conditional DDPM training...\n")


for epoch in range(EPOCHS):

    # ---------------------
    # TRAIN
    # ---------------------

    model.train()

    train_loss = 0.0

    for images, class_labels in train_loader:

        images = images.to(device)
        class_labels = class_labels.to(device)

        noise = torch.randn_like(images)

        timesteps = torch.randint(
            0,
            NUM_TIMESTEPS,
            (images.shape[0],),
            device=device
        ).long()

        noisy_images = scheduler.add_noise(
            images,
            noise,
            timesteps
        )

        noise_pred = model(
            noisy_images,
            timesteps,
            class_labels=class_labels
        ).sample

        loss = F.mse_loss(
            noise_pred,
            noise
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # ---------------------
    # VALIDATION
    # ---------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, class_labels in validation_loader:

            images = images.to(device)
            class_labels = class_labels.to(device)

            noise = torch.randn_like(images)

            timesteps = torch.randint(
                0,
                NUM_TIMESTEPS,
                (images.shape[0],),
                device=device
            ).long()

            noisy_images = scheduler.add_noise(
                images,
                noise,
                timesteps
            )

            noise_pred = model(
                noisy_images,
                timesteps,
                class_labels=class_labels
            ).sample

            loss = F.mse_loss(
                noise_pred,
                noise
            )

            val_loss += loss.item()


    val_loss /= len(validation_loader)


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Validation MSE: {val_loss:.4f}"
    )


    # ---------------------
    # SAVE BEST MODEL
    # ---------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        model.save_pretrained(
            MODEL_DIR
        )

        scheduler.save_pretrained(
            MODEL_DIR
        )

        print("  -> Best model saved.")

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping.")

            break


# =========================
# FINISHED
# =========================

elapsed = (
    time.time() - start_time
) / 60

print("\n==============================")
print("CONDITIONAL TRAINING COMPLETE")
print("==============================")

print(
    f"Best Validation MSE: "
    f"{best_val_loss:.4f}"
)

print(
    f"Training time: "
    f"{elapsed:.1f} minutes"
)

print(
    f"Model saved to: "
    f"{MODEL_DIR}"
)