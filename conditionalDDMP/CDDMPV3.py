import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler

# ==========================================
# CONDITIONAL DDPM V3 — CLASSIFIER-FREE GUIDANCE
# ==========================================

DATASET_NAME = "huggan/smithsonian_butterflies_subset"
LABEL_FILE = "data/labels.csv"
MODEL_DIR = "models/conditional_butterfly_ddpm_v3"

IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000
NUM_CLASSES = 4

# Extra class ID used for "no color condition"
NULL_CLASS = NUM_CLASSES

# Probability of dropping the color condition during training
COND_DROP_PROB = 0.15

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# DATASET
# ==========================================

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
        return (
            self.transform(image),
            torch.tensor(class_id, dtype=torch.long)
        )


# ==========================================
# LOAD LABELS
# ==========================================

labels = []

with open(LABEL_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append((
            int(row["image_id"]),
            int(row["color_id"])
        ))

print(f"\nLabeled images: {len(labels)}")

class_names = ["Brown", "Orange", "Red", "Yellow"]
counts = np.bincount(
    [x[1] for x in labels],
    minlength=NUM_CLASSES
)

print("\nFull dataset:")
for i, name in enumerate(class_names):
    print(f"{name:<8}: {counts[i]}")


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading butterfly dataset...")
hf_dataset = load_dataset(DATASET_NAME, split="train")

indices = np.arange(len(labels))
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

val_count = int(len(indices) * 0.2)

train_labels = [labels[i] for i in indices[val_count:]]
val_labels = [labels[i] for i in indices[:val_count]]

train_dataset = ButterflyDataset(hf_dataset, train_labels)

# Separate validation dataset without augmentation
class ValidationDataset(ButterflyDataset):
    def __init__(self, hf_dataset, labels):
        super().__init__(hf_dataset, labels)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

val_dataset = ValidationDataset(hf_dataset, val_labels)


# ==========================================
# BALANCED SAMPLING
# ==========================================

train_class_ids = [x[1] for x in train_labels]

train_counts = np.bincount(
    train_class_ids,
    minlength=NUM_CLASSES
)

print("\nTraining distribution:")
for i, name in enumerate(class_names):
    print(f"{name:<8}: {train_counts[i]}")

class_weights = 1.0 / np.maximum(train_counts, 1)

sample_weights = [
    class_weights[class_id]
    for class_id in train_class_ids
]

sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ==========================================
# MODEL
# ==========================================

print("\nCreating CFG conditional UNet...")

# 4 real classes + 1 null/unconditional class
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
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
    num_class_embeds=NUM_CLASSES + 1
).to(device)

scheduler = DDPMScheduler(
    num_train_timesteps=NUM_TIMESTEPS
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# TRAIN
# ==========================================

best_val_loss = float("inf")
history = []

start_time = time.time()

print("\n==========================================")
print("STARTING CONDITIONAL DDPM V3")
print("Classifier-Free Guidance Training")
print("==========================================\n")

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for images, class_labels in train_loader:

        images = images.to(device)
        class_labels = class_labels.to(device)

        # Randomly remove the color condition for some examples.
        drop_mask = torch.rand(
            class_labels.shape,
            device=device
        ) < COND_DROP_PROB

        training_labels = class_labels.clone()
        training_labels[drop_mask] = NULL_CLASS

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
            class_labels=training_labels
        ).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for images, class_labels in val_loader:

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

            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    history.append((epoch + 1, train_loss, val_loss))

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Validation MSE: {val_loss:.4f}"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save_pretrained(MODEL_DIR)
        scheduler.save_pretrained(MODEL_DIR)

        torch.save(
            {
                "epoch": epoch + 1,
                "best_validation_mse": best_val_loss,
                "null_class": NULL_CLASS
            },
            os.path.join(
                MODEL_DIR,
                "training_info.pt"
            )
        )

        print("  -> New best model saved.")


# ==========================================
# SAVE HISTORY
# ==========================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

with open(history_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_mse",
        "validation_mse"
    ])
    writer.writerows(history)


elapsed = (time.time() - start_time) / 60

print("\n==========================================")
print("V3 TRAINING COMPLETE")
print("==========================================")
print(f"Best Validation MSE: {best_val_loss:.4f}")
print(f"Training time: {elapsed:.1f} minutes")
print(f"Model saved to: {MODEL_DIR}")