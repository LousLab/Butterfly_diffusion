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


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "huggan/smithsonian_butterflies_subset"
LABEL_FILE = "data/labels.csv"
MODEL_DIR = "models/conditional_butterfly_ddpm"

IMAGE_SIZE = 64
BATCH_SIZE = 16

# Train longer than the first experiment.
EPOCHS = 40

LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000
NUM_CLASSES = 4

SEED = 42


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# DATASET
# ==========================================

class ButterflyDataset(Dataset):

    def __init__(self, hf_dataset, labels, train=True):

        self.dataset = hf_dataset
        self.labels = labels
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]
                )
            ])
        else:
            # No random augmentation in validation.
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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


# ==========================================
# LOAD LABELS
# ==========================================

print("\nLoading labels...")

labels = []

with open(LABEL_FILE, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        labels.append((
            int(row["image_id"]),
            int(row["color_id"])
        ))

print(f"Labeled images: {len(labels)}")


# ==========================================
# SHOW CLASS COUNTS
# ==========================================

class_names = [
    "Brown",
    "Orange",
    "Red",
    "Yellow"
]

counts = np.bincount(
    [x[1] for x in labels],
    minlength=NUM_CLASSES
)

print("\nFull dataset distribution:")

for i, name in enumerate(class_names):
    print(f"{name:<8}: {counts[i]}")


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading butterfly dataset...")

hf_dataset = load_dataset(
    DATASET_NAME,
    split="train"
)


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

# Keep the label list together with the image IDs.
indices = np.arange(len(labels))

rng = np.random.default_rng(SEED)
rng.shuffle(indices)

validation_count = int(
    len(indices) * 0.2
)

validation_indices = indices[:validation_count]
training_indices = indices[validation_count:]


train_labels = [
    labels[i]
    for i in training_indices
]

validation_labels = [
    labels[i]
    for i in validation_indices
]


train_dataset = ButterflyDataset(
    hf_dataset,
    train_labels,
    train=True
)

validation_dataset = ButterflyDataset(
    hf_dataset,
    validation_labels,
    train=False
)


# ==========================================
# BALANCED TRAINING SAMPLER
# ==========================================

train_class_ids = [
    label[1]
    for label in train_labels
]

train_counts = np.bincount(
    train_class_ids,
    minlength=NUM_CLASSES
)

print("\nTraining distribution:")

for i, name in enumerate(class_names):
    print(f"{name:<8}: {train_counts[i]}")


# Inverse-frequency weights.
# Red gets more sampling probability because
# it has fewer real examples.
class_weights = 1.0 / np.maximum(
    train_counts,
    1
)

sample_weights = [
    class_weights[class_id]
    for class_id in train_class_ids
]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)


# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")


# ==========================================
# CONDITIONAL UNET
# ==========================================

print("\nCreating conditional UNet...")

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

    # Four class embeddings:
    # 0 = Brown
    # 1 = Orange
    # 2 = Red
    # 3 = Yellow
    num_class_embeds=NUM_CLASSES

).to(device)


# ==========================================
# DDPM
# ==========================================

scheduler = DDPMScheduler(
    num_train_timesteps=NUM_TIMESTEPS
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# SAVE DIRECTORY
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================
# TRAINING
# ==========================================

best_val_loss = float("inf")
start_time = time.time()

history = []

print("\n==========================================")
print("STARTING CONDITIONAL DDPM V2")
print("==========================================\n")


for epoch in range(EPOCHS):

    # --------------------------------------
    # TRAIN
    # --------------------------------------

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


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

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

    history.append(
        (epoch + 1, train_loss, val_loss)
    )


    # --------------------------------------
    # PRINT
    # --------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Validation MSE: {val_loss:.4f}"
    )


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save_pretrained(
            MODEL_DIR
        )

        scheduler.save_pretrained(
            MODEL_DIR
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "best_validation_mse": best_val_loss
            },
            os.path.join(
                MODEL_DIR,
                "training_info.pt"
            )
        )

        print("  -> New best model saved.")


    # --------------------------------------
    # PERIODIC CHECKPOINT
    # --------------------------------------

    if (epoch + 1) % 5 == 0:

        checkpoint_dir = os.path.join(
            MODEL_DIR,
            f"checkpoint_epoch_{epoch + 1}"
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        model.save_pretrained(
            checkpoint_dir
        )

        scheduler.save_pretrained(
            checkpoint_dir
        )

        print(
            f"  -> Checkpoint saved: "
            f"epoch {epoch + 1}"
        )


# ==========================================
# SAVE TRAINING HISTORY
# ==========================================

history_file = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

with open(
    history_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_mse",
        "validation_mse"
    ])

    writer.writerows(history)


# ==========================================
# FINISHED
# ==========================================

elapsed = (
    time.time() - start_time
) / 60

print("\n==========================================")
print("CONDITIONAL DDPM V2 COMPLETE")
print("==========================================")

print(
    f"Best Validation MSE: "
    f"{best_val_loss:.4f}"
)

print(
    f"Training time: "
    f"{elapsed:.1f} minutes"
)

print(
    f"Best model: "
    f"{MODEL_DIR}"
)

print(
    f"History: "
    f"{history_file}"
)