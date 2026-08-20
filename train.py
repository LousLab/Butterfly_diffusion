import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import os


# =========================
# 1. Configuration
# =========================

IMAGE_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. Load Dataset
# =========================

dataset = load_dataset(
    "huggan/smithsonian_butterflies_subset",
    split="train"
)

# 90% training, 10% validation
split_dataset = dataset.train_test_split(test_size=0.1)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


# =========================
# 3. Preprocessing
# =========================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def transform_images(examples):
    images = [
        preprocess(image.convert("RGB"))
        for image in examples["image"]
    ]

    return {"images": images}


train_dataset.set_transform(transform_images)
val_dataset.set_transform(transform_images)


def collate_fn(batch):
    return torch.stack([
        item["images"] for item in batch
    ])


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================
# 4. U-Net
# =========================

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
    )
).to(DEVICE)


# =========================
# 5. DDPM Scheduler
# =========================

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000
)


# =========================
# 6. Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# 7. Training
# =========================

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

    model.train()

    total_train_loss = 0

    for clean_images in train_loader:

        clean_images = clean_images.to(DEVICE)

        noise = torch.randn_like(clean_images)

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=DEVICE
        ).long()

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps
        )

        noise_prediction = model(
            noisy_images,
            timesteps
        ).sample

        loss = F.mse_loss(
            noise_prediction,
            noise
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    train_losses.append(average_train_loss)


    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for clean_images in val_loader:

            clean_images = clean_images.to(DEVICE)

            noise = torch.randn_like(clean_images)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=DEVICE
            ).long()

            noisy_images = noise_scheduler.add_noise(
                clean_images,
                noise,
                timesteps
            )

            noise_prediction = model(
                noisy_images,
                timesteps
            ).sample

            val_loss = F.mse_loss(
                noise_prediction,
                noise
            )

            total_val_loss += val_loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )

    val_losses.append(average_val_loss)


    # ---------------------
    # Print metrics
    # ---------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train MSE: {average_train_loss:.4f} | "
        f"Validation MSE: {average_val_loss:.4f}"
    )


# =========================
# 8. Save Model
# =========================

os.makedirs(
    "models/butterfly_ddpm",
    exist_ok=True
)

model.save_pretrained(
    "models/butterfly_ddpm"
)

noise_scheduler.save_pretrained(
    "models/butterfly_ddpm"
)

print("\nModel saved successfully!")


# =========================
# 9. Plot Loss
# =========================

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Training MSE"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    label="Validation MSE"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("DDPM Training and Validation Loss")
plt.legend()

plt.savefig("training_loss.png")

plt.show()