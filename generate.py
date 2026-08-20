import torch
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt
import os


# =========================
# 1. Configuration
# =========================

IMAGE_SIZE = 64
MODEL_PATH = "models/butterfly_ddpm"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. Load trained model
# =========================

model = UNet2DModel.from_pretrained(
    MODEL_PATH
).to(DEVICE)

model.eval()


# =========================
# 3. Load DDPM scheduler
# =========================

noise_scheduler = DDPMScheduler.from_pretrained(
    MODEL_PATH
)


# =========================
# 4. Generate butterflies
# =========================

def generate_butterflies(num_images=1):

    # Start with random Gaussian noise
    sample = torch.randn(
        num_images,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )

    # Use the scheduler's inference timesteps
    noise_scheduler.set_timesteps(1000)

    # Gradually remove noise
    for timestep in noise_scheduler.timesteps:

        with torch.no_grad():

            noise_prediction = model(
                sample,
                timestep
            ).sample

        sample = noise_scheduler.step(
            noise_prediction,
            timestep,
            sample
        ).prev_sample

    # Convert from [-1, 1] to [0, 1]
    images = (sample.clamp(-1, 1) + 1) / 2

    return images.cpu()


# =========================
# 5. Terminal interface
# =========================

while True:

    print("\n==============================")
    print("   Butterfly Diffusion Model")
    print("==============================")
    print("1. Generate 1 butterfly")
    print("2. Generate 4 butterflies")
    print("3. Exit")

    choice = input("\nChoose an option: ")

    if choice == "1":

        print("\nGenerating butterfly...")

        images = generate_butterflies(1)

        os.makedirs(
            "results/samples",
            exist_ok=True
        )

        filename = "results/samples/butterfly.png"

        image = images[0].permute(1, 2, 0).numpy()

        plt.imsave(
            filename,
            image
        )

        print(f"Saved to: {filename}")

        plt.imshow(image)
        plt.axis("off")
        plt.show()


    elif choice == "2":

        print("\nGenerating 4 butterflies...")

        images = generate_butterflies(4)

        os.makedirs(
            "results/samples",
            exist_ok=True
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(12, 3)
        )

        for i, ax in enumerate(axes):

            image = images[i].permute(
                1, 2, 0
            ).numpy()

            ax.imshow(image)
            ax.axis("off")

            plt.imsave(
                f"results/samples/butterfly_{i+1}.png",
                image
            )

        plt.show()

        print("Saved 4 butterflies to results/samples/")


    elif choice == "3":

        print("Exiting...")
        break


    else:

        print("Invalid choice. Please enter 1, 2, or 3.")