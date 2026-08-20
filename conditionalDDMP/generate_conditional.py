import os
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = "models/conditional_butterfly_ddpm"
IMAGE_SIZE = 64
NUM_GENERATED = 4
NUM_INFERENCE_STEPS = 100

COLORS = {
    1: ("Brown", 0),
    2: ("Orange", 1),
    3: ("Red", 2),
    4: ("Yellow", 3),
}

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
# LOAD BEST SAVED MODEL
# ==========================================

print("\nLoading conditional DDPM...")

model = UNet2DModel.from_pretrained(
    MODEL_DIR
).to(device)

scheduler = DDPMScheduler.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded successfully.")

# ==========================================
# GENERATION
# ==========================================

def generate_butterflies(class_id):

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    sample = torch.randn(
        NUM_GENERATED,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )

    class_labels = torch.full(
        (NUM_GENERATED,),
        class_id,
        device=device,
        dtype=torch.long
    )

    with torch.no_grad():

        for timestep in scheduler.timesteps:

            timesteps = torch.full(
                (NUM_GENERATED,),
                timestep,
                device=device,
                dtype=torch.long
            )

            noise_pred = model(
                sample,
                timesteps,
                class_labels=class_labels
            ).sample

            sample = scheduler.step(
                noise_pred,
                timestep,
                sample
            ).prev_sample

    sample = (sample.clamp(-1, 1) + 1) / 2

    return sample.cpu()

# ==========================================
# TERMINAL INTERFACE
# ==========================================

print("\n======================================")
print("     CONDITIONAL BUTTERFLY GENERATOR")
print("======================================")
print("1. Brown")
print("2. Orange")
print("3. Red")
print("4. Yellow")
print("5. Exit")

while True:

    choice = input("\nChoose a color (1-5): ").strip()

    if choice == "5":
        print("Exiting.")
        break

    if not choice.isdigit() or int(choice) not in COLORS:
        print("Invalid choice. Enter 1, 2, 3, or 4.")
        continue

    choice = int(choice)

    color_name, class_id = COLORS[choice]

    print(
        f"\nGenerating {NUM_GENERATED} "
        f"{color_name} butterflies..."
    )

    images = generate_butterflies(class_id)

    # ======================================
    # DISPLAY
    # ======================================

    fig, axes = plt.subplots(
        1,
        NUM_GENERATED,
        figsize=(12, 3)
    )

    fig.suptitle(
        f"Conditional DDPM - {color_name}"
    )

    for i, ax in enumerate(axes):

        image = images[i].permute(
            1, 2, 0
        ).numpy()

        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()

    # ======================================
    # SAVE
    # ======================================

    os.makedirs("results", exist_ok=True)

    output_path = os.path.join(
        "results",
        f"generated_{color_name.lower()}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Saved to: {output_path}")