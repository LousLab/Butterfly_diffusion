import os
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

# ==========================================
# CONDITIONAL DDPM V3 GENERATOR
# ==========================================

MODEL_DIR = "models/conditional_butterfly_ddpm_v3"

IMAGE_SIZE = 64
NUM_GENERATED = 4
NUM_INFERENCE_STEPS = 100

# Class IDs from label_colors.py
COLORS = {
    1: ("Brown", 0),
    2: ("Orange", 1),
    3: ("Red", 2),
    4: ("Yellow", 3)
}

NUM_CLASSES = 4
NULL_CLASS = NUM_CLASSES

# CFG strength.
# Higher = stronger conditioning, but too high can hurt quality.
GUIDANCE_SCALE = 3.0


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
# LOAD MODEL
# ==========================================

print("\nLoading V3 conditional DDPM...")

model = UNet2DModel.from_pretrained(
    MODEL_DIR
).to(device)

scheduler = DDPMScheduler.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded successfully.")


# ==========================================
# GENERATE WITH CLASSIFIER-FREE GUIDANCE
# ==========================================

def generate_butterflies(class_id):

    scheduler.set_timesteps(
        NUM_INFERENCE_STEPS
    )

    sample = torch.randn(
        NUM_GENERATED,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )

    conditional_labels = torch.full(
        (NUM_GENERATED,),
        class_id,
        device=device,
        dtype=torch.long
    )

    unconditional_labels = torch.full(
        (NUM_GENERATED,),
        NULL_CLASS,
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

            # Unconditional prediction
            noise_uncond = model(
                sample,
                timesteps,
                class_labels=unconditional_labels
            ).sample

            # Conditional prediction
            noise_cond = model(
                sample,
                timesteps,
                class_labels=conditional_labels
            ).sample

            # Classifier-free guidance
            noise_pred = (
                noise_uncond +
                GUIDANCE_SCALE *
                (noise_cond - noise_uncond)
            )

            sample = scheduler.step(
                noise_pred,
                timestep,
                sample
            ).prev_sample

    return (
        (sample.clamp(-1, 1) + 1) / 2
    ).cpu()


# ==========================================
# TERMINAL INTERFACE
# ==========================================

print("\n======================================")
print("   CONDITIONAL BUTTERFLY GENERATOR V3")
print("======================================")
print(f"Guidance scale: {GUIDANCE_SCALE}")
print(f"Inference steps: {NUM_INFERENCE_STEPS}")
print()
print("1. Brown")
print("2. Orange")
print("3. Red")
print("4. Yellow")
print("5. Exit")


while True:

    choice = input(
        "\nChoose a color (1-5): "
    ).strip()

    if choice == "5":
        print("Exiting.")
        break

    if not choice.isdigit() or int(choice) not in COLORS:
        print(
            "Invalid choice. "
            "Enter 1, 2, 3, 4, or 5."
        )
        continue

    choice = int(choice)

    color_name, class_id = COLORS[choice]

    print(
        f"\nGenerating {NUM_GENERATED} "
        f"{color_name} butterflies..."
    )

    images = generate_butterflies(
        class_id
    )


    # ======================================
    # DISPLAY
    # ======================================

    fig, axes = plt.subplots(
        1,
        NUM_GENERATED,
        figsize=(12, 3)
    )

    fig.suptitle(
        f"Conditional DDPM V3 - {color_name}"
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

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = os.path.join(
        "results",
        f"v3_{color_name.lower()}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Saved result to: {output_path}"
    )