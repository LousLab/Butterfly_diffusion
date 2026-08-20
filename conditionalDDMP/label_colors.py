import os
import csv
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "huggan/smithsonian_butterflies_subset"

OUTPUT_DIR = "data"
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.csv")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "label_samples")

MIN_SATURATION = 50
MIN_VALUE = 35
MAX_VALUE = 245

# Minimum percentage required for a color
COLOR_THRESHOLD = 5.0

# Brown needs a stronger threshold
BROWN_IMAGE_THRESHOLD = 15.0

# Maximum samples per class
MAX_PER_CLASS = 200


# ==========================================
# COLOR IDs
# ==========================================

COLOR_TO_ID = {
    "Brown": 0,
    "Orange": 1,
    "Red": 2,
    "Yellow": 3
}


# ==========================================
# COLOR MASKS
# ==========================================

def get_color_masks(hsv):

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    usable = (
        (s >= MIN_SATURATION) &
        (v >= MIN_VALUE) &
        (v <= MAX_VALUE)
    )

    masks = {}

    # --------------------------------------
    # RED
    # --------------------------------------

    masks["Red"] = usable & (
        ((h <= 8) | (h >= 171)) &
        (s >= 110) &
        (v >= 70)
    )

    # --------------------------------------
    # ORANGE
    # --------------------------------------

    masks["Orange"] = usable & (
        (h >= 9) &
        (h <= 20) &
        (s >= 125) &
        (v >= 80)
    )

    # --------------------------------------
    # YELLOW
    # --------------------------------------

    masks["Yellow"] = usable & (
        (h >= 21) &
        (h <= 38) &
        (s >= 90) &
        (v >= 90)
    )

    # --------------------------------------
    # BROWN
    # --------------------------------------
    # Stricter than before.
    # Higher value threshold helps prevent
    # black wings from becoming brown.

    masks["Brown"] = (
        (h >= 5) &
        (h <= 30) &
        (s >= 45) &
        (s <= 130) &
        (v >= 55) &
        (v <= 180)
    )

    return masks


# ==========================================
# ANALYZE ONE IMAGE
# ==========================================

def analyze_image(image):

    rgb = np.array(
        image.convert("RGB")
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    masks = get_color_masks(hsv)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------
    # Estimate actual butterfly/object area
    # --------------------------------------

    object_mask = (
        (saturation > 20) |
        (value < 235)
    )

    object_pixels = np.sum(
        object_mask
    )

    if object_pixels < 100:
        return None, rgb


    # --------------------------------------
    # Vibrant pixels
    # --------------------------------------

    vibrant_mask = (
        (saturation >= MIN_SATURATION) &
        (value >= MIN_VALUE) &
        (value <= MAX_VALUE)
    )

    vibrant_pixels = np.sum(
        vibrant_mask
    )

    if vibrant_pixels == 0:
        return None, rgb


    proportions = {}


    # ======================================
    # RED
    # Use actual object area
    # ======================================

    red_pixels = np.sum(
        masks["Red"]
    )

    proportions["Red"] = (
        red_pixels /
        object_pixels
    ) * 100


    # ======================================
    # ORANGE
    # Use vibrant pixel area
    # ======================================

    orange_pixels = np.sum(
        masks["Orange"]
    )

    proportions["Orange"] = (
        orange_pixels /
        vibrant_pixels
    ) * 100


    # ======================================
    # YELLOW
    # Use vibrant pixel area
    # ======================================

    yellow_pixels = np.sum(
        masks["Yellow"]
    )

    proportions["Yellow"] = (
        yellow_pixels /
        vibrant_pixels
    ) * 100


    # ======================================
    # BROWN
    # Use actual object area
    # ======================================

    brown_pixels = np.sum(
        masks["Brown"]
    )

    proportions["Brown"] = (
        brown_pixels /
        object_pixels
    ) * 100


    return proportions, rgb


# ==========================================
# MAIN
# ==========================================

def main():

    print(
        f"Loading dataset: {DATASET_NAME}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    total_images = len(dataset)

    print(
        f"Loaded {total_images} images."
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        SAMPLES_DIR,
        exist_ok=True
    )


    # ======================================
    # FIND CANDIDATES
    # ======================================

    candidates = defaultdict(list)

    print(
        "\nAnalyzing images..."
    )


    for image_id, sample in enumerate(dataset):

        result = analyze_image(
            sample["image"]
        )

        if result is None:
            continue

        proportions, image = result


        # ----------------------------------
        # RED
        # ----------------------------------

        if proportions["Red"] >= COLOR_THRESHOLD:

            candidates["Red"].append({
                "image_id": image_id,
                "percentage": proportions["Red"],
                "image": image
            })


        # ----------------------------------
        # ORANGE
        # ----------------------------------

        if proportions["Orange"] >= COLOR_THRESHOLD:

            candidates["Orange"].append({
                "image_id": image_id,
                "percentage": proportions["Orange"],
                "image": image
            })


        # ----------------------------------
        # YELLOW
        # ----------------------------------

        if proportions["Yellow"] >= COLOR_THRESHOLD:

            candidates["Yellow"].append({
                "image_id": image_id,
                "percentage": proportions["Yellow"],
                "image": image
            })


        # ----------------------------------
        # BROWN
        # ----------------------------------

        if proportions["Brown"] >= BROWN_IMAGE_THRESHOLD:

            candidates["Brown"].append({
                "image_id": image_id,
                "percentage": proportions["Brown"],
                "image": image
            })


    # ======================================
    # RAW COUNTS
    # ======================================

    print(
        "\nRaw color candidates:"
    )

    for color in COLOR_TO_ID:

        print(
            f"{color:<10}: "
            f"{len(candidates[color])}"
        )


    # ======================================
    # SELECT DATA
    # ======================================

    selected = defaultdict(list)

    # --------------------------------------
    # RED FIRST
    # --------------------------------------

    red_candidates = sorted(
        candidates["Red"],
        key=lambda x: x["percentage"],
        reverse=True
    )

    selected["Red"] = (
        red_candidates[:MAX_PER_CLASS]
    )

    used_ids = {
        item["image_id"]
        for item in selected["Red"]
    }


    # --------------------------------------
    # OTHER COLORS
    # --------------------------------------

    for color in [
        "Brown",
        "Orange",
        "Yellow"
    ]:

        available = [
            item
            for item in candidates[color]
            if item["image_id"] not in used_ids
        ]

        available.sort(
            key=lambda x: x["percentage"],
            reverse=True
        )

        selected[color] = (
            available[:MAX_PER_CLASS]
        )

        used_ids.update(
            item["image_id"]
            for item in selected[color]
        )


    # ======================================
    # SAVE LABELS
    # ======================================

    rows = []

    for color in COLOR_TO_ID:

        for item in selected[color]:

            rows.append({

                "image_id":
                    item["image_id"],

                "color":
                    color,

                "color_id":
                    COLOR_TO_ID[color],

                "color_percentage":
                    round(
                        item["percentage"],
                        2
                    )
            })


    with open(
        LABEL_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "color",
                "color_id",
                "color_percentage"
            ]
        )

        writer.writeheader()

        writer.writerows(rows)


    # ======================================
    # FINAL DISTRIBUTION
    # ======================================

    counts = Counter(
        row["color"]
        for row in rows
    )

    print("\n")
    print("=" * 60)
    print(
        "FINAL CONDITIONAL DATASET"
    )
    print("=" * 60)

    for color in COLOR_TO_ID:

        print(
            f"{color:<10}: "
            f"{counts[color]} images"
        )

    print("-" * 60)

    print(
        f"Total labeled images: "
        f"{len(rows)}"
    )

    print(
        f"Saved labels to: "
        f"{LABEL_FILE}"
    )


    # ======================================
    # CREATE SAMPLE MONTAGES
    # ======================================

    print(
        "\nCreating verification images..."
    )


    for color in COLOR_TO_ID:

        samples = selected[color][:5]

        if not samples:
            continue


        fig, axes = plt.subplots(
            1,
            5,
            figsize=(15, 3)
        )

        fig.suptitle(
            f"Training examples: {color}"
        )


        for i, ax in enumerate(axes):

            if i < len(samples):

                item = samples[i]

                ax.imshow(
                    item["image"]
                )

                ax.set_title(
                    f"{item['percentage']:.1f}%"
                )

            ax.axis("off")


        plt.tight_layout()


        path = os.path.join(
            SAMPLES_DIR,
            f"{color.lower()}_samples.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()


    print(
        f"Sample images saved to: "
        f"{SAMPLES_DIR}"
    )

    print(
        "\nLabeling complete."
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()