from pathlib import Path
from PIL import Image


IMAGE_DIR = Path(
    "data/raw/memotion_dataset_7k/images"
)


bad_images = []

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
]


print(
    f"Checking {len(image_files)} images..."
)


for i, image_path in enumerate(image_files):

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception as e:

        bad_images.append(
            (image_path, str(e))
        )

    if (i + 1) % 500 == 0:
        print(
            f"Checked {i + 1}/{len(image_files)}"
        )


print("\n" + "=" * 50)

print(
    f"Total images: {len(image_files)}"
)

print(
    f"Bad images: {len(bad_images)}"
)


if bad_images:

    print("\nBad image files:")

    for path, error in bad_images:
        print(
            f"{path} -> {error}"
        )

else:

    print("\nAll images are valid!")