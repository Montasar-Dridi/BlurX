from pathlib import Path
import cv2


# Internal Function | Used by iter_images function.
def list_image_paths(path):
    """Return a list of image file paths from a single file or directory"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    image_path_list = []
    file_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if p.is_file():
        if p.suffix.lower() in file_extensions:
            image_path_list.append(str(p))
            return image_path_list
        else:
            raise ValueError(f"File is not a supported image format: {p}")

    for file_path in p.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            image_path_list.append(str(file_path))

    image_path_list.sort()

    if not image_path_list:
        raise FileNotFoundError(f"No images found in directory: {p}")

    return image_path_list


# Internal Function | Used by iter_images function.
def load_image(path):
    """Load a single image and return a Numpy array"""
    image_array = cv2.imread(path, cv2.IMREAD_COLOR)

    if image_array is None:
        raise ValueError(f"Failed to load image: {path}")

    return image_array


def iter_images(path):
    """Read images one at a time. Keepa the memory usage low"""
    images_list = list_image_paths(path)
    for image_path in images_list:
        yield image_path, load_image(image_path)
