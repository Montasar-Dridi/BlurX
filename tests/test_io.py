import os
from pathlib import Path
import numpy as np
import cv2
import pytest

# 👉 change this to your actual module path
from blurx.utils.io import list_image_paths, load_image, iter_images


def _make_image(p: Path, shape=(16, 12, 3)) -> Path:
    """Create a simple image on disk and return its path."""
    p.parent.mkdir(parents=True, exist_ok=True)
    img = np.full(shape, 127, dtype=np.uint8)
    ok = cv2.imwrite(str(p), img)
    assert ok, f"Failed to write image: {p}"
    return p


def test_list_image_paths_directory_basic(tmp_path: Path):
    # create images and a junk file
    p1 = _make_image(tmp_path / "a.jpg")
    p2 = _make_image(tmp_path / "b.png")
    (tmp_path / "notes.txt").write_text("not an image")

    paths = list_image_paths(tmp_path)
    assert paths == sorted([str(p1), str(p2)])


def test_list_image_paths_single_file_ok(tmp_path: Path):
    p = _make_image(tmp_path / "only.jpg")
    paths = list_image_paths(p)
    assert paths == [str(p)]


def test_list_image_paths_unsupported_file_raises(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("hello")
    with pytest.raises(ValueError):
        _ = list_image_paths(p)


def test_list_image_paths_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ = list_image_paths(tmp_path)


def test_list_image_paths_nonexistent_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ = list_image_paths(tmp_path / "nope")


def test_list_image_paths_recursive_and_sorted(tmp_path: Path):
    # nested structure
    p1 = _make_image(tmp_path / "z" / "img2.png")
    p2 = _make_image(tmp_path / "a" / "img1.jpg")
    paths = list_image_paths(tmp_path)
    assert paths == sorted([str(p1), str(p2)]), (
        "Paths should be sorted lexicographically"
    )


def test_load_image_reads_bgr_uint8(tmp_path: Path):
    p = _make_image(tmp_path / "pic.jpg", shape=(10, 8, 3))
    img = load_image(str(p))
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (10, 8, 3)


def test_load_image_unreadable_raises(tmp_path: Path):
    # zero-byte file with image extension causes cv2.imread to return None
    p = tmp_path / "corrupt.jpg"
    p.touch()
    with pytest.raises(ValueError):
        _ = load_image(str(p))


def test_iter_images_yields_path_and_image(tmp_path: Path):
    p1 = _make_image(tmp_path / "a.jpg", shape=(9, 7, 3))
    p2 = _make_image(tmp_path / "b.png", shape=(12, 10, 3))
    got = list(iter_images([str(p1), str(p2)]))

    assert len(got) == 2
    (gp1, img1), (gp2, img2) = got
    assert gp1 == str(p1) and gp2 == str(p2)
    assert img1.shape == (9, 7, 3)
    assert img2.shape == (12, 10, 3)


def test_iter_images_is_lazy(tmp_path: Path):
    """Prove the generator only loads when advanced."""
    good = _make_image(tmp_path / "good.jpg")
    bad = tmp_path / "bad.jpg"
    bad.touch()  # unreadable image (zero bytes)

    gen = iter_images([str(good), str(bad)])

    # First next() should succeed (loads only 'good.jpg')
    path1, img1 = next(gen)
    assert path1 == str(good)
    assert img1 is not None

    # Consuming the rest should raise on 'bad.jpg'
    with pytest.raises(ValueError):
        list(gen)
