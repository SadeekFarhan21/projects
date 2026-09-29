"""Minimal readers for COLMAP sparse models (binary and text).

Only what the trainer needs: intrinsics, per-image world-to-camera pose,
and the sparse point cloud (xyz + rgb) used to initialise the Gaussians.
Poses are taken as given; no bundle adjustment happens here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
}
MODEL_BY_NAME = {v[0]: (k, v[1]) for k, v in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole(self) -> tuple[float, float, float, float]:
        """Return fx, fy, cx, cy. Distortion terms are ignored (the images
        we train on are assumed undistorted, as COLMAP's image_undistorter makes them)."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # w, x, y, z  (world -> camera rotation)
    tvec: np.ndarray  # world -> camera translation
    camera_id: int
    name: str

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def world_to_cam(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R()
        M[:3, 3] = self.tvec
        return M

    def center(self) -> np.ndarray:
        return -self.R().T @ self.tvec


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read(f, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))


def read_cameras_bin(path: Path) -> dict[int, Camera]:
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, model_id, w, h = _read(f, "iiQQ")
            name, npar = CAMERA_MODELS[model_id]
            params = np.array(_read(f, "d" * npar))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images_bin(path: Path) -> dict[int, Image]:
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            vals = _read(f, "idddddddi")
            iid = vals[0]
            q = np.array(vals[1:5])
            t = np.array(vals[5:8])
            cam_id = vals[8]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n2d,) = _read(f, "Q")
            f.seek(24 * n2d, 1)  # skip x, y, point3D_id per keypoint
            imgs[iid] = Image(iid, q, t, cam_id, name.decode())
    return imgs


def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        xyz = np.empty((n, 3), np.float64)
        rgb = np.empty((n, 3), np.uint8)
        for i in range(n):
            vals = _read(f, "QdddBBBd")
            xyz[i] = vals[1:4]
            rgb[i] = vals[4:7]
            (tl,) = _read(f, "Q")
            f.seek(8 * tl, 1)  # track: image_id, point2D_idx (int32 each)
    return xyz, rgb


def read_cameras_txt(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cams[int(e[0])] = Camera(int(e[0]), e[1], int(e[2]), int(e[3]), np.array(e[4:], float))
    return cams


def read_images_txt(path: Path) -> dict[int, Image]:
    imgs = {}
    lines = [l for l in open(path) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        e = lines[i].split()
        if not e:
            continue
        iid = int(e[0])
        imgs[iid] = Image(iid, np.array(e[1:5], float), np.array(e[5:8], float), int(e[8]), e[9])
    return imgs


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        xyz.append([float(v) for v in e[1:4]])
        rgb.append([int(v) for v in e[4:7]])
    return np.array(xyz, float).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def read_model(sparse_dir: Path):
    """Read cameras, images and points from a COLMAP sparse dir (bin or txt)."""
    d = Path(sparse_dir)
    if (d / "cameras.bin").exists():
        return (
            read_cameras_bin(d / "cameras.bin"),
            read_images_bin(d / "images.bin"),
            read_points3d_bin(d / "points3D.bin"),
        )
    return (
        read_cameras_txt(d / "cameras.txt"),
        read_images_txt(d / "images.txt"),
        read_points3d_txt(d / "points3D.txt"),
    )


def write_model_bin(sparse_dir: Path, cams: dict[int, Camera], imgs: dict[int, Image], xyz, rgb) -> None:
    """Writer used by tests to round-trip the binary format (no 2D keypoints, no tracks)."""
    d = Path(sparse_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams.values():
            mid, _ = MODEL_BY_NAME[c.model]
            f.write(struct.pack("<iiQQ", c.id, mid, c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *c.params))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for im in imgs.values():
            f.write(struct.pack("<idddddddi", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(struct.pack("<QdddBBBd", i + 1, *p, *[int(v) for v in c], 0.0))
            f.write(struct.pack("<Q", 0))
