from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Type, List, Union
import numpy as np
from .enums import GraphicObservationColorMap
import socket
from contextlib import closing


@dataclass(frozen=True)
class ColorMapEncoder:
    palette: np.ndarray  # (K,3) uint8
    names: Tuple[str, ...]  # length K
    name2id: Dict[str, int]
    unknown_id: int

    @classmethod
    def from_enum(
        cls,
        enum_cls: Type[GraphicObservationColorMap] = GraphicObservationColorMap,
        *,
        unknown_to: str | int = "BLANK",
    ) -> "ColorMapEncoder":

        items = [(k, v) for k, v in enum_cls.__dict__.items() if k.isupper()]
        names = tuple(k for k, _ in items)
        palette = np.asarray([v for _, v in items], dtype=np.uint8)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError("Palette must be shape (K,3) of uint8 RGB.")
        name2id = {name: i for i, name in enumerate(names)}

        if isinstance(unknown_to, str):
            if unknown_to not in name2id:
                raise KeyError(f"unknown_to='{unknown_to}' not in {list(name2id)}")
            unknown_id = name2id[unknown_to]
        else:
            unknown_id = int(unknown_to)
            if not (0 <= unknown_id < len(names)):
                raise ValueError(f"unknown_to id out of range 0..{len(names)-1}")

        return cls(palette=palette, names=names, name2id=name2id, unknown_id=unknown_id)

    def encode_ids(self, img: np.ndarray) -> np.ndarray:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)
        id_map = matches.argmax(axis=-1).astype(np.uint8)
        if not matched_any.all():
            id_map[~matched_any] = np.uint8(self.unknown_id)
        return id_map

    def encode_onehot(self, img: np.ndarray) -> np.ndarray:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)
        onehot = matches.astype(np.uint8).transpose(2, 0, 1)  # (K,H,W)
        if not matched_any.all():
            onehot[:, ~matched_any] = 0
            onehot[self.unknown_id, ~matched_any] = 1
        return onehot

    def encode(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_hwc = self._to_hwc(img)
        matches = (img_hwc[..., None, :] == self.palette[None, None, :, :]).all(
            axis=-1
        )  # (H,W,K)
        matched_any = matches.any(axis=-1)

        id_map = matches.argmax(axis=-1).astype(np.uint8)
        if not matched_any.all():
            id_map[~matched_any] = np.uint8(self.unknown_id)

        onehot = matches.astype(np.uint8).transpose(2, 0, 1)  # (K,H,W)
        if not matched_any.all():
            onehot[:, ~matched_any] = 0
            onehot[self.unknown_id, ~matched_any] = 1

        return id_map, onehot

    @staticmethod
    def _to_hwc(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise AssertionError("expected 3D array")
        if img.shape[0] == 3 and img.shape[-1] != 3:
            # (3,H,W) -> (H,W,3)
            img = np.moveaxis(img, 0, -1)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return img


DEFAULT_ENCODER = ColorMapEncoder.from_enum()

# (y, x0, x1) where platform (2.00) must exist.
# Note: x1 is inclusive.
PLATFORM_FORCE_SPANS = [
    (29,  96, 105),
    (38,  54,  75),
    (45,  24,  34),
    (53,  54,  65),
    (53, 128, 143),
    (64,  35,  45),
    (69,  68, 113),
]

def force_fix_platform_inplace(id_map: np.ndarray) -> None:
    """
    Fast in-place fix.
    For each span, replace only 0.00 -> 2.00.
    Supports id_map shape (90,160) or (160,90).
    """
    id_map = np.asarray(id_map)

    # normalize to (90,160) indexing [y,x]
    if id_map.shape == (160, 90):
        id_map = id_map.T  # view

    if id_map.shape != (90, 160):
        raise ValueError(f"[platform-fix] unexpected id_map shape: {id_map.shape}")

    # vectorized per-span replacement (no inner python loop over x)
    for y, x0, x1 in PLATFORM_FORCE_SPANS:
        seg = id_map[y, x0:x1 + 1]
        # only overwrite true blanks
        seg[seg == 0.00] = 2.00


def fix_obs_structure(obs, id_map=None):
    """
    obs: (90*160*3,) 처럼 납작하게 들어온 RGB를
         (3, 90, 160) CHW로 복구.

    id_map: (90, 160) 이 같이 들어오면 platform 누락을 하드픽스까지 수행.
    """
    h, w = 90, 160

    # 1) (H, W, 3)
    corrected_hwc = obs.reshape(h, w, 3)

    # 2) (3, H, W)
    corrected_chw = np.transpose(corrected_hwc, (2, 0, 1))
    return corrected_chw

def colormap_to_ids_and_onehot(img_chw: np.ndarray):
    img_chw = fix_obs_structure(img_chw)
    id_map, onehot = DEFAULT_ENCODER.encode(img_chw)
    force_fix_platform_inplace(id_map)
    return id_map, onehot


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


def find_n_free_tcp_ports(n: int, host: str = "127.0.0.1") -> List[int]:
    ports = []
    for _ in range(n):
        ports.append(find_free_tcp_port(host))
    if len(set(ports)) != len(ports):
        return find_n_free_tcp_ports(n, host)
    return ports


def parse_observation(
    observation: Union[Dict[str, np.ndarray], List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(observation, dict):
        graphic_obs = observation["graphic"]
        vector_obs = observation["vector"]
    elif isinstance(observation, list) and len(observation) == 2:
        graphic_obs, vector_obs = observation
    else:
        raise ValueError("Invalid observation format")

    return graphic_obs, vector_obs

__all__ = [
    "ColorMapEncoder",
    "colormap_to_ids_and_onehot",
    "DEFAULT_ENCODER",
    "find_free_tcp_port",
    "find_n_free_tcp_ports",
]
