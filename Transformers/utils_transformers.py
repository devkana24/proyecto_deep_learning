# utils.py
import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import shutil
from collections import defaultdict

# === reemplaza prepare_or_detect_splits y helpers relacionados ===
import shutil

def _is_already_split(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, d)) for d in ("train", "val", "test"))

def _list_class_files(flat_root: str):
    if not os.path.isdir(flat_root):
        raise FileNotFoundError(f"Ruta no existe: {flat_root}")
    classes = [d for d in os.listdir(flat_root) if os.path.isdir(os.path.join(flat_root, d))]
    if not classes:
        raise RuntimeError(f"No se encontraron clases dentro de: {flat_root}")
    data = {}
    for c in classes:
        cdir = os.path.join(flat_root, c)
        vids = [os.path.join(cdir, f) for f in os.listdir(cdir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        vids.sort()
        data[c] = vids
    return data

def create_simple_split(flat_root: str, dest_root: str,
                        train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, seed=42):
    os.makedirs(dest_root, exist_ok=True)
    rng = random.Random(seed)
    by_class = _list_class_files(flat_root)

    for split in ("train", "val", "test"):
        for c in by_class.keys():
            os.makedirs(os.path.join(dest_root, split, c), exist_ok=True)

    for c, files in by_class.items():
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:],
        }
        for split, lst in splits.items():
            for src in lst:
                dst = os.path.join(dest_root, split, c, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print(f"✅ Split creado en: {dest_root}")

def prepare_or_detect_splits(dataset_dir: str) -> str:
    dataset_dir = os.path.abspath(dataset_dir)
    suffix = "_splits"

    # 0) Ruta no existe: intentar usar la variante *_splits si existe
    if not os.path.exists(dataset_dir):
        maybe = dataset_dir + suffix
        if os.path.exists(maybe) and _is_already_split(maybe):
            print(f"ℹ️ Usando split existente: {maybe}")
            return maybe
        raise FileNotFoundError(
            f"Ruta '{dataset_dir}' no existe. "
            f"Usa la raíz plana (p.ej. 'Real_Life_Violence_Dataset') "
            f"o crea primero '{maybe}'."
        )

    # 1) Ya es un directorio con train/val/test
    if _is_already_split(dataset_dir):
        return dataset_dir

    # 2) Si ya termina en _splits pero está vacío/no tiene train/val/test
    if dataset_dir.endswith(suffix) and not _is_already_split(dataset_dir):
        original = dataset_dir[:-len(suffix)]
        if os.path.exists(original) and not _is_already_split(original):
            print(f"🧩 Dataset plano detectado en '{original}'. Creando split en '{dataset_dir}'…")
            create_simple_split(original, dataset_dir, 0.75, 0.15, 0.10, seed=42)
            return dataset_dir
        raise RuntimeError(
            f"'{dataset_dir}' no contiene train/val/test y no se encontró raíz plana '{original}'."
        )

    # 3) Directorio plano (clases adentro) → crear *_splits
    dest = dataset_dir + suffix
    if _is_already_split(dest):
        print(f"ℹ️ Usando split existente: {dest}")
        return dest
    print(f"🧩 Dataset plano detectado en '{dataset_dir}'. Creando split en '{dest}'…")
    create_simple_split(dataset_dir, dest, 0.75, 0.15, 0.10, seed=42)
    return dest

# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _resize_shorter_side(shorter: int):
    """
    Resize preservando aspecto: hace que el lado corto sea `shorter`.
    (Usamos torchvision.transforms.v2)
    """
    # v2.Resize recibe int => aplica al lado corto, preservando aspecto.
    return v2.Resize(shorter)

# =========================
# Temporal Index Builders
# =========================

def uniform_indices_full_video(total_frames: int, L: int) -> np.ndarray:
    """Indices uniformes que cubren TODO el video con exactamente L frames."""
    if total_frames <= 0:
        return np.zeros(L, dtype=int)
    if total_frames < L:
        # Si hay menos frames que L, repetimos uniformemente
        base = np.linspace(0, total_frames - 1, num=total_frames, dtype=int)
        rep = np.pad(base, (0, L - total_frames), mode='edge')
        return rep
    return np.linspace(0, total_frames - 1, num=L, dtype=int)

def sliding_window_starts(total_frames: int, L: int, overlap_ratio: float) -> List[int]:
    """Devuelve lista de 'starts' (frames iniciales) para ventanas deslizantes."""
    stride = max(1, int(L * (1.0 - overlap_ratio)))
    if total_frames <= L:
        return [0]
    starts = list(range(0, total_frames - L + 1, stride))
    if len(starts) == 0:
        starts = [max(0, total_frames - L)]
    return starts

def indices_for_clip_in_window(start: int, total_frames: int, L: int) -> np.ndarray:
    """
    Indices uniformes dentro de la ventana [start, start+L), acotando al final del video.
    """
    end = min(total_frames, start + L)
    if end - start < L:
        start = max(0, end - L)
    if total_frames <= 0:
        return np.zeros(L, dtype=int)
    idxs = np.linspace(start, end - 1, num=L, dtype=int)
    return idxs

# =========================
# Temporal Augmentations (opcionales)
# =========================

@dataclass
class TemporalAugCfg:
    use_jitter: bool = True
    jitter_max: int = 3           # +/- frames
    use_frame_drop: bool = True
    drop_prob: float = 0.12       # 8% aprox
    use_speed_perturb: bool = True
    sp_min: float = 0.85          # 0.85x
    sp_max: float = 1.20          # 1.20x

def apply_temporal_augs(
    idxs: np.ndarray,
    total_frames: int,
    L: int,
    cfg: Optional[TemporalAugCfg],
    train: bool
) -> np.ndarray:
    """Aplica jitter, frame-drop y speed-perturb sobre índices y re-muestrea a L."""
    if not train or cfg is None:
        return np.clip(idxs, 0, max(0, total_frames - 1))

    out = idxs.astype(float)

    # Speed perturbation (warping)
    if cfg.use_speed_perturb and total_frames > 1:
        sp = np.random.uniform(cfg.sp_min, cfg.sp_max)
        # Reescalar posiciones relativas
        rel = (out - out.min()) / (out.max() - out.min() + 1e-6)
        rel = np.clip(rel * sp, 0.0, 1.0)
        out = rel * (idxs.max() - idxs.min()) + idxs.min()

    # Jitter
    if cfg.use_jitter and cfg.jitter_max > 0:
        jitter = np.random.randint(-cfg.jitter_max, cfg.jitter_max + 1, size=out.shape)
        out = out + jitter

    # Frame drop + reinterpolado
    if cfg.use_frame_drop and cfg.drop_prob > 0.0 and L > 4:
        keep_mask = np.random.rand(out.size) > cfg.drop_prob
        kept = out[keep_mask]
        if kept.size < 2:  # asegurar al menos 2 puntos
            kept = out
        # re-sample a L
        pos = np.linspace(0, kept.size - 1, num=L)
        out = np.interp(pos, np.arange(kept.size), kept)

    out = np.clip(np.round(out).astype(int), 0, max(0, total_frames - 1))
    # Asegurar longitud exacta L
    if out.size != L:
        pos = np.linspace(0, out.size - 1, num=L)
        out = np.round(np.interp(pos, np.arange(out.size), out)).astype(int)
        out = np.clip(out, 0, max(0, total_frames - 1))

    return out

# =========================
# Dataset
# =========================

class VideoDatasetSlidingWindows(Dataset):
    """
    Dataset para clasificación de video con:
      - Muestreo temporal UNIFORME
      - Ventanas deslizantes con solape configurable
      - Preservación de aspecto vía Resize(short_side) + (RandomResized/Center)Crop
      - Aumentos temporales opcionales (jitter, drop, speed)
    Retorna tensores [C, T, H, W].
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 32,
        img_size: int = 224,
        mode: str = "train",
        overlap_ratio: float = 0.5,
        max_clips_per_video: Optional[int] = None,
        temporal_aug_cfg: Optional[TemporalAugCfg] = TemporalAugCfg(),
        preserve_aspect: bool = True,
    ):
        super().__init__()
        assert mode in ("train", "val", "test")
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.mode = mode
        self.overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.95))
        self.temporal_aug_cfg = temporal_aug_cfg
        self.preserve_aspect = preserve_aspect

        # Política de clips por video (conservadora por modo)
        if max_clips_per_video is None:
            self.max_clips_per_video = 4 if mode == "train" else (2 if mode == "val" else 1)
        else:
            self.max_clips_per_video = max(1, int(max_clips_per_video))

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Transforms (espacial)
        if self.preserve_aspect:
            # lado corto 256, crop 224
            if mode == "train":
                self.transform = v2.Compose([
                    _resize_shorter_side(256),
                    v2.RandomResizedCrop(self.img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                    v2.RandomHorizontalFlip(0.5),
                    v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = v2.Compose([
                    _resize_shorter_side(256),
                    v2.CenterCrop(self.img_size),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            # redimensiona directo a cuadrado (menos recomendado)
            if mode == "train":
                self.transform = v2.Compose([
                    v2.RandomHorizontalFlip(0.5),
                    v2.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                    v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = v2.Compose([
                    v2.Resize((self.img_size, self.img_size)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        self.samples: List[Tuple[str, int, int]] = []  # (video_path, label, clip_index)
        self._build_index()

    def _is_valid_video(self, path: str) -> bool:
        try:
            cap = cv2.VideoCapture(path)
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            cap.release()
            return (fc > 0 and fps > 0 and ret and frame is not None)
        except Exception:
            return False

    def _build_index(self):
        total_videos, valid_videos, total_clips = 0, 0, 0
        for cls in self.classes:
            pcls = os.path.join(self.root_dir, cls)
            videos = sorted(glob.glob(os.path.join(pcls, "*.mp4")))
            total_videos += len(videos)
            for vp in videos:
                if not self._is_valid_video(vp):
                    # Log mínimo para no ensuciar salida
                    continue
                valid_videos += 1
                # contamos cuántos clips generamos (máximo)
                cap = cv2.VideoCapture(vp)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                starts = sliding_window_starts(total_frames, self.num_frames, self.overlap_ratio)
                if len(starts) > self.max_clips_per_video:
                    starts = starts[:self.max_clips_per_video]

                for clip_idx in range(len(starts)):
                    self.samples.append((vp, self.class_to_idx[cls], clip_idx))
                    total_clips += 1

        # Mensajes
        print(f"[{self.mode}] videos totales={total_videos} | válidos={valid_videos} | clips={total_clips} | clips/válido≈{(total_clips / max(valid_videos,1)):.2f}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frames_by_indices(self, cap: cv2.VideoCapture, idxs: np.ndarray, H: int, W: int) -> List[torch.Tensor]:
        frames = []
        last_valid = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret or frame is None:
                # fallback: repetir último válido o negro
                if last_valid is not None:
                    frames.append(last_valid.clone())
                else:
                    frames.append(torch.zeros(3, H, W))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Ojo: el resize/crop lo hace self.transform (mejor preservación de aspecto)
            # Aquí convertimos a tensor CHW uint8 -> luego transform a float y norm
            ten = torch.from_numpy(frame).permute(2, 0, 1)  # [C,H,W], uint8
            # Aplicar transform espacial por frame
            ten = self.transform(ten)
            frames.append(ten)
            last_valid = ten
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label, clip_idx = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculamos ventanas y el start del clip deseado
        starts = sliding_window_starts(total_frames, self.num_frames, self.overlap_ratio)
        if len(starts) == 0:
            starts = [0]
        if clip_idx >= len(starts):
            clip_idx = len(starts) - 1
        start = starts[clip_idx]

        # Indices base uniformes en ventana
        idxs = indices_for_clip_in_window(start, total_frames, self.num_frames)

        # Aumentos temporales (solo en train)
        idxs = apply_temporal_augs(
            idxs, total_frames, self.num_frames,
            self.temporal_aug_cfg, train=(self.mode == "train")
        )

        # Leemos frames por indices
        # Para transform v2 centrada en PIL/float, ya hicimos ToDtype dentro transform.
        # Pero transform requiere tensor CHW en v2 (ok).
        # Aseguramos dimensiones finales consistentemente en transform.
        # (Aquí solo pasamos CHW uint8 y transform hace resize/crop/normalize)
        # Dummy size (no se usa, resize lo hace transform):
        H = W = self.img_size

        frames = self._read_frames_by_indices(cap, idxs, H, W)
        cap.release()

        clip = torch.stack(frames, dim=1)  # [C, T, H, W]
        return clip, label

# =========================
# DataLoaders
# =========================

def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 4,
    num_frames: int = 16,
    img_size: int = 224,
    overlap_ratio: float = 0.5,
    max_clips_train: int = 8,
    max_clips_val: int = 2,
    max_clips_test: int = 1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders consistentes para train/val/test.
    """
    set_seed(seed)
    print("⚙️ Creando DataLoaders (sliding windows + muestreo uniforme)…")

    # 👇 NUEVO: detectar/crear split si hace falta
    dataset_dir = prepare_or_detect_splits(dataset_dir)

    print("⚙️ Creando DataLoaders (sliding windows + muestreo uniforme)…")

    train_ds = VideoDatasetSlidingWindows(
        root_dir=os.path.join(dataset_dir, "train"),
        num_frames=num_frames,
        img_size=img_size,
        mode="train",
        overlap_ratio=overlap_ratio,
        max_clips_per_video=max_clips_train,
        temporal_aug_cfg=TemporalAugCfg(),      # ON en train
        preserve_aspect=True,
    )

    val_ds = VideoDatasetSlidingWindows(
        root_dir=os.path.join(dataset_dir, "val"),
        num_frames=num_frames,
        img_size=img_size,
        mode="val",
        overlap_ratio=overlap_ratio,
        max_clips_per_video=max_clips_val,
        temporal_aug_cfg=TemporalAugCfg(       # OFF (por train=False) pero pasamos objeto por consistencia
            use_jitter=False, use_frame_drop=False, use_speed_perturb=False
        ),
        preserve_aspect=True,
    )

    test_ds = VideoDatasetSlidingWindows(
        root_dir=os.path.join(dataset_dir, "test"),
        num_frames=num_frames,
        img_size=img_size,
        mode="test",
        overlap_ratio=overlap_ratio,
        max_clips_per_video=max_clips_test,
        temporal_aug_cfg=TemporalAugCfg(
            use_jitter=False, use_frame_drop=False, use_speed_perturb=False
        ),
        preserve_aspect=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"├─ Train: {len(train_ds)} clips ({len(train_loader)} batches)")
    print(f"├─ Val:   {len(val_ds)} clips ({len(val_loader)} batches)")
    print(f"└─ Test:  {len(test_ds)} clips ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader

# =========================
# Inference Aggregation Helper
# =========================

def aggregate_logits(logits_list: List[torch.Tensor], method: str = "mean") -> torch.Tensor:
    """
    Agrega logits de múltiples clips de un mismo video.
    - logits_list: lista de tensores [B] o [B,1]
    - method: "mean" o "max"
    Retorna tensor [B]
    """
    if len(logits_list) == 0:
        raise ValueError("logits_list vacío")
    xs = [x.view(-1) for x in logits_list]
    X = torch.stack(xs, dim=0)  # [num_clips, B]
    if method == "max":
        out, _ = X.max(dim=0)
    else:
        out = X.mean(dim=0)
    return out
