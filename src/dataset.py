"""
dataset.py — Carregamento, Augmentation e Split de dados
Projeto: Classificação de Terreno por Imagens de Satélite
Disciplina: Applied Computer Vision — FIAP Global Solution 2026

Dataset recomendado: EuroSAT (RGB)
- 27.000 imagens 64x64, 10 classes de terreno
- Download: https://madm.dfki.de/files/sentinel/EuroSAT.zip
- Alternativa via torchvision: EuroSAT (requires torchvision >= 0.12)
  from torchvision.datasets import EuroSAT

Classes:
    0 - AnnualCrop       | 1 - Forest         | 2 - HerbaceousVegetation
    3 - Highway          | 4 - Industrial      | 5 - Pasture
    6 - PermanentCrop    | 7 - Residential     | 8 - River
    9 - SeaLake
"""

import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import EuroSAT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./data"
IMAGE_SIZE   = 64          # EuroSAT já vem em 64x64; ajuste se usar outro dataset
BATCH_SIZE   = 64
NUM_WORKERS  = 4           # Ajuste para o número de núcleos disponíveis
SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15        # deve somar 1.0 com TRAIN e VAL

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
# Estatísticas calculadas sobre o EuroSAT (RGB):
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD  = [0.2037, 0.1366, 0.1148]

def get_transforms(split: str) -> transforms.Compose:
    """
    Retorna transformações específicas por split.

    Treino  → augmentation agressivo (flip, rotação, color jitter, perspectiva)
    Val/Teste → apenas resize + normalização (sem augmentation para avaliação limpa)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # ── Data Augmentation ─────────────────────────────────────────────
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.75, 1.0),   # zoom entre 75% e 100%
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # ── Para tensor e normalização ─────────────────────────────────────
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])
    else:  # val e test — sem augmentation
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO E SPLIT
# ──────────────────────────────────────────────────────────────────────────────
def load_eurosat(dataset_root: str = DATASET_ROOT):
    """
    Carrega o EuroSAT via torchvision (faz download automático na 1ª execução).
    Divide em train/val/test mantendo estratificação por classe.
    
    Retorna:
        train_ds, val_ds, test_ds  →  objetos Dataset com transform correto
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Carrega dataset completo com transform de treino para calcular tamanhos
    full_dataset = EuroSAT(root=dataset_root, download=True)

    total = len(full_dataset)  # 27.000 imagens
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)
    n_test  = total - n_train - n_val

    print(f"  Total de imagens : {total:,}")
    print(f"  Treino           : {n_train:,} ({TRAIN_RATIO:.0%})")
    print(f"  Validação        : {n_val:,}   ({VAL_RATIO:.0%})")
    print(f"  Teste            : {n_test:,}   ({TEST_RATIO:.0%})")

    # Índices embaralhados para divisão aleatória
    indices = torch.randperm(total).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # Cria três datasets com transforms distintos usando Subset + wrapper
    train_ds = _DatasetWithTransform(full_dataset, train_idx, get_transforms("train"))
    val_ds   = _DatasetWithTransform(full_dataset, val_idx,   get_transforms("val"))
    test_ds  = _DatasetWithTransform(full_dataset, test_idx,  get_transforms("test"))

    return train_ds, val_ds, test_ds


class _DatasetWithTransform(torch.utils.data.Dataset):
    """
    Wrapper que aplica transform específico a um subconjunto de índices,
    permitindo transforms diferentes para train/val/test sobre o MESMO dataset base.
    """
    def __init__(self, base_dataset, indices, transform):
        self.base      = base_dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        # EuroSAT retorna PIL Image; aplica transform
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADERS
# ──────────────────────────────────────────────────────────────────────────────
def get_dataloaders(dataset_root: str = DATASET_ROOT):
    """
    Cria e retorna DataLoaders para train, val e test.
    
    Configurações de performance:
      - pin_memory=True  → transferência CPU→GPU mais rápida (essencial com RTX 3060)
      - persistent_workers=True → mantém workers vivos entre epochs
      - prefetch_factor=2  → pré-busca de 2 batches por worker
    """
    train_ds, val_ds, test_ds = load_eurosat(dataset_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        drop_last=True,   # evita batch pequeno na última iteração
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print(f"\n  Batches (treino)    : {len(train_loader)}")
    print(f"  Batches (validação) : {len(val_loader)}")
    print(f"  Batches (teste)     : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS DE VISUALIZAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
def show_sample_images(loader: DataLoader, n: int = 16, save_path: str = None):
    """Plota uma grade de amostras do dataset com os rótulos reais."""
    images, labels = next(iter(loader))

    # De-normaliza para visualização
    mean = torch.tensor(EUROSAT_MEAN).view(3, 1, 1)
    std  = torch.tensor(EUROSAT_STD).view(3, 1, 1)
    images_vis = images[:n].cpu() * std + mean
    images_vis = images_vis.clamp(0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Amostras do Dataset EuroSAT — Terrenos por Satélite", fontsize=14, y=1.01)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img_np = images_vis[i].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(CLASS_NAMES[labels[i].item()], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figura salva em: {save_path}" if save_path else "")


def show_class_distribution(loader: DataLoader, split_name: str, save_path: str = None):
    """Plota distribuição de classes para verificar balanceamento."""
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy().tolist())

    counter = Counter(all_labels)
    counts  = [counter[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_NAMES, counts, color="#2E86AB", edgecolor="white", linewidth=0.8)
    ax.set_title(f"Distribuição de Classes — {split_name}", fontsize=13)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de Imagens")
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  Carregando EuroSAT e criando DataLoaders...")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_dataloaders()
    show_sample_images(train_loader, save_path="outputs/sample_images.png")
    show_class_distribution(train_loader, "Treino", save_path="outputs/class_dist_train.png")
    print("\nDataset pronto!")
