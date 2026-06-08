"""
train.py — Loop de Treinamento Otimizado para GPU (RTX 3060 12GB)
Projeto: Classificação de Terreno por Imagens de Satélite
Disciplina: Applied Computer Vision — FIAP Global Solution 2026

Otimizações para RTX 3060 12GB:
  1. torch.cuda.amp (Automatic Mixed Precision, FP16) → +30-50% throughput
  2. torch.backends.cudnn.benchmark = True → CUDA escolhe algoritmo ótimo
  3. pin_memory nos DataLoaders → transferência CPU→GPU assíncrona
  4. Gradient clipping → estabilidade do treino
  5. OneCycleLR scheduler → convergência mais rápida e suave
  6. EarlyStopping customizado → interrompe ao atingir plateau de val_loss
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem display para ambientes headless
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, NUM_CLASSES, CLASS_NAMES
from src.models  import TerraNetShallow, TerraNetDeep, count_parameters

# ──────────────────────────────────────────────────────────────────────────────
# HIPERPARÂMETROS
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS        = 60        # EarlyStopping geralmente para antes
LR            = 3e-3      # Learning rate inicial (otimizado para OneCycleLR)
WEIGHT_DECAY  = 1e-4      # L2 regularização no optimizer
GRAD_CLIP     = 1.0       # Clip de norma do gradiente
PATIENCE      = 10        # EarlyStopping: epochs sem melhora antes de parar
MIN_DELTA     = 1e-4      # Mínima melhora considerada relevante pelo ES
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Para o treinamento quando val_loss para de melhorar.

    Salva o melhor estado do modelo automaticamente (best checkpoint).
    Estratégia: monitorar val_loss (e não val_acc) para detectar overfitting
    com mais sensibilidade — a loss captura incerteza nas predições,
    enquanto a acurácia pode mascarar degradação gradual.
    """
    def __init__(self, patience: int, min_delta: float, checkpoint_path: str):
        self.patience   = patience
        self.min_delta  = min_delta
        self.checkpoint = checkpoint_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE TREINO E VALIDAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    scheduler=None
) -> tuple[float, float]:
    """
    Executa uma epoch de treino com Automatic Mixed Precision (AMP).

    AMP (FP16 + FP32):
      - Operações de forward pass em FP16 → menor uso de memória, CUDA cores mais rápidos
      - GradScaler previne underflow de gradientes em FP16
      - Operações críticas (BN, softmax) mantidas em FP32 automaticamente
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # non_blocking + pin_memory = async
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none=True libera memória vs zero_

        # Forward em precisão mista (FP16)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        # Backward com GradScaler (corrige escala dos gradientes FP16)
        scaler.scale(loss).backward()

        # Gradient clipping: previne explosão de gradiente
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()  # OneCycleLR atualiza a cada step (não epoch)

        # Métricas
        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss    / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Avaliação sem gradientes — mais rápido e sem consumo extra de memória."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ──────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL DE TREINAMENTO
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Treina um modelo e retorna histórico completo de métricas.

    Estratégia de otimização:
      - AdamW: Adam com decoupled weight decay (mais estável que Adam puro)
      - OneCycleLR: começa baixo, sobe até LR_max, desce suavemente
        → alcança convergência mais rápida e generalizações melhores
        (Smith & Touvron, 2018 — "Super-Convergence")
    """
    print(f"\n{'═'*60}")
    print(f"  Treinando: {model_name}")
    print(f"  Parâmetros: {count_parameters(model):,}")
    print(f"  Device: {device}")
    print(f"{'═'*60}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # label_smoothing=0.1: suaviza os alvos de [0,1] para [0.01, 0.91]
    # → reduz overconfidence, melhora calibração e generalização

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # OneCycleLR: aquecer até pct_start*epochs, depois decair com cosseno
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,          # 15% das steps = warmup
        anneal_strategy="cos",   # decaimento por cosseno (suave)
        div_factor=25.0,         # LR inicial = max_lr / 25
        final_div_factor=1e4,    # LR final = max_lr / 10000
    )

    scaler = torch.amp.GradScaler('cuda')  

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    early_stopping  = EarlyStopping(PATIENCE, MIN_DELTA, checkpoint_path)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         []
    }

    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - t0

        # Log
        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # EarlyStopping
        if early_stopping(val_loss, model):
            print(f"\n  EarlyStopping ativado na epoch {epoch}.")
            print(f"  Melhor val_loss: {early_stopping.best_loss:.4f}")
            break

    total_time = time.time() - start_time
    print(f"\n  Treino concluído em {total_time/60:.1f} min")
    print(f"  Melhor val_acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Carrega melhor checkpoint para avaliação final
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Salva histórico em JSON
    hist_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS DE CURVAS DE APRENDIZADO
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    history1: dict, name1: str,
    history2: dict, name2: str,
    save_path: str = "outputs/training_curves.png"
):
    """
    Plota curvas de Acurácia e Loss para os dois modelos lado a lado.
    4 subplots: Train Acc | Val Acc | Train Loss | Val Loss
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Curvas de Aprendizado — Comparação dos Modelos", fontsize=14)

    colors = {"Shallow": ("#2E86AB", "#A8DADC"), "Deep": ("#E63946", "#F4A261")}
    
    pairs = [
        (axes[0, 0], "train_acc",  "Acurácia — Treino",     "Acurácia"),
        (axes[0, 1], "val_acc",    "Acurácia — Validação",  "Acurácia"),
        (axes[1, 0], "train_loss", "Loss — Treino",         "Loss"),
        (axes[1, 1], "val_loss",   "Loss — Validação",      "Loss"),
    ]

    for ax, key, title, ylabel in pairs:
        for hist, name, color_pair in [(history1, name1, colors["Shallow"]),
                                        (history2, name2, colors["Deep"])]:
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], color=color_pair[0],
                    linewidth=2, label=name, alpha=0.9)
            # Suavização para visualização (média móvel)
            if len(hist[key]) > 5:
                smooth = np.convolve(hist[key], np.ones(5)/5, mode="valid")
                ax.plot(range(3, len(smooth)+3), smooth,
                        color=color_pair[1], linewidth=1, linestyle="--", alpha=0.6)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curvas salvas em: {save_path}")


def plot_lr_schedule(history: dict, model_name: str, save_path: str = None):
    """Plota a curva do learning rate ao longo das epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["lr"], color="#2E86AB", linewidth=2)
    ax.set_title(f"Learning Rate Schedule — {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Configuração de GPU ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # cudnn.benchmark: CUDA testa e escolhe o algoritmo de conv mais rápido
        # para o tamanho de input fixo — ganho de 10-30% no throughput
        torch.backends.cudnn.benchmark   = True
        torch.backends.cudnn.deterministic = False  # mais rápido (menos reprodutível)
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU detectada: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n  AVISO: GPU não detectada. Treinando em CPU (lento).")
        print("  Certifique-se de ter instalado: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("\nCarregando dataset...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Treina Modelo 1 ──────────────────────────────────────────────────────
    model1 = TerraNetShallow(num_classes=NUM_CLASSES)
    hist1  = train_model(model1, "TerraNetShallow", train_loader, val_loader, device)

    # ── Treina Modelo 2 ──────────────────────────────────────────────────────
    model2 = TerraNetDeep(num_classes=NUM_CLASSES)
    hist2  = train_model(model2, "TerraNetDeep", train_loader, val_loader, device)

    # ── Plots comparativos ───────────────────────────────────────────────────
    plot_training_curves(hist1, "TerraNet-Shallow", hist2, "TerraNet-Deep")
    plot_lr_schedule(hist1, "TerraNet-Shallow", "outputs/lr_shallow.png")
    plot_lr_schedule(hist2, "TerraNet-Deep",    "outputs/lr_deep.png")

    print("\nTreinamento completo! Execute evaluate.py para comparação final.")
