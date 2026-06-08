"""
models.py — Duas Arquiteturas CNN do Zero
Projeto: Classificação de Terreno por Imagens de Satélite
Disciplina: Applied Computer Vision — FIAP Global Solution 2026

ARQUITETURA 1: TerraNet-Shallow (ampla e rasa)
  Foco em LARGURA: muitos filtros por camada, poucas camadas convolucionais.
  Captação de padrões de textura e cor de baixa frequência — eficiente para
  terrenos com cores dominantes (oceano, floresta, área urbana).

ARQUITETURA 2: TerraNet-Deep (profunda e estreita)
  Foco em PROFUNDIDADE: muitas camadas convolucionais com filtros menores.
  Captação de padrões hierárquicos e espaciais de alta frequência — eficiente
  para terrenos com detalhes estruturais finos (rodovias, rios, plantações).

DIFERENÇAS TÉCNICAS INTENCIONAIS:
──────────────────────────────────────────────────────────────────────────────
|  Aspecto              | TerraNet-Shallow    | TerraNet-Deep         |
|-----------------------|---------------------|-----------------------|
| Blocos conv           | 3 blocos            | 5 blocos              |
| Filtros por bloco     | 64 → 128 → 256      | 32 → 64 → 128 → 256 → 256 |
| Kernel size           | 5x5 e 3x3           | 3x3 consistente       |
| Pooling               | MaxPool 2x2         | MaxPool + AvgPool mix |
| Batch Normalization   | Sim                 | Sim                   |
| Dropout conv          | Não                 | Sim (SpatialDropout)  |
| Dropout FC            | 0.5                 | 0.4                   |
| Camadas FC            | 2 (512 → 10)        | 3 (512 → 256 → 10)    |
| Parâmetros aprox.     | ~3.8M               | ~2.1M                 |
──────────────────────────────────────────────────────────────────────────────

JUSTIFICATIVA TÉCNICA:
  A comparação entre redes LARGAS (mais filtros, menos profundidade) e PROFUNDAS
  (menos filtros, mais profundidade) é um tema clássico em arquitetura de CNNs
  (LeCun et al., 1989; Simonyan et al., 2014 — VGGNet).

  Hipótese H1: TerraNet-Shallow será mais eficiente em terrenos com padrões
    de cor/textura simples (oceano azul, floresta verde homogênea).
  Hipótese H2: TerraNet-Deep será mais eficiente em terrenos com padrões
    estruturais complexos (rodovias finas, cursos d'água sinuosos).

  A Matriz de Confusão e a análise por classe revelarão qual hipótese se confirma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# ──────────────────────────────────────────────────────────────────────────────
# BLOCO DE CONSTRUÇÃO REUTILIZÁVEL
# ──────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """
    Bloco conv padrão: Conv2d → BatchNorm → ReLU → [Dropout2d opcional]
    
    BatchNorm é essencial aqui pois:
      - Estabiliza o gradiente em redes profundas (reduz Internal Covariate Shift)
      - Age como regularizador implícito
      - Permite learning rates maiores → treino mais rápido
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, use_dropout: bool = False, dropout_p: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_p) if use_dropout else None

    def forward(self, x):
        x = self.block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# ARQUITETURA 1: TerraNet-Shallow (AMPLA e RASA)
# ──────────────────────────────────────────────────────────────────────────────
class TerraNetShallow(nn.Module):
    """
    Rede CNN larga e rasa — 3 blocos convolucionais, filtros maiores.

    Inspiração: conceito de "width over depth" explorado em redes como
    GoogLeNet Inception modules — mais filtros por camada captam mais
    "aspectos visuais simultâneos" antes de reduzir dimensionalidade.

    Arquitetura:
      Entrada: (B, 3, 64, 64)
      ┌─ Bloco 1 ────────────────────────────────────────────────┐
      │  Conv(3→64, k=5, p=2) → BN → ReLU                       │
      │  Conv(64→64, k=3, p=1) → BN → ReLU                      │  ← dupla conv
      │  MaxPool(2x2) → (B, 64, 32, 32)                         │
      └──────────────────────────────────────────────────────────┘
      ┌─ Bloco 2 ────────────────────────────────────────────────┐
      │  Conv(64→128, k=3, p=1) → BN → ReLU                     │
      │  Conv(128→128, k=3, p=1) → BN → ReLU                    │
      │  MaxPool(2x2) → (B, 128, 16, 16)                        │
      └──────────────────────────────────────────────────────────┘
      ┌─ Bloco 3 ────────────────────────────────────────────────┐
      │  Conv(128→256, k=3, p=1) → BN → ReLU                    │
      │  Conv(256→256, k=3, p=1) → BN → ReLU                    │
      │  MaxPool(2x2) → (B, 256, 8, 8)                          │
      └──────────────────────────────────────────────────────────┘
      AdaptiveAvgPool(4x4) → (B, 256, 4, 4)
      Flatten → (B, 4096)
      FC(4096→512) → ReLU → Dropout(0.5)
      FC(512→10)   → Logits
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Bloco 1 — kernel 5x5 no início para campo receptivo maior nas primeiras camadas
        self.block1 = nn.Sequential(
            ConvBlock(3,  64, kernel=5, padding=2),    # campo receptivo mais amplo
            ConvBlock(64, 64, kernel=3, padding=1),    # refinamento
            nn.MaxPool2d(kernel_size=2, stride=2),     # 64 → 32
        )

        # Bloco 2 — dobra canais
        self.block2 = nn.Sequential(
            ConvBlock(64,  128, kernel=3, padding=1),
            ConvBlock(128, 128, kernel=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 32 → 16
        )

        # Bloco 3 — dobra canais novamente
        self.block3 = nn.Sequential(
            ConvBlock(128, 256, kernel=3, padding=1),
            ConvBlock(256, 256, kernel=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 16 → 8
        )

        # Pooling adaptativo: garante saída fixa independente do tamanho de entrada
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 8 → 4

        # Cabeça classificadora
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                         # regularização forte na FC
            nn.Linear(512, num_classes),
        )

        # Inicialização de pesos (He/Kaiming para ReLU)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        Inicialização de Kaiming (He et al., 2015):
        Adequada para ReLU — mantém variância do sinal ao longo das camadas,
        evitando vanishing/exploding gradients em redes profundas.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# ARQUITETURA 2: TerraNet-Deep (PROFUNDA e ESTREITA)
# ──────────────────────────────────────────────────────────────────────────────
class TerraNetDeep(nn.Module):
    """
    Rede CNN profunda e estreita — 5 blocos convolucionais, filtros menores.

    Inspiração: VGGNet (Simonyan & Zisserman, 2014) — "profundidade beneficia
    a representação de padrões hierárquicos complexos".

    DIFERENÇAS CHAVE em relação à TerraNet-Shallow:
      1. 5 blocos vs. 3 blocos (maior profundidade)
      2. SpatialDropout2d nos blocos conv (regularização no espaço de features)
      3. Mix de MaxPool + AvgPool (MaxPool para bordas, AvgPool para texturas)
      4. 3 camadas FC (mais capacidade de classificação após features)
      5. Começa com apenas 32 filtros (menos parâmetros no início = menos overfitting)

    Arquitetura:
      Entrada: (B, 3, 64, 64)
      Bloco 1: Conv(3→32)  + Pool → (B, 32, 32, 32)
      Bloco 2: Conv(32→64) + Pool → (B, 64, 16, 16)
      Bloco 3: Conv(64→128)+ Pool → (B, 128, 8, 8)
      Bloco 4: Conv(128→256)      → (B, 256, 8, 8)    ← sem pool aqui
      Bloco 5: Conv(256→256)+ Pool→ (B, 256, 4, 4)
      AdaptiveAvgPool(2x2) → (B, 256, 2, 2)
      FC: 1024 → 512 → 256 → 10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Bloco 1 — entrada estreita, MaxPool para preservar bordas
        self.block1 = nn.Sequential(
            ConvBlock(3,  32, kernel=3, padding=1, use_dropout=True, dropout_p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 64 → 32
        )

        # Bloco 2 — cresce para 64 canais
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel=3, padding=1, use_dropout=True, dropout_p=0.05),
            ConvBlock(64, 64, kernel=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32 → 16
        )

        # Bloco 3 — cresce para 128 canais; AvgPool suaviza texturas
        self.block3 = nn.Sequential(
            ConvBlock(64,  128, kernel=3, padding=1, use_dropout=True, dropout_p=0.1),
            ConvBlock(128, 128, kernel=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),      # 16 → 8 (suaviza para texturas)
        )

        # Bloco 4 — cresce para 256 SEM pooling (mantém resolução espacial)
        # Isso permite que a rede aprenda padrões de maior complexidade nessa escala
        self.block4 = nn.Sequential(
            ConvBlock(128, 256, kernel=3, padding=1, use_dropout=True, dropout_p=0.1),
            ConvBlock(256, 256, kernel=1, padding=0),   # conv 1x1 = projeção de canais
        )

        # Bloco 5 — final com MaxPool (bordas críticas para classes como Highway/River)
        self.block5 = nn.Sequential(
            ConvBlock(256, 256, kernel=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 8 → 4
        )

        # Pooling adaptativo para garantir saída fixada
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # 4 → 2

        # Cabeça classificadora mais profunda — 3 camadas FC
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Conta apenas parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_comparison(num_classes: int = 10):
    """Imprime sumário e comparação dos dois modelos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(1, 3, 64, 64).to(device)

    print("=" * 70)
    print("  COMPARAÇÃO DAS ARQUITETURAS")
    print("=" * 70)

    for name, ModelClass in [("TerraNet-Shallow", TerraNetShallow),
                              ("TerraNet-Deep",    TerraNetDeep)]:
        model = ModelClass(num_classes).to(device)
        params = count_parameters(model)
        out = model(dummy)

        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(f"  Parâmetros treináveis : {params:,}")
        print(f"  Shape de entrada      : {dummy.shape}")
        print(f"  Shape de saída        : {out.shape}")

        # Sumário detalhado via torchinfo
        try:
            summary(model, input_size=(1, 3, 64, 64), verbose=0,
                    col_names=["input_size", "output_size", "num_params", "trainable"])
        except ImportError:
            print("  (instale torchinfo para sumário detalhado: pip install torchinfo)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_model_comparison()
