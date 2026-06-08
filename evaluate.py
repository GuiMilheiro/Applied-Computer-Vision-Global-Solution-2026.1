# evaluate.py
import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.models import TerraNetShallow, TerraNetDeep

def evaluate_model(model_class, model_name, test_loader, device):
    print(f"\n{'='*50}")
    print(f" Avaliando: {model_name}")
    print(f"{'='*50}")
    
    model = model_class(num_classes=NUM_CLASSES).to(device)
    weights_path = os.path.join("checkpoints", f"{model_name}_best.pt")
    
    if not os.path.exists(weights_path):
        print(f"Erro: Pesos não encontrados em {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 1. Classification Report (Acurácia, Precision, Recall, F1)
    print("\nRelatório de Classificação:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    
    # 2. Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/cm_{model_name}.png", dpi=150)
    print(f"\nMatriz de confusão salva em outputs/cm_{model_name}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()
    
    # Avalia as duas arquiteturas no conjunto de Teste
    evaluate_model(TerraNetShallow, "TerraNetShallow", test_loader, device)
    evaluate_model(TerraNetDeep, "TerraNetDeep", test_loader, device)