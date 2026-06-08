import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd

# Importações do seu projeto
from src.dataset import CLASS_NAMES
from src.models import TerraNetShallow

# 1. Configuração da Página (O segredo para não scrollar)
st.set_page_config(page_title="TerraNet | FIAP", page_icon="🛰️", layout="wide")

# Customização CSS leve para deixar os cards mais bonitos
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .big-font { font-size:24px !important; font-weight: bold; color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

# 2. Carregando o Modelo (com cache para ficar super rápido)
@st.cache_resource
def load_model():
    model = TerraNetShallow(num_classes=len(CLASS_NAMES))
    # Ajuste o caminho se necessário
    model.load_state_dict(torch.load("checkpoints/TerraNetShallow_best.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Transformação idêntica ao treino
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Ajuste se usou outra normalização
])

# ==========================================
# BARRA LATERAL (Centro de Comando)
# ==========================================
with st.sidebar:
    st.markdown("### 🕹️ Centro de Comando")
    st.info("Faça o upload de uma captura orbital para iniciar a varredura neural.")
    
    uploaded_file = st.file_uploader("Injetar Dados Orbitais (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.markdown("### 📖 Manual do Sistema")
    st.caption("O modelo TerraNetShallow filtra texturas e analisa borrões de cor para determinar a classe do terreno com precisão.")

# ==========================================
# TELA PRINCIPAL
# ==========================================
st.title("🛰️ TerraNet: Monitoramento de Solo")
st.markdown("Motor de Inteligência Artificial para classificação de dados do satélite Sentinel-2.")

if uploaded_file is not None:
    # Lendo a imagem
    image = Image.open(uploaded_file).convert("RGB")
    
    # Criando 3 colunas para colocar tudo lado a lado
    col1, col2, col3 = st.columns([1, 1.2, 1.5])
    
    # Pré-processamento e predição
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        
    class_name = CLASS_NAMES[predicted_class.item()]
    conf_percent = confidence.item() * 100

    # --- COLUNA 1: A Imagem ---
    with col1:
        st.markdown("#### 📷 Sensor Óptico")
        st.image(image, use_container_width=True, caption="Alvo Fixado")

    # --- COLUNA 2: O Diagnóstico ---
    with col2:
        st.markdown("#### 🧠 Diagnóstico Neural")
        if conf_percent > 85:
            st.success(f"PREDIÇÃO DA IA: {class_name}")
            status_color = "🟢 ESTÁVEL"
        elif conf_percent > 50:
            st.warning(f"PREDIÇÃO DA IA: {class_name}")
            status_color = "🟡 ATENÇÃO"
        else:
            st.error(f"PREDIÇÃO DA IA: {class_name}")
            status_color = "🔴 ALERTA DE DOMÍNIO"
            
        st.markdown(f"<p class='big-font'>Confiança: {conf_percent:.1f}%</p>", unsafe_allow_html=True)
        
        st.markdown("#### Pensamento Analítico")
        st.info(f"O classificador convolucional determinou que a assinatura topográfica primária pertence à classe **{class_name}**. Status de leitura: {status_color}.")

    # --- COLUNA 3: Gráfico Compacto ---
    with col3:
        st.markdown("#### 📊 Distribuição de Probabilidades")
        
        # Transformando as probabilidades em um DataFrame para o Streamlit
        probs_df = pd.DataFrame({
            'Classe': CLASS_NAMES,
            'Confiança (%)': (probabilities.numpy() * 100).round(1)
        })
        # Ordenando para ficar mais visual
        probs_df = probs_df.sort_values('Confiança (%)', ascending=True)
        
        # Usando um gráfico de barras horizontal nativo (super compacto)
        st.bar_chart(probs_df.set_index('Classe'), height=350)

else:
    # Tela de espera caso nenhuma imagem tenha sido enviada
    st.write("Aguardando injeção de dados no Centro de Comando (Menu Lateral)...")