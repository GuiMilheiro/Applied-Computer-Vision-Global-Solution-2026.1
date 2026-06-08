# 🌍 TerraNet: Classificação de Terreno por Imagens de Satélite

Este projeto foi desenvolvido para a disciplina de **Applied Computer Vision** (Global Solution FIAP 2026). O objetivo é classificar imagens de satélite em 10 classes distintas de uso do solo, utilizando Redes Neurais Convolucionais (CNNs) **construídas do zero**, sem o uso de Transfer Learning.

## 📊 O Dataset (EuroSAT)
Utilizamos o dataset EuroSAT, composto por 27.000 imagens (64x64) multiespectrais e RGB cobrindo a Europa. O dataset foi dividido em 70% Treino, 15% Validação e 15% Teste.
* **Data Augmentation:** Aplicado exclusivamente no conjunto de treino (RandomFlip, RandomRotation, ColorJitter, RandomResizedCrop) para melhorar a generalização e evitar *overfitting*.

## 🧠 Arquiteturas Desenvolvidas (From Scratch)
Para avaliar o impacto da topologia na extração de features, construímos duas CNNs distintas:

1. **TerraNet-Shallow (Larga e Rasa):** 3 blocos convolucionais com filtros maiores (5x5). Focada na extração de texturas de baixa frequência (áreas homogêneas como florestas e pastos).
2. **TerraNet-Deep (Estreita e Profunda):** 5 blocos convolucionais com filtros menores (3x3) consistentes. Focada na construção de features hierárquicas e detecção de bordas complexas (rios e rodovias).

## 🚀 Loop de Treinamento Otimizado
O modelo foi treinado aproveitando aceleração local via GPU, contendo:
* **AMP (Automatic Mixed Precision):** Cálculos em FP16 para maximizar o throughput.
* **OneCycleLR:** Agendador de taxa de aprendizado para convergência mais estável.
* **Early Stopping:** Monitoramento da perda de validação para interrupção precoce.

## 📈 Resultados e Comparação
O treinamento superou expressivamente a métrica de 88% de acurácia exigida pelo escopo do projeto. A avaliação no conjunto de teste (imagens invisíveis ao modelo durante o treino) revelou os seguintes resultados:

* 🏆 **TerraNet-Shallow (Campeã):** Acurácia de **96.62%**
* 🥈 **TerraNet-Deep:** Acurácia de **95.28%**

**Análise Técnica das Matrizes de Confusão:**
A arquitetura *Shallow* venceu pois imagens de satélite em baixa resolução (64x64) são majoritariamente compostas por texturas globais (borrões de cor), favorecendo os filtros mais largos (5x5). Seus maiores erros lógicos foram a confusão entre `AnnualCrop` e `PermanentCrop` (23 incidências), o que é esperado visualmente.
Em contraste, a arquitetura *Deep* (filtros 3x3 focados em micro-detalhes) forçou a busca por estruturas complexas inexistentes nesse nível de abstração, resultando em confusões clássicas de redes profundas, como classificar áreas `Industrial` como `Residential` (27 incidências) devido à similaridade de telhados.

*(Nota: As matrizes de confusão detalhadas encontram-se na pasta `/outputs`).*

## ⚠️ Limitações e Generalização (Domain Shift)
Durante os testes de validação do deploy, submetemos a rede a imagens *Out-of-Distribution* (OOD) retiradas do Google Maps em alta resolução e perspectiva isométrica (3D). 

O modelo apresentou uma queda drástica de confiança (média de 34%), evidenciando um claro fenômeno de **Domain Shift**. Isso ocorre por duas limitações físico-matemáticas mapeadas:
1. **Resolução Espacial:** O EuroSAT (Sentinel-2) opera com 10 metros/pixel, enquanto o Google Maps opera na escala de centímetros/pixel. O "zoom" destrói a escala da textura aprendida.
2. **Perspectiva:** O dataset de treino é 100% *Nadir* (topo-baixo chapado). A renderização de sombras e laterais de prédios em 3D confunde a extração de features da CNN.

**Conclusão:** Para um ambiente de produção comercial, o dataset precisaria ser enriquecido com imagens multi-escala para garantir robustez fora de ambientes de satélite de baixa órbita.

## 💻 Como Rodar a Demonstração (Deploy)
Foi desenvolvida uma interface interativa usando Streamlit.
1. Instale as dependências: `pip install torch torchvision matplotlib seaborn scikit-learn streamlit`
2. Execute a aplicação: `streamlit run app.py`
3. Acesse via browser (`localhost:8501`), faça o upload de uma imagem `.jpg` e visualize a predição em tempo real.

### Integrantes
- Guilherme Dejulio Milheiro (RM550295)
- Enzo Vasconcelos (RM550702)
- Ricardo Queiroz (RM94241)
- Jhonatan Curci (RM94188)
- Felipe Hideki (RM98323) 

## link do Streamlit
https://applied-computer-vision-global-solution-2026-1.streamlit.app/
