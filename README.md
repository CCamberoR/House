# 🏥 H.O.U.S.E - Health Observation, Understanding, and Symptom Evaluation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Un chatbot médico inteligente basado en RAG (Retrieval-Augmented Generation) que utiliza DeepSeek R1 y técnicas de recuperación híbrida para proporcionar información médica basada en consultas previas de profesionales médicos.

## 🚨 **ADVERTENCIA IMPORTANTE**

Este es un **prototipo de demostración** con fines educativos y de investigación. **NO es un sustituto del consejo médico profesional**. La información proporcionada no debe considerarse un diagnóstico médico. **Siempre consulta a un profesional de la salud cualificado** para obtener diagnósticos y tratamientos adecuados.

## ✨ Características

- **RAG Híbrido**: Combina recuperación densa (sentence-transformers) y dispersa (BM25) para obtener mejores resultados
- **Modelo de Lenguaje Avanzado**: Utiliza DeepSeek R1 Distill Llama 8B con cuantización para eficiencia
- **Interfaz Web Intuitiva**: Aplicación Streamlit con chat en tiempo real
- **Persistencia de Índices**: Los embeddings se guardan para cargas rápidas posteriores
- **Historial de Conversación**: Mantiene el contexto de la conversación
- **Configuración Flexible**: Parámetros ajustables para top-k y otros hiperparámetros

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │  Hybrid RAG      │ -> │  DeepSeek R1    │
│                 │    │  Retriever       │    │  Language Model │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐  ┌──────▼────────┐
            │ Dense Retriever│  │Sparse Retriever│
            │(Sentence-BERT) │  │    (BM25)     │
            └────────────────┘  └───────────────┘
```

### Componentes Principales

1. **Retrievers**: 
   - `SparseRetriever`: Utiliza BM25 para búsqueda basada en palabras clave
   - `DenseRetriever`: Utiliza sentence-transformers para búsqueda semántica
   - `HybridRetriever`: Combina ambos enfoques con pesos configurables

2. **Chatbot**: Integra el retriever con el modelo de lenguaje DeepSeek R1

3. **Interfaz Web**: Aplicación Streamlit para interacción del usuario

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- CUDA (opcional, para aceleración GPU)
- Al menos 8GB de RAM
- Espacio en disco: ~10GB para modelos y datos

### Configuración del Entorno

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd rag_solved
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requeriments.txt
```

### Configuración de APIs

1. **Hugging Face Token**:
   - Obtén tu token de [Hugging Face](https://huggingface.co/settings/tokens)
   - Reemplaza `"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"` en el código con tu token

2. **Groq API Key**:
   - Obtén tu API key de [Groq](https://groq.com)
   - Reemplaza `"groq_api_key"` en el código con tu clave real

### Preparación de Datos

1. **Crear directorio de datos**:
```bash
mkdir data
```

2. **Descargar dataset**:
   - Coloca los archivos `medicare_110k_train.json` y `medicare_110k_test.json` en el directorio `data/`

## 🎯 Uso

### Ejecutar la Aplicación

```bash
streamlit run rag_solved.py
```

La aplicación estará disponible en `http://localhost:8501`

### Primera Ejecución

En la primera ejecución, el sistema:
1. Descargará los modelos necesarios
2. Procesará el dataset médico
3. Creará y guardará los índices de búsqueda (esto puede tomar varios minutos)

### Funcionalidades de la Interfaz

- **Chat Principal**: Interfaz de conversación con el chatbot
- **Configuración Lateral**: 
  - Ajustar número de documentos recuperados (top_k)
  - Limpiar historial de conversación
- **Historial Persistente**: Las conversaciones se mantienen durante la sesión

## 🔧 Configuración Avanzada

### Parámetros del Retriever Híbrido

```python
retriever = HybridRetriever(
    weight_sparse=0.3,     # Peso para BM25
    weight_dense=0.7,      # Peso para búsqueda semántica
    model='sentence-transformers/all-MiniLM-L6-v2'
)
```

### Configuración del Modelo

```python
# Configuración de cuantización
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="fp8",
    bnb_8bit_use_double_quant=False
)
```

## 📁 Estructura del Proyecto

```
rag_solved/
├── rag_solved.py              # Archivo principal de la aplicación
├── rag_solved.ipynb           # Notebook de desarrollo
├── chatbot_finetuning.ipynb   # Notebook para fine-tuning
├── requeriments.txt           # Dependencias del proyecto
├── data/                      # Directorio de datos (crear manualmente)
│   ├── medicare_110k_train.json
│   └── medicare_110k_test.json
├── saved_retriever/           # Índices guardados (se crea automáticamente)
│   ├── documents.pkl
│   ├── sparse_bm25.pkl
│   ├── dense_embeddings.pkl
│   └── config.pkl
└── models/                    # Modelos descargados (se crea automáticamente)
```

## 🛠️ Clases Principales

### `TextPreprocessor`
Preprocesa texto eliminando stopwords y normalizando.

### `Retriever` (Clase Abstracta)
- `SparseRetriever`: Implementación con BM25
- `DenseRetriever`: Implementación con embeddings densos
- `HybridRetriever`: Combinación de ambos enfoques

### `Chatbot`
Clase principal que integra:
- Recuperación de documentos relevantes
- Generación de prompts estructurados
- Manejo del modelo de lenguaje
- Gestión del historial de conversación

## 🎨 Personalización

### Cambiar el Modelo de Embeddings

```python
retriever = HybridRetriever(
    model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)
```

### Modificar el Modelo de Lenguaje

```python
chatbot = Chatbot(
    retriever=retriever,
    model_name="microsoft/DialoGPT-medium"
)
```

### Ajustar Parámetros de Generación

```python
self.pipeline = pipeline(
    "text-generation",
    model=self.model,
    tokenizer=self.tokenizer,
    max_new_tokens=500,        # Aumentar para respuestas más largas
    temperature=0.8,           # Ajustar creatividad
    top_k=50,                  # Parámetros de muestreo
    top_p=0.95,
    repetition_penalty=1.2
)
```

## 📊 Rendimiento y Optimización

### Requisitos de Sistema
- **CPU**: Intel/AMD de 4+ núcleos recomendado
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: NVIDIA con 6GB+ VRAM (opcional pero recomendado)
- **Almacenamiento**: 10GB libres

### Optimizaciones Implementadas
- Cuantización a 8-bit para reducir uso de memoria
- Caching de embeddings para cargas rápidas
- Streaming de respuestas para mejor UX
- Paralelización en búsqueda híbrida

## 🐛 Solución de Problemas

### Error de Memoria
```bash
# Reducir el tamaño del batch o usar cuantización más agresiva
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
```

### Problemas con CUDA
```bash
# Verificar instalación de PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Errores de Dependencias
```bash
# Actualizar pip y reinstalar
pip install --upgrade pip
pip install -r requeriments.txt --force-reinstall
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🔗 Enlaces Útiles

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DeepSeek Models](https://huggingface.co/deepseek-ai)

## 📧 Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

---

**Recuerda**: Este proyecto es solo para fines educativos y de demostración. No debe utilizarse para diagnósticos médicos reales. Siempre consulta a profesionales médicos cualificados.
