import torch
import os
from huggingface_hub import login
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Tuple
import nltk
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import numpy as np
from datasets import load_dataset, Dataset
import re
import pickle
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import streamlit as st
from transformers import BitsAndBytesConfig, BioGptForCausalLM
import time
import random

st.set_page_config(page_title="HOUSE", page_icon="⚕️")

class TextPreprocessor:

    @classmethod
    def preprocess(cls, text: str, lang='english') -> str:
        if isinstance(text, tuple): text = ' '.join(text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words(lang))
        tokens = [t for t in tokens if t not in stop_words]

        return ' '.join(tokens)


class Retriever(ABC):

    def __init__(self, name='abstract_retriever'):
        self.name = name

    def get_name(self):
        return self.name

    """
    Este método recibe un conjunto de documentos y los indexa para poder realizar búsquedas posteriores
    """
    @abstractmethod
    def build_index(self, documents: List[str], lang: str = 'english'):
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve una lista con el la posición (index) del documento encontrado y su score de relevancia.
    """
    @abstractmethod
    def search(self, query: str, top_k: int = 3, lang:str = 'english') -> List[Tuple[int, float]]:
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve los documentos que considera relevantes.
    """
    @abstractmethod
    def search_documents(self, query: str, top_k: int = 3, lang:str = 'english') -> List[str]:
        pass


class SparseRetrieverNM(Retriever):

    def __init__(self):
        super().__init__("sparse_retriever_nm")
        self.vectorizer = TfidfVectorizer()
        self.nn_model = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="auto")

    """
    Construye el índice usando TF-IDF
    """
    def build_index(self, documents: List[str], lang: str = 'english'):
        self.documents = documents
        # Limpiar tokens innecesarios
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar embeddings dispersos TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        # Construir un modelo de búsqueda eficiente
        self.nn_model.fit(self.tfidf_matrix)

    def search(self, query: str, top_k: int = 5, lang: str = 'english') -> List[Tuple[int, float]]:
        # Vectorizar la consulta
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_vector = self.vectorizer.transform([processed_query])

        # Encontrar los vecinos más cercanos
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=top_k)
        # Retornar resultados como documentos y distancias inversas (para similitud)
        return [(idx, score) for idx, score in zip(indices[0], distances[0])][::-1]

    def search_documents(self, query: str, top_k: int = 3, lang: str = 'english') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

class SparseRetriever(Retriever):
    def __init__(self):
        super().__init__('sparse_retriever')

    def build_index(self, documents: List[str], lang:str = 'english'):
         self.documents = documents
         processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
         tokenized_docs = [doc.split() for doc in processed_docs]
         self.bm25 = BM25Okapi(tokenized_docs)


    def search(self, query: str, top_k: int = 3, lang:str = 'english') -> List[Tuple[int, float]]:
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_tokens = processed_query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def search_documents(self, query: str, top_k: int = 3, lang:str = 'english') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]


class DenseRetriever(Retriever):

    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__('dense_retriever' + model)
        # Cargar modelo de embeddings multilingüe
        self.model = SentenceTransformer(model)


    def build_index(self, documents: List[str], lang: str = 'english'):
        # Guardar documentos originales
        self.documents = documents
        #  Procesar texto eliminando tokens no relevantes
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar y almacenar embeddings
        self.embeddings = self.model.encode(processed_docs, show_progress_bar=True)


    def search(self, query: str, top_k: int = 3, lang: str = 'english') -> List[Tuple[int, float]]:
        # Realiza búsqueda por similitud de embeddings.
        processed_query = TextPreprocessor.preprocess(query, lang)
        # Generar embedding de la query
        query_embedding = self.model.encode([processed_query])
        # Calcular similitud
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]


    def search_documents(self, query: str, top_k: int = 3, lang: str = 'english') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

class HybridRetriever(Retriever):

    def __init__(self, weight_sparse: float = 0.3,
                 weight_dense: float = 0.7, model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__('hybrid_retriever' + model)
        self.model = model
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense

    def build_index(self, documents: List[str], lang: str = 'english'):
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(self.model)
        self.sparse_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)
        self.documents = documents

    def search(self, query: str, top_k: int = 3, lang: str = 'english') -> List[Tuple[int, float]]:
        # Obtener resultados de ambos retrievers
        sparse_results = self.sparse_retriever.search(query, top_k=top_k, lang=lang)
        dense_results = self.dense_retriever.search(query, top_k=top_k, lang=lang)

        # Combinar scores
        combined_scores = {}
        for idx, score in sparse_results:
            combined_scores[idx] = score * self.weight_sparse

        for idx, score in dense_results:
            if idx in combined_scores:
                combined_scores[idx] += score * self.weight_dense
            else:
                combined_scores[idx] = score * self.weight_dense

        # Ordenar resultados finales
        sorted_results = sorted(combined_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)[:top_k]
        # Preparar resultados
        return [(idx, score) for idx, score in sorted_results]

    def search_documents(self, query: str, top_k: int = 3, lang: str = 'english') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

def preprocess_conversation(text):
    text = text.strip()
    # Separar las intervenciones del humano y la IA
    parts = re.split(r"\[\|Human\|\]|\[\|AI\|\]", text)
    parts = [p.strip() for p in parts]

    if len(parts) != 3:
        return None, None  # Manejar conversaciones mal formateadas

    human_text = parts[1]
    ai_text = parts[2]

    return human_text, ai_text

def load_and_preprocess_data(dataset): # Ya no recibe un filepath, sino el dataset cargado
    """Carga el dataset (ya cargado) y preprocesa las conversaciones."""
    conversations = []
    print(len(dataset))
    for i, conv in enumerate(dataset):  # Iterar directamente sobre el dataset cargado
        human_text, ai_text = preprocess_conversation(conv['Conversation'])
        conversations.append((human_text, ai_text))
    return conversations


def save_retriever(retriever, save_dir='saved_retriever'):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the documents
    with open(os.path.join(save_dir, 'documents.pkl'), 'wb') as f:
        pickle.dump(retriever.documents, f)

    # Save sparse retriever data
    with open(os.path.join(save_dir, 'sparse_bm25.pkl'), 'wb') as f:
        pickle.dump(retriever.sparse_retriever.bm25, f)

    # Save dense retriever embeddings
    with open(os.path.join(save_dir, 'dense_embeddings.pkl'), 'wb') as f:
        pickle.dump(retriever.dense_retriever.embeddings, f)

    # Save model name and weights
    config = {
        'model_name': retriever.model,
        'weight_sparse': retriever.weight_sparse,
        'weight_dense': retriever.weight_dense
    }
    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Retriever saved in {save_dir}")

def load_retriever(save_dir='saved_retriever'):
    # Load config
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Initialize retriever with saved config
    retriever = HybridRetriever(
        weight_sparse=config['weight_sparse'],
        weight_dense=config['weight_dense'],
        model=config['model_name']
    )

    retriever.sparse_retriever = SparseRetriever()
    retriever.dense_retriever = DenseRetriever(config['model_name'])

    # Load documents
    documents_path = os.path.join(save_dir, 'documents.pkl')
    if os.path.exists(documents_path):
        with open(documents_path, 'rb') as f:
            retriever.documents = pickle.load(f)
    else:
        retriever.documents = []

    # Initialize index if does not exist the path
    if not os.path.exists(os.path.join(save_dir, 'sparse_bm25.pkl')) or not os.path.exists(os.path.join(save_dir, 'dense_embeddings.pkl')):
        retriever.build_index(documents)

    # Load sparse retriever data
    with open(os.path.join(save_dir, 'sparse_bm25.pkl'), 'rb') as f:
        retriever.sparse_retriever.bm25 = pickle.load(f)

    # Load dense embeddings
    with open(os.path.join(save_dir, 'dense_embeddings.pkl'), 'rb') as f:
        retriever.dense_retriever.embeddings = pickle.load(f)

    print(f"Retriever loaded from {save_dir}")
    return retriever


class Chatbot:
    def __init__(self, retriever, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="auto", load_in_8bit=False, load_in_4bit=True, ):
        self.retriever = retriever
        self.conversation_history = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loading Model")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="fp8",  # Menos agresivo que nf4
            bnb_8bit_use_double_quant=False  # Sin doble cuantización para más estabilidad
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,  #  Importante:  usar device_map
            cache_dir="models/" + model_name,
            quantization_config=bnb_config,  # Aquí aplicamos la cuantización
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=400,  # Ajusta según necesidad
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True), # Para ver la respuesta en tiempo real
            use_cache=True,
        )
        self.relevant_docs = None

    def generate_prompt(self, query: str, context_documents: List[str]) -> str:
        """Genera el prompt para DeepSeek R1."""
        context_documents = [doc if isinstance(doc, str) else ' '.join(doc) for doc in context_documents]

        context = "\n".join(context_documents)

        system_prompt = f"""
            <|system|>
                Eres un asistente de análisis de información médica. Tu función es analizar la consulta del usuario y, *basándote en la información proporcionada en el contexto de consultas médicas anteriores relativas a otros pacientes que te sirven de experiencia previa*, identificar posibles condiciones médicas que *podrían* estar relacionadas con los síntomas descritos.  **Importante:**
                *   **No eres un médico y no estás proporcionando un diagnóstico médico.**
                *   **La información que proporcionas es solo para fines informativos y de orientación, y NO debe interpretarse como un diagnóstico definitivo.**
                *   **Siempre se debe buscar el consejo de un profesional médico cualificado para obtener un diagnóstico y tratamiento adecuados.**
                *   **Si la información en el contexto no es suficiente para identificar posibles condiciones, indica claramente: "No tengo suficiente información para ofrecer posibles condiciones relacionadas.  Es fundamental que consultes a un profesional médico para una evaluación completa."**
                * Debes responder de forma estructurada, siguiendo este formato (y solo este formato):
                1. **Análisis de la consulta**: Breve resumen de los síntomas y la pregunta principal del usuario.
                2. **Posibles condiciones (basadas en el contexto):** Lista de *posibles* condiciones médicas que, *según el contexto proporcionado*, podrían estar relacionadas.  Usa un lenguaje condicional ("podría ser", "es posible que", "se asemeja a"). *No* afirmes categóricamente.  Si hay varias posibilidades, ordénalas de más a menos probable *según la frecuencia con la que aparecen en el contexto*.
                3. **Advertencia:** Reiteración de que esto NO es un diagnóstico y que se debe consultar a un médico.

                Responde en español.
                </s>
            """

        rag_context = f"""
            <|rag_context|>
            Contexto (Recuerda que esto no tiene que ver con la consulta del paciente, solo sirve de ejemplo):
            {context}
            </s>
        """

        user_prompt = f"""
                <|user|>
                Consultas previas del paciente: {query}
                </s>

                <|assistant|>
        """

        prompt = system_prompt + rag_context + user_prompt
        return prompt

    def chat(self, query: str, lang: str = 'english', top_k: int = 3, device_setup = "auto"):
        """Función principal de chat (adaptada para streaming)."""
        relevant_docs = self.retriever.search_documents(query, top_k=top_k, lang=lang)
        prompt = self.generate_prompt(query, relevant_docs)

        # Usamos .generate() directamente para tener más control (streaming y length_penalty)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_setup)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        #  Generamos en un hilo separado para no bloquear la interfaz
        with st.spinner('Generando respuesta...'):  # Muestra un spinner mientras genera
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  #  ¡Ajusta! Valor inicial bajo
                repetition_penalty=1.2,
                #length_penalty=1.5,   #  Opcional: descomenta para usar length_penalty
                do_sample=True,
                top_k=50,
                top_p=0.95,
                streamer=streamer,  #  Usa el streamer
            )

        #  Extraemos la respuesta completa (ya que el streamer la imprimió)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        #  Aislar la respuesta del chatbot (quitando el prompt)
        response = full_response.replace(prompt, "").strip()


        # ---  ACTUALIZA EL HISTORIAL  ---
        self.conversation_history += f"<|user|>\nConsulta del paciente: {query}\n</s>\n<|assistant|>\n{response}\n</s>\n"

        return response

    def clear_history(self):
        """Limpia el historial de la conversación."""
        self.conversation_history = ""
        self.relevant_docs = None

def main():

    token = "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"
    print("Hugging Face logging")

    login(token)
    device_setup= "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ",device_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Descarga de recursos necesarios
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)


    #st.image("doctor.png", width=200)  # Add your image path here
    st.title("HOUSE")
    st.write("Bienvenido. Soy H.O.U.S.E (Hybrid Ontology and Understanding System for Examination). Te proporciono información basada en consultas médicas realizadas por profesionales.  **No sustituyo a un profesional médico. De hecho, por mi nombre deberías saber ya que tienes lupus**")

    # --- Carga de datos y modelo (una sola vez) ---
    with st.spinner('Cargando datos y modelo...'):
        # Carga el dataset (solo si no se ha cargado ya)
        if 'dataset' not in st.session_state:
            path = "data/"
            ragbench = load_dataset(
                "json",
                data_files=[
                    path + "medicare_110k_train.json",
                    path + "medicare_110k_test.json",
                ],
                split="train"
            )
            st.session_state.dataset = load_and_preprocess_data(ragbench)
        # Carga el retriever (o lo crea y guarda si no existe)
        if 'retriever' not in st.session_state:
            if os.path.exists('saved_retriever'):
                print(f"Loading embedings")
                st.session_state.retriever = load_retriever()
            else:
                print(f"Creating embedings")
                retriever = HybridRetriever()
                retriever.build_index(st.session_state.dataset, lang='english')
                save_retriever(retriever) #Lo guardamos para la próxima
                st.session_state.retriever = retriever

        # Inicializa el chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = Chatbot(st.session_state.retriever, device_map=device_setup)

    chatbot = st.session_state.chatbot  # Accede al chatbot desde el estado de la sesión

    print("Más debugging, ha inciado el chatbot xd")
    # --- Barra lateral (opciones) ---
    with st.sidebar:
        st.header("Opciones")
        top_k = st.slider("Número de documentos a recuperar (top_k)", min_value=1, max_value=10, value=3)
        if st.button("Limpiar historial"):
            chatbot.clear_history()
            st.success("Historial limpiado.")
        st.write("---")
        st.write("**Advertencia:** Este es un prototipo. La información proporcionada no debe considerarse un diagnóstico médico. Consulta siempre a un profesional de la salud.")


    # --- Interfaz principal (chat) ---

    # Muestra el historial de la conversación
    for message in chatbot.conversation_history.split("</s>\n"): # Separamos por los tokens de fin
        if message.strip():  # Evita mensajes vacíos
            if "<|user|>" in message:
                with st.chat_message('user'):
                    st.write(message.replace("<|user|>", "").replace("Consulta del paciente:","").strip())
            elif "<|assistant|>" in message:
                with st.chat_message('assistant'):
                    st.write(message.replace("<|assistant|>", "").strip())

    # Input del usuario
    user_query = st.chat_input("Escribe tu consulta aquí...")

    if user_query:  #  Si el usuario ha escrito algo
        with st.chat_message("user"):
            st.write(user_query) # Mostramos su pregunta
        response = chatbot.chat(user_query, top_k=top_k, device_setup=device_setup) # Obtenemos la respuesta

        # filter just the answer of the chatbot
        answer = response.split("**Respuesta de un doctor en menos de 100 palabras**:\n")[-1]

        with st.chat_message("assistant"):
            st.write(response)  # Mostramos la respuesta

if __name__ == '__main__':
    main()

