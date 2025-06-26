import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import os
import time

# ---- Settings ----
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# ---- Data Load ----
@st.cache_resource
def load_faq_data():
    df = pd.read_csv("all_files/jupiter_faqs_cleaned.csv")
    questions = df['question'].tolist()
    return df, questions

# ---- OpenAI Embedding ----
def get_openai_embedding(text, api_key):
    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return np.array(resp.data[0].embedding, dtype='float32')

# ---- Embedding File Management ----
def compute_and_save_embeddings(questions, api_key, file_path="all_files/faq_openai_embeddings.npy"):
    embeddings = []
    for q in stqdm(questions, desc="Generating embeddings..."):
        emb = get_openai_embedding(q, api_key)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    np.save(file_path, embeddings)
    return embeddings

def stqdm(iterable, desc=""):
    # Streamlit-friendly progress bar
    progress = st.progress(0)
    total = len(iterable)
    for idx, item in enumerate(iterable):
        yield item
        progress.progress((idx + 1) / total, text=f"{desc} ({idx+1}/{total})")
    progress.empty()

# ---- FAISS Build/Load ----
def build_faiss_index(embeddings):
    # Normalize for cosine similarity
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(norm_emb.shape[1])
    index.add(norm_emb)
    return index, norm_emb

# ---- Retrieval ----
def retrieval_faiss(user_query, model, index, df, embeddings, questions, api_key, threshold=0.65, top_k=1):
    user_emb = get_openai_embedding(user_query, api_key)
    user_emb = user_emb / np.linalg.norm(user_emb)
    D, I = index.search(user_emb.reshape(1, -1), top_k)
    similarity = float(D[0][0])
    idx = int(I[0][0])
    if similarity < threshold:
        return None, None, similarity, None
    return questions[idx], df.iloc[idx]['answer'], similarity, idx

# ---- OpenAI LLM Rephrasing Function ----
def openai_llm_answer(user_query, retrieved_q, retrieved_a, api_key):
    if retrieved_q is None or retrieved_a is None:
        return "Sorry, I don't know the answer to that. Please contact Jupiter support!"
    prompt = (
        "You are a helpful, friendly FAQ assistant for Jupiter Money (the Indian fintech app, not the planet). "
        "Answer only based on the FAQ answer provided below. "
        "Do not ask for further questions. "
        "Rephrase briefly for friendliness. "
        "If there is no relevant answer, politely say you don't know.\n\n"
        f"User asked: {user_query}\n"
        f"FAQ question: {retrieved_q}\n"
        f"FAQ answer: {retrieved_a}\n"
        "Final user-facing answer:"
    )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ---- Suggest Related FAQ Questions (FAISS Semantic) ----
def suggest_related_faqs_faiss(user_query, index, questions, embeddings, api_key, top_n=4, exclude_idx=None, sim_threshold=0.55):
    user_emb = get_openai_embedding(user_query, api_key)
    user_emb = user_emb / np.linalg.norm(user_emb)
    D, I = index.search(user_emb.reshape(1, -1), top_n + 5)
    suggestions = []
    for similarity, idx_ in zip(D[0], I[0]):
        if exclude_idx is not None and idx_ == exclude_idx:
            continue
        if similarity < sim_threshold:
            continue
        if questions[idx_] not in suggestions:
            suggestions.append(questions[idx_])
        if len(suggestions) == top_n:
            break
    if len(suggestions) < top_n:
        extra = [q for q in questions if q not in suggestions and (exclude_idx is None or questions.index(q) != exclude_idx)]
        suggestions += extra[:top_n - len(suggestions)]
    return suggestions

# ---- Streamlit UI ----

st.title("💡 Jupiter FAQ AI Chatbot (Multilingual)")
st.markdown("> Ask your question in English, Hindi, Hinglish about Jupiter Money.")

openai_api_key = st.text_input("Enter your OpenAI API key:", type="password", help="Required for embeddings and LLM.")

df, questions = load_faq_data()

# Embedding/Index loading/generation (only happens once per API key)
if openai_api_key:
    emb_file = "all_files/faq_openai_embeddings.npy"
    if os.path.exists(emb_file):
        try:
            embeddings = np.load(emb_file)
        except Exception:
            st.warning("Failed to load cached embeddings. Regenerating.")
            embeddings = compute_and_save_embeddings(questions, openai_api_key, emb_file)
    else:
        embeddings = compute_and_save_embeddings(questions, openai_api_key, emb_file)

    index, norm_emb = build_faiss_index(embeddings)
else:
    embeddings = None
    index = None

user_query = st.text_input("Ask your question:")

if user_query and openai_api_key and embeddings is not None:
    with st.spinner("Finding the best answers..."):
        t0 = time.time()
        ret_q, ret_a, ret_sim, idx = retrieval_faiss(
            user_query, None, index, df, norm_emb, questions, openai_api_key, threshold=0.65
        )
        t1 = time.time()
        retrieval_time_ms = (t1 - t0) * 1000

        hallucinated = (ret_q is None or ret_sim < 0.65)

        if not hallucinated:
            t2 = time.time()
            llm_response = openai_llm_answer(user_query, ret_q, ret_a, openai_api_key)
            t3 = time.time()
            llm_time_ms = (t3 - t2) * 1000
        else:
            llm_response = "Sorry, I don't know the answer to that. Please contact Jupiter support!"
            llm_time_ms = 0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔎 Retrieval-Only")
            st.write(ret_a if not hallucinated else "No good match found.")
            st.caption(
                (f"**Similarity:** {ret_sim:.2f} &nbsp;|&nbsp; "
                 f"**Time:** {retrieval_time_ms:.0f} ms") if not hallucinated else
                f"Time: {retrieval_time_ms:.0f} ms"
            )

        with col2:
            st.subheader("🤖 LLM Rephrased")
            st.write(llm_response)
            st.caption(f"Time: {llm_time_ms:.0f} ms" if not hallucinated else "No LLM answer generated.")

        # Related suggestions after query
        related = suggest_related_faqs_faiss(user_query, index, questions, norm_emb, openai_api_key, top_n=4, exclude_idx=idx)
        st.markdown("**You might also ask:**")
        for r in related:
            if st.button(r, key=f"sugg_{r}"):
                st.session_state["input_box"] = r
                st.experimental_rerun()

st.markdown("---")
st.info(
    "You can test with English, Hindi, or Hinglish queries! Embeddings and index are managed automatically for your API key."
)
