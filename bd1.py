import streamlit as st
import faiss
import pandas as pd
import numpy as np
import requests
import json
import urllib3

# Suppress HTTPS certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Function to generate embeddings using Ada API ---
def get_embedding(text, api_key):
    url = "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/ada-002/embeddings?api-version=2024-02-01"
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "api-key": api_key
    }
    payload = {
        "input": text,
        "user": "example_user",
        "input_type": "query",
        "model": "string"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            return np.array(embedding, dtype='float32')
        else:
            st.error(f"Ada API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Exception: {str(e)}")
        return None

# --- Function to use GPT-4-o-mini API ---
def generate_response(retrieved_rows, user_query, api_key):
    url = "https://innovate-openai-api-mgt.azure-api.net/innovate-tracked/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-01"
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "api-key": api_key
    }
    context = "\n".join([str(row) for row in retrieved_rows])
    prompt = f"Context:\n{context}\n\nUser Query: {user_query}\n\nProvide a concise and helpful response based on the context."
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"GPT-4-o-mini API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Exception: {str(e)}")
        return None

# --- Initialize FAISS Index ---
def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index

# --- Streamlit App ---
def main():
    st.title("FAISS Search + GPT-4-o-mini Integration")
    st.write("Upload your Excel file to get started.")

    # Input: API Keys
    ada_api_key = st.text_input("Enter your Ada API Key:", type="password")
    gpt_api_key = st.text_input("Enter your GPT-4-o-mini API Key:", type="password")

    # Step 1: Upload Excel File
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "csv"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        if st.button("Generate Embeddings"):
            if not ada_api_key:
                st.error("Ada API Key is required!")
                return

            embeddings = []
            for _, row in df.iterrows():
                text = " ".join(map(str, row.values))
                embedding = get_embedding(text, ada_api_key)
                if embedding is not None:
                    embeddings.append(embedding)

            if embeddings:
                st.success("Embeddings Generated!")
                index = create_faiss_index(embeddings)
                st.success("FAISS Index Created!")
                st.session_state["index"] = index
                st.session_state["df"] = df

    # Step 2: Query Interface
    if "index" in st.session_state and "df" in st.session_state:
        query = st.text_input("Enter your query:")
        if query and st.button("Search"):
            query_embedding = get_embedding(query, ada_api_key)
            if query_embedding is not None:
                D, I = st.session_state["index"].search(np.array([query_embedding]), k=5)
                retrieved_rows = [st.session_state["df"].iloc[idx] for idx in I[0]]
                st.write("Search Results:")
                for row in retrieved_rows:
                    st.write(row)

                # Generate GPT-4-o-mini response
                if gpt_api_key:
                    gpt_response = generate_response(retrieved_rows, query, gpt_api_key)
                    st.write("GPT-4-o-mini Response:")
                    st.write(gpt_response)

if __name__ == "__main__":
    main()