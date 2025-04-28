# 🛍️ On-Premise AI Product Recommendation System (LLM + RAG)

## 📚 Project Overview
An on-premise AI system developed in collaboration with **Narosu Co., Ltd.** to recommend the **Top 5 most relevant products** based on user queries, using **LLMs**, **RAG**, **NLP**, and **vector search** technologies.

---

## 🛠️ Key Technologies

| Component             | Technology Used                   |
|------------------------|------------------------------------|
| LLM                    | LLaMA 3.1 8B instant               |
| Embedding Model        | OpenAI text-embedding-ada-002      |
| Vector DB / Search     | Qdrant                             |
| LLM Framework          | LangChain                          |
| RAG Retriever          | LangChain + Qdrant + LLM           |
| API Architecture       | Gradio + FastAPI                   |
| Chatbot UI             | Gradio                             |

---

## 🏗️ System Architecture

- User queries processed via **Gradio Chatbot UI**.
- Semantic search performed using **Qdrant** vector database.
- **RAG Retriever** enhances queries by fetching relevant data.
- **LLaMA 3.1 8B** generates intelligent alternative recommendations.
- Final Output includes: Product Code, Title, Price, Shipping Cost, Origin, and Product Image.

![System Diagram](https://github.com/user-attachments/assets/16fec1e9-113e-4cc9-b0b0-1862298d3889)

---

## 🚀 Main Features

- Supports **English** and **Japanese** queries (product data remains in Korean).
- Top 5 product recommendations:
  - **3 products** from OwnerClan dataset
  - **2 products** from Naver (via API or web crawling)
- Dynamic follow-up questioning for better recommendations.
- Optimized response time within **5–10 seconds**.

---

## 📦 Dataset

- **OwnerClan Dataset**:
  - 7,000 items (initial batch)
  - 4 million items (full dataset)
- Data includes product descriptions, attributes, and keyword metadata.
- Embeddings created with `text-embedding-ada-002` and stored in **Qdrant**.

---

## 🧠 Sample Output

- Interactive chatbot provides product suggestions based on user queries.
- Displays multiple images and product details.

![Chatbot UI](https://github.com/user-attachments/assets/fd7570dd-8416-4856-99e4-1c52efa14471)

---

## 📌 Future Enhancements

- Dynamic LLM and embedding model switching.
- Improved multi-source retrieval and hybrid ranking algorithms.

---

# 📈 Project Goal

Deliver a high-performance, multilingual AI-based product search and recommendation system, using cutting-edge **LLM** and **RAG** techniques, optimized for independent online stores.
![Chatbot Sample Output](https://github.com/user-attachments/assets/f634e1c3-4237-499c-9370-aeeaf4926224)

---
