# Jupiter Help Centre FAQ Chatbot

An interactive system that answers user queries using FAQs scraped from Jupiter Money’s official Help Centre. It automates scraping, preprocessing, semantic search, and optional GPT-based rephrasing to deliver friendly, accurate responses.

---

## Live Demo Link

```
https://jupiterproject-jd6tl8vrdbp9vdkheywqhc.streamlit.app/
```

## Features

- Automated FAQ scraping

- Data cleaning and normalization

- Semantic similarity search

- Fast retrieval with optional FAISS indexing

- LLM-based answer rephrasing for user-friendly responses

- Evaluation of accuracy and latency

# Setup Instructions

1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder-name>
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook

```bash
jupyter notebook Main.ipynb
```

- Scrape and clean FAQs

- Generate and save embeddings

- Build FAISS index

- Test retrieval and LLM rephrasing

4. Run the Streamlit App

```bash
streamlit run app.py
```

- Enter your OpenAI API key.

- Input any question (English/Hindi/Hinglish).

- See:

  - Retrieval-only answer with similarity score and latency

  - LLM-rephrased user-friendly answer

  - Suggested related FAQ questions

---

## Folder Structure

```bash
/Jupiter_Project
├── all_files/                       # Data artifacts and outputs
│   ├── final_jupiter_faqs.csv       # Raw scraped FAQ data
│   ├── jupiter_faqs_cleaned.csv     # Cleaned FAQ data
│   ├── faq_question_embeddings.npy  # Generated embeddings
│   ├── faq_faiss.index              # Saved FAISS index
│
├── Main.ipynb                       # End-to-end notebook pipeline
├── app.py                           # Optional app interface
├── faq_comparison_results.csv       # Evaluation results
├── requirements.txt                 # Required Python packages
└── README.md                        # This file: project overview and usage
```
