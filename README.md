# 🌩️ Cloud Service Cost Analyzer & 🧠 Architron

This repository contains two Streamlit-based tools:

1. **Cloud Service Cost Analyzer (`cloud.py`)** – an AI-assisted tool to recommend and estimate cloud service costs.
2. **Architron (`Architron1.py`)** – an intelligent assistant to recommend software architectures, analyze code, and visualize system components.

---

## 🌩️ Cloud Service Cost Analyzer

### Overview

Cloud Service Cost Analyzer leverages Google Gemini AI to recommend cloud services based on user input and visualizes cost projections.

### Features

- 🔍 AI-based service recommendations (AWS, GCP, Azure)
- 📊 Cost visualization with Plotly
- 💡 Dynamic projections: daily, weekly, monthly
- 📈 Linear regression for cost scaling

### Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run cloud.py
   ```

---

## 🧠 Architron: Intelligent Architecture & Code Analyzer

### Overview

Architron analyzes code to extract architecture components, recommends software architecture types, and generates system diagrams.

### Features

- 📦 Language detection and parsing for Python, JavaScript, Java, and C#
- 🧱 Extract classes, functions, imports, and relationships
- 🧠 LLM-assisted architecture recommendation and code understanding
- 🛠️ Mermaid and NetworkX diagrams for visualization
- 🧑‍💻 Roadmap and deployment cost estimates

### Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:

   ```bash
   streamlit run Architron1.py

   ```

---

## 🔧 Requirements

Ensure you have the following in `requirements.txt`:

```
streamlit
pandas
numpy
plotly
scikit-learn
google-generativeai
spacy
graphviz
networkx
matplotlib
python-dotenv
requests
groq
```

---

## 📂 Structure

```
├── cloud.py             # Cloud cost estimator and visualization
├── Architron1.py        # Architecture recommender and code analyzer
└── README.md            # Project documentation
```
