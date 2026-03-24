# 🧠 Graph-RAG Hearthstone Card Generation

> A data-driven pipeline that integrates Knowledge Graphs, Large Language Models, and Diffusion Models to generate **logically consistent** and **style-aligned** game cards.

---

## 📌 Overview

This project explores a novel paradigm for AI-generated game content by introducing **structured priors** into the generation process.

Unlike traditional text-to-image or LLM-based approaches, which often suffer from:

* ❌ Logical inconsistencies (e.g., invalid game mechanics)
* ❌ Style mismatch across generated assets

We propose a **Graph-RAG pipeline** that combines:

* Knowledge Graphs for **mechanical reasoning**
* Graph Neural Networks (GNNs) for **balance prediction**
* Graph-based Retrieval-Augmented Generation for **style alignment**

---

## 🚀 Key Idea

> **Graph = Logic, Model = Creativity**

We decouple the generation process into two parts:

* 🧠 **Knowledge Graph** → ensures game logic and balance
* 🎨 **Diffusion Model** → generates high-quality visuals

---

## 🏗️ Pipeline

### 1. Data Collection

* Source: Blizzard API
* Extract:

  * Card metadata (cost, stats, class)
  * Text descriptions
  * Card images

---

### 2. LLM-based Relation Extraction

We use LLMs to extract **implicit mechanics and synergies** from card descriptions.

Example:

```json
Input:
"Spell Damage +1. Battlecry: Draw a card."

Output:
{
  "mechanics": ["Spell Damage", "Battlecry", "Draw"],
  "synergy": ["Spell-heavy decks"]
}
```

---

### 3. Knowledge Graph Construction

Using NetworkX:

* Nodes:

  * Cards
  * Mechanics
  * Classes
* Edges:

  * Explicit (from metadata)
  * Implicit (from LLM extraction)

This enables **multi-hop reasoning** between cards.

---

### 4. Graph Machine Learning (GNN)

We apply Graph Neural Networks for:

* 🔗 **Link Prediction**

  * Suggest missing relations (e.g., tribe/type)
* 📊 **Node Regression**

  * Predict balanced card stats (cost, attack, health)

Framework:

* PyTorch
* PyTorch Geometric (PyG)

---

### 5. Graph-RAG Retrieval

For a newly generated card:

1. Traverse the graph
2. Retrieve topologically similar cards
3. Extract their artwork

---

### 6. Conditional Image Generation

We use:

* Stable Diffusion
* IP-Adapter / ControlNet

Inputs:

* Text prompt (card description)
* Image prompt (retrieved card art)

Output:

* 🎨 Style-consistent card artwork

---

## 🛠️ Tech Stack

| Component       | Tools                  |
| --------------- | ---------------------- |
| Data Collection | requests, pandas       |
| LLM             | OpenAI API / LangChain |
| Graph           | NetworkX               |
| GNN             | PyTorch, PyG           |
| Retrieval       | Graph traversal        |
| Diffusion       | diffusers, IP-Adapter  |

---

## 📂 Project Structure

```
project-root/
│
├── data/                # Raw and processed card data
├── scripts/
│   ├── fetch_data.py
│   ├── extract_relations.py
│   ├── build_graph.py
│   └── train_gnn.py
│
├── models/              # Trained GNN / checkpoints
├── graph/               # Serialized graph objects
├── generation/          # Diffusion pipelines
│
├── notebooks/           # Experiments / demos
├── docs/                # Project documentation
│
└── README.md
```

---

## 🧪 Example Workflow

1. Fetch cards from Blizzard API
2. Extract mechanics via LLM
3. Build knowledge graph
4. Train GNN for prediction
5. Insert a new card node
6. Retrieve similar cards (Graph-RAG)
7. Generate final artwork

---

## 📊 Motivation

Existing approaches:

* Pure LLM → lacks structured reasoning
* Pure diffusion → lacks semantic control

Our approach:

* ✅ Logical consistency via graph structure
* ✅ Style alignment via graph-based retrieval
* ✅ Interpretable pipeline

---

## 📚 Related Work

* Retrieval-Augmented Generation (RAG)
* GraphRAG
* Knowledge Graph + LLM integration
* Conditional diffusion (IP-Adapter)

---

## 🗓️ Timeline

* **Phase 1**: Data & Graph Construction
* **Phase 2**: GNN Training & Evaluation
* **Phase 3**: Graph-RAG + Diffusion
* **Phase 4**: Integration & Visualization

---

## 🔮 Future Work

* More advanced graph embeddings (e.g., Graph Transformer)
* Automatic balance validation
* Interactive card generation UI

---

## 🤝 Contributions

This project is part of a course combining:

* Graph Machine Learning
* Computer Vision
* Large Language Models

---

## 📎 Notes

This repository is a **research-oriented prototype**, focusing on:

* Pipeline design
* System integration
* Concept validation

---
