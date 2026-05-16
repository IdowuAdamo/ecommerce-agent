# 🇳🇬 NaijaShop AI — Multi-Agent Nigerian E-Commerce Intelligence Platform

> **DSN × BCT LLM Agent Challenge Hackathon Submission**

NaijaShop AI is a context-aware, multi-agent AI commerce platform that deeply understands Nigerian shopping behavior. It combines conversational AI, behavioral user modeling, live Jumia product scraping, DeBERTa-powered price fairness scoring, and Nigerian-localized review simulation.

---

## ✨ Key Differentiators

| Feature | What Makes It Unique |
|---|---|
| 🔬 **Price Fairness AI** | Fine-tuned DeBERTa-v3-small + LoRA on 18,360 Jumia products — estimates fair market prices |
| 🇳🇬 **Nigerian Persona Modeling** | 6 archetypes: Budget Student, Lagos Professional, Nigerian Mum, Tech Enthusiast, Online Skeptic, Market Trader |
| 🤖 **7-Agent Architecture** | LangGraph DAG with Discovery → UserModeling → CommerceIntel → TrustValue → Recommendation → Explanation → ReviewSim |
| 🕷️ **Live Jumia Scraping** | robots.txt compliant, ClaudeBot UA, 15 product categories |
| 📊 **Full Evaluation Suite** | NDCG@10, Hit Rate, MRR (Task B) + ROUGE, BERTScore, RMSE (Task A) |

---

## 🏗️ Architecture

```
User Query
    ↓
[1] Conversational Discovery Agent  ← Extracts intent, budget, category, Nigerian signals
    ↓
[2] User Modeling Agent             ← Cold-start personas, behavioral priors
    ↓
[3] Commerce Intelligence Agent     ← Live Jumia scraping + Pinecone vector search
    ↓
[4] Trust & Value Agent ★           ← DeBERTa price prediction + fairness scoring
    ↓
[5] Recommendation & Ranking Agent  ← Composite scoring + MMR diversity
    ↓
[6] Explanation Agent               ← Transparent reasoning cards
    ↓
[7] Review Simulation Agent         ← Nigerian persona reviews (Task A)
    ↓
Final Response + Ranked Products + Explanations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1-Command Setup

```bash
git clone <repo-url>
cd ecommerce-agent

# Copy env template and fill in your API keys
cp .env.example backend/.env

# Start full stack
make up
```

Then open:
- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
- **MLflow:** http://localhost:5000

### Manual Setup (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
cp ../.env.example .env  # edit .env with your keys
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key (GPT-4o-mini used by default) |
| `HF_TOKEN` | ✅ | HuggingFace token to load `Idowenst/ecommerce-price-predictor-v1` |
| `SUPABASE_URL` | ✅ | Supabase project URL for PostgreSQL |
| `PINECONE_API_KEY` | ✅ | Pinecone API key for vector search |
| `PRICE_MODEL_DEVICE` | ⬜ | `cpu` (default) or `cuda` |
| `OPENAI_MODEL` | ⬜ | `gpt-4o-mini` (default) or `gpt-4o` |

---

## 📡 API Endpoints

### Chat
```http
POST /api/v1/chat
{
  "session_id": "user_abc123",
  "message": "Find me a laptop under ₦400k for machine learning"
}
```

### Review Simulation (Task A)
```http
POST /api/v1/reviews/simulate
{
  "product_id": "prod001",
  "product_name": "HP 255 G9 Laptop",
  "product_category": "laptops",
  "actual_price": 450000,
  "predicted_fair_price": 420000,
  "num_reviews": 3
}
```

### Price Prediction
```http
POST /api/v1/predict-price
{
  "product_text": "category: laptops | name: HP 255 G9 | description: AMD Ryzen 5, 16GB RAM"
}
```

### Evaluation (Task B)
```http
POST /api/v1/evaluate/task-b
{
  "relevance_scores": [[1, 0, 1, 0, 0]],
  "predicted_scores": [[0.9, 0.3, 0.7, 0.2, 0.1]],
  "relevant_items": [["prod1", "prod3"]],
  "recommended_items": [["prod1", "prod5", "prod3", "prod2", "prod4"]],
  "k": 10
}
```

---

## 🧪 Running Evaluation

```bash
# Task A: ROUGE + BERTScore + RMSE
make evaluate-a

# Task B: NDCG@10 + HitRate + MRR
make evaluate-b

# All tests
make test

# Price model test
make test-price-model

# Live Jumia scraping test
make scrape
```

---

## 🧠 Price Predictor Model

The core competitive advantage — a DeBERTa-v3-small + LoRA model trained on Jumia data:

| Property | Value |
|---|---|
| Base Model | `microsoft/deberta-v3-small` |
| Adapter | LoRA (r=8, α=16) |
| Dataset | `Idowenst/jumia_dataset` (18,360 products) |
| HuggingFace | `Idowenst/ecommerce-price-predictor-v1` |
| Val RMSE | ₦142,632 |
| Val MAE | ₦53,675 |

**How it's used:**
1. Every fetched product gets a predicted fair price
2. Actual vs. predicted → fairness score (0-1)
3. Fairness score feeds into recommendation ranking (25% weight)
4. Fairness influences simulated review ratings (Task A)

---

## 🌐 Jumia Scraping (robots.txt Compliant)

- **User-Agent:** `ClaudeBot` (explicitly allowed by Jumia's robots.txt)
- **Allowed paths:** Category listings (`/laptops/`, `/phones/`, etc.)
- **Disallowed paths:** `/catalog/`, `/ratingreview/`, facet URLs — all respected
- **Polite delay:** 2s between requests (configurable)
- **Fallback:** Mock catalog if scraping fails

---

## 🗂️ Project Structure

```
ecommerce-agent/
├── backend/
│   ├── app/
│   │   ├── agents/          # All 7 AI agents + LangGraph orchestrator
│   │   ├── models/          # DeBERTa price predictor + embeddings
│   │   ├── services/        # Jumia scraper, Pinecone, PostgreSQL
│   │   ├── schemas/         # Pydantic data models
│   │   ├── evaluation/      # NDCG, ROUGE, BERTScore, RMSE
│   │   ├── data/            # Mock catalog fallback
│   │   └── main.py          # FastAPI application
│   ├── tests/               # pytest unit tests
│   ├── requirements.txt     # Pinned dependencies
│   └── Dockerfile
├── frontend/                # Next.js 14 + Tailwind premium UI
├── model_training/          # price-predictor.ipynb (DeBERTa training)
├── docker-compose.yml       # Full-stack container setup
├── .env.example             # Environment template
├── Makefile                 # One-command operations
└── README.md
```

---

## 📊 Hackathon Scoring Strategy

| Criterion | Our Approach | Expected Impact |
|---|---|---|
| Recommendation Quality | Composite scoring: semantic + behavioral + price fairness + trust | High |
| Contextual Relevance | Nigerian persona priors + slot extraction | High |
| Cold-Start Handling | 6 archetype templates mapped to Nigerian signals | High |
| Agentic Reasoning | LangGraph DAG, agent trace exposed in UI | High |
| Originality | DeBERTa price fairness — no other team has this | Very High |
| Reproducibility | Docker Compose, Makefile, pinned deps, seed=42 | High |
| Architecture Quality | Hexagonal architecture, async, typed schemas | High |
| Task A (ROUGE/BERTScore) | GPT-4o with Nigerian persona-constrained prompting | High |
| Task B (NDCG@10) | MMR reranking + multi-signal composite score | High |

---

## 🤝 Team

Built for the **DSN × BCT LLM Agent Challenge**

---

*NaijaShop AI — because every Nigerian deserves fair prices and honest recommendations* 🇳🇬