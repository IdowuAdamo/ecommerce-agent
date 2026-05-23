# NaijaShop AI

**DSN x BCT LLM Agent Challenge Hackathon Submission**

NaijaShop AI is a context-aware, multi-agent e-commerce intelligence platform tailored for the Nigerian market. It leverages conversational AI, behavioral user modeling, live data extraction, and OpenAI-powered price fairness scoring to provide transparent and hyper-personalized shopping recommendations.

## System Architecture

The platform operates via a directed acyclic graph (DAG) of 7 specialized AI agents:

1. **Discovery Agent**: Extracts structured shopping intent, budget constraints, and cultural context from natural language queries.
2. **User Modeling Agent**: Maps users to one of 6 predefined Nigerian archetypes (e.g., Budget Student, Lagos Professional) to handle cold-start recommendations.
3. **Commerce Intelligence Agent**: Executes live scraping against target storefronts (respecting robots.txt) with vector-based semantic fallbacks.
4. **Trust & Value Agent**: Evaluates price fairness using the OpenAI API with engineered prompts and calculates seller trust metrics.
5. **Recommendation Agent**: Ranks products using a multi-signal composite score and applies Maximal Marginal Relevance (MMR) for result diversity.
6. **Explanation Agent**: Generates transparent, human-readable reasoning cards justifying each recommendation.
7. **Review Simulation Agent**: (Task A) Synthesizes highly realistic Nigerian-style product reviews constrained by persona and price fairness data.

## Key Features

- **Price Fairness Scoring**: Uses the OpenAI API (`gpt-4o-mini`) with few-shot prompt engineering to predict Nigerian market prices and flag overpriced items.
- **Semantic Search**: Uses OpenAI `text-embedding-3-small` (384-dim) for product embedding and Pinecone vector similarity search.
- **Provider Failover Layer**: OpenAI serves as the primary LLM provider with automatic failover to Gemini upon rate limits or errors.
- **Live Market Data**: Integrates live scraping capabilities targeting specific categories on Jumia Nigeria.
- **Nigerian Contextualization**: Native understanding of local colloquialisms, slang, and cultural shopping constraints.

---

## v2.0.0 — Migration from DeBERTa to OpenAI API

### What Changed

| Component | v1.x (Before) | v2.0 (Now) |
|---|---|---|
| **Price Prediction** | DeBERTa-v3-small + LoRA fine-tuned model (`Idowenst/ecommerce-price-predictor-v1`) | OpenAI `gpt-4o-mini` with engineered prompts + JSON-mode |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) | OpenAI `text-embedding-3-small` (API, 384-dim) |
| **Memory (RAM)** | ~2 GB (PyTorch + model weights) | ~200 MB (pure Python) |
| **Cold-start time** | 60–120 seconds (model loading) | ~5 seconds |
| **Required HF Token** | Yes | No (optional) |
| **Deployment plan** | Render Standard ($25/mo) | Render Starter ($7/mo) |

### Why the Change Was Made

1. **Deployment resource limits**: PyTorch + DeBERTa required ~2 GB RAM, exceeding Railway's free tier and causing `resource limit exceeded` errors. The Standard Render plan ($25/month) was the minimum viable tier.
2. **Cold-start latency**: Loading local model weights took 60–120 seconds per cold start, causing health check failures on constrained platforms.
3. **Maintainability**: Maintaining a fine-tuned model requires periodic retraining as Nigerian market prices shift. The OpenAI API approach adapts automatically.
4. **Vercel compatibility**: Vercel serverless functions have a 250 MB bundle limit. Removing PyTorch brings the backend to ~250 MB Docker image, enabling more flexible deployment options.

### Prompt Engineering Details

The OpenAI price predictor uses:
- **Role definition**: System prompt establishes the model as a "Nigerian e-commerce pricing expert"
- **Explicit output format**: JSON-mode (`response_format={"type": "json_object"}`) with strict key name
- **Few-shot examples**: 10 representative Nigerian product/price pairs (smartphones, laptops, freezers, generators, etc.)
- **Temperature 0.0**: Deterministic output for consistent pricing
- **In-memory cache**: Repeated queries for the same product skip the API call entirely
- **Heuristic fallback**: Category-based fallback if the API call fails

### Reverting to DeBERTa

The original model code is fully preserved in comment blocks:
- `backend/app/models/price_predictor.py` — DeBERTa implementation at the bottom
- `backend/app/models/embeddings.py` — sentence-transformers implementation at the bottom
- `backend/requirements.ml.txt` — all original heavy dependencies

---

## Environment Variables

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Primary LLM + embedding + price prediction provider |
| `GEMINI_API_KEY` | Fallback LLM provider (optional but recommended) |
| `SUPABASE_URL` | PostgreSQL database for user profiles |
| `PINECONE_API_KEY` | Vector database for semantic product search |

### Optional (Legacy — only for DeBERTa revert)

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace token (only needed if reverting to DeBERTa) |

### Frontend

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API URL (set in Vercel dashboard) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Running via Docker Compose

```bash
docker-compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API Docs: `http://localhost:8000/docs`

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## Deploying to Render + Vercel

### Backend → Render

1. Push your code to GitHub
2. Go to **https://render.com** → New → **Blueprint**
3. Select your repository (Render auto-detects `render.yaml`)
4. In **Environment** tab, add your secret keys:
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `SUPABASE_URL`
5. Set plan to **Starter** ($7/mo, 512 MB RAM — sufficient without local ML models)
6. Deploy. First build: ~3 minutes. Cold-start: ~5 seconds.
7. Note your backend URL: `https://naijashop-backend.onrender.com`

### Frontend → Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy (from repo root)
vercel

# Set backend URL
vercel env add NEXT_PUBLIC_API_URL production
# Enter: https://naijashop-backend.onrender.com

# Production deploy
vercel --prod
```

---

## API Documentation

- `POST /api/v1/chat` — Conversational endpoint; returns ranked products and explanation cards
- `WS /ws/chat/{session_id}` — WebSocket for real-time agent trace streaming
- `POST /api/v1/reviews/simulate` — Generates persona-driven product reviews (Task A)
- `POST /api/v1/predict-price` — Direct OpenAI price prediction interface
- `POST /api/v1/evaluate/task-b` — Runs NDCG@10, HitRate, MRR metrics

## Testing

```bash
cd backend
pytest tests/ -v
```

## Repository Structure

- `/backend` — FastAPI application, LangGraph agent orchestration, OpenAI model wrappers
- `/frontend` — Next.js web application with glassmorphism UI
- `/model_training` — Jupyter notebooks for the original DeBERTa fine-tuning (reference)

---

*NaijaShop AI — Intelligent, transparent commerce for the Nigerian market.*