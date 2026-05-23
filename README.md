# NaijaShop AI

**DSN x BCT LLM Agent Challenge Hackathon Submission**

NaijaShop AI is a context-aware, multi-agent e-commerce intelligence platform tailored for the Nigerian market. It leverages conversational AI, behavioral user modeling, live data extraction, and a custom-trained DeBERTa price fairness model to provide transparent and hyper-personalized shopping recommendations.

## System Architecture

The platform operates via a directed acyclic graph (DAG) of 7 specialized AI agents:

1. **Discovery Agent**: Extracts structured shopping intent, budget constraints, and cultural context from natural language queries.
2. **User Modeling Agent**: Maps users to one of 6 predefined Nigerian archetypes (e.g., Budget Student, Lagos Professional) to handle cold-start recommendations.
3. **Commerce Intelligence Agent**: Executes live scraping against target storefronts (respecting robots.txt) with vector-based semantic fallbacks.
4. **Trust & Value Agent**: Evaluates price fairness using a fine-tuned DeBERTa-v3-small model and calculates seller trust metrics.
5. **Recommendation Agent**: Ranks products using a multi-signal composite score and applies Maximal Marginal Relevance (MMR) for result diversity.
6. **Explanation Agent**: Generates transparent, human-readable reasoning cards justifying each recommendation.
7. **Review Simulation Agent**: (Task A) Synthesizes highly realistic Nigerian-style product reviews constrained by persona and price fairness data.

## Key Features

- **Price Fairness Scoring**: Uses a DeBERTa-v3-small model (fine-tuned on 18,360 local e-commerce listings via LoRA) to predict market prices and flag overpriced items.
- **Provider Failover Layer**: Features a robust LLM provider manager. OpenAI serves as the primary provider, with automatic failover to Gemini upon rate limits, timeouts, or API errors, ensuring high availability.
- **Live Market Data**: Integrates live scraping capabilities targeting specific categories to ensure recommendations reflect current stock and pricing.
- **Nigerian Contextualization**: Native understanding of local colloquialisms, slang, and cultural shopping constraints.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker and Docker Compose (optional but recommended)

### Environment Configuration

Copy the example environment file and populate it with your credentials:

```bash
cp .env.example .env
```

Required Keys:
- `OPENAI_API_KEY`: Primary LLM provider.
- `GEMINI_API_KEY`: Fallback LLM provider.
- `HF_TOKEN`: HuggingFace token for the DeBERTa model (`Idowenst/ecommerce-price-predictor-v1`).
- `SUPABASE_URL`: PostgreSQL database connection.
- `PINECONE_API_KEY`: Vector database access.

### Running via Docker Compose (Recommended)

```bash
docker-compose up --build
```
This provisions the FastAPI backend, Next.js frontend, and Redis cache.
- Frontend: `http://localhost:3000`
- Backend API Docs: `http://localhost:8000/docs`

### Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## API Documentation

The backend exposes several REST and WebSocket endpoints:

- `POST /api/v1/chat`: Main conversational endpoint. Returns ranked products and explanation cards.
- `WS /ws/chat/{session_id}`: WebSocket endpoint for real-time streaming of the agent reasoning trace.
- `POST /api/v1/reviews/simulate`: Generates persona-driven product reviews (Task A evaluation).
- `POST /api/v1/predict-price`: Direct interface to the DeBERTa price prediction model.
- `POST /api/v1/evaluate/task-b`: Runs NDCG@10, HitRate, and MRR metrics against the recommendation engine.

## Repository Structure

- `/backend`: FastAPI application, LangGraph agent orchestration, model definitions, and evaluation scripts.
- `/frontend`: Next.js 14 web application featuring a glassmorphism UI and real-time agent trace streaming.
- `/model_training`: Jupyter notebooks for fine-tuning the DeBERTa price prediction model.

## Evaluation & Testing

Run the automated test suite and evaluation metrics:

```bash
cd backend
pytest tests/
```

*NaijaShop AI — Intelligent, transparent commerce for the Nigerian market.*