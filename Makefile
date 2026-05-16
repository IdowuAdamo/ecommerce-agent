.PHONY: setup install run dev test evaluate-a evaluate-b scrape up down logs clean

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "🚀 Setting up NaijaShop AI..."
	cp -n .env.example backend/.env || true
	$(MAKE) install
	@echo "✅ Setup complete! Edit backend/.env with your API keys, then run: make up"

install:
	@echo "📦 Installing backend dependencies..."
	cd backend && pip install -r requirements.txt

# ── Development ───────────────────────────────────────────────────────────────
dev:
	@echo "🔥 Starting backend in dev mode..."
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "🎨 Starting frontend in dev mode..."
	cd frontend && npm run dev

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	@echo "🐳 Starting full stack with Docker Compose..."
	docker-compose up --build -d
	@echo "✅ Services running:"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  MLflow:   http://localhost:5000"

down:
	docker-compose down

logs:
	docker-compose logs -f backend

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running tests..."
	cd backend && pytest tests/ -v --tb=short

test-price-model:
	@echo "🧠 Testing price predictor..."
	cd backend && python -c "\
from app.models.price_predictor import PricePredictorModel; \
m = PricePredictorModel.get_instance(); \
price = m.predict('category: laptops | name: HP 255 G9 16GB RAM | description: AMD Ryzen 5 Windows 11'); \
print(f'Predicted price: ₦{price:,.0f}')"

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate-a:
	@echo "📊 Running Task A evaluation (ROUGE + BERTScore)..."
	cd backend && python -m app.evaluation.run_task_a

evaluate-b:
	@echo "📊 Running Task B evaluation (NDCG@10 + HitRate)..."
	cd backend && python -m app.evaluation.run_task_b

# ── Data ──────────────────────────────────────────────────────────────────────
scrape:
	@echo "🕷️  Scraping fresh product data from Jumia..."
	cd backend && python -c "\
import asyncio; \
from app.services.jumia_client import JumiaScraper; \
async def main(): \
    scraper = JumiaScraper(); \
    products = await scraper.scrape_category('laptops', max_pages=3); \
    print(f'Scraped {len(products)} laptops'); \
    [print(f'  ₦{p.price:,} — {p.name[:50]}') for p in products[:5]]; \
asyncio.run(main())"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	docker-compose down -v --remove-orphans 2>/dev/null || true
