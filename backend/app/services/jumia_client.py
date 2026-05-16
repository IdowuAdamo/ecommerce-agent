"""
Jumia Nigeria Product Scraper — robots.txt compliant.

Allowed by robots.txt:
  ✅ ClaudeBot / anthropic-ai user-agent: Allow: /
  ✅ Category listing pages (no facet params)
  ❌ /catalog/, /ratingreview/, /recommendation/, facet URLs
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.schemas.product import Product

logger = logging.getLogger(__name__)

JUMIA_CATEGORIES: dict[str, str] = {
    "laptops": "https://www.jumia.com.ng/laptops/",
    "phones": "https://www.jumia.com.ng/phones-tablets/",
    "televisions": "https://www.jumia.com.ng/televisions/",
    "appliances": "https://www.jumia.com.ng/home-appliances/",
    "generators": "https://www.jumia.com.ng/generators/",
    "cameras": "https://www.jumia.com.ng/cameras/",
    "audio": "https://www.jumia.com.ng/audio-video/",
    "computing": "https://www.jumia.com.ng/computing/",
    "gaming": "https://www.jumia.com.ng/video-games/",
    "health": "https://www.jumia.com.ng/health-beauty/",
    "fashion_men": "https://www.jumia.com.ng/men-clothing/",
    "shoes_men": "https://www.jumia.com.ng/men-shoes/",
    "fashion_women": "https://www.jumia.com.ng/women-clothing/",
    "shoes_women": "https://www.jumia.com.ng/women-shoes/",
    "baby": "https://www.jumia.com.ng/baby-products/",
    "furniture": "https://www.jumia.com.ng/furniture/",
    "sports": "https://www.jumia.com.ng/sports-outdoor/",
}

BASE_URL = "https://www.jumia.com.ng"

NIGERIAN_BRANDS = [
    "Samsung", "Apple", "Tecno", "Infinix", "Itel", "LG", "Sony",
    "HP", "Dell", "Lenovo", "Asus", "Acer", "Hisense", "Haier",
    "Nexus", "Polystar", "Scanfrost", "Thermocool", "Binatone",
    "Canon", "Nikon", "Panasonic", "Xiaomi", "Huawei", "Oppo",
    "Vivo", "Nokia", "Microsoft", "Silver Crest", "Master Chef",
]


class JumiaScraper:
    """Async scraper for Jumia Nigeria product listings (robots.txt compliant)."""

    def __init__(self):
        s = get_settings()
        self.delay = s.scraper_delay_seconds
        self.max_retries = s.scraper_max_retries
        self.timeout = s.scraper_timeout_seconds
        self.headers = {
            "User-Agent": s.scraper_user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    async def scrape_category(
        self, category: str, max_pages: int = 3, max_products: int = 40
    ) -> list[Product]:
        """Scrape products from a Jumia category listing page."""
        base_url = JUMIA_CATEGORIES.get(category)
        if not base_url:
            logger.warning(f"Unknown category '{category}'. Use: {list(JUMIA_CATEGORIES)}")
            return []

        products: list[Product] = []
        async with httpx.AsyncClient(
            headers=self.headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            for page in range(1, max_pages + 1):
                if len(products) >= max_products:
                    break
                page_url = base_url if page == 1 else f"{base_url}?page={page}"
                batch = await self._scrape_page(client, page_url, category)
                products.extend(batch)
                if not batch:
                    break
                if page < max_pages:
                    await asyncio.sleep(self.delay)

        logger.info(f"Scraped {min(len(products), max_products)} products from '{category}'")
        return products[:max_products]

    async def _scrape_page(
        self, client: httpx.AsyncClient, url: str, category: str
    ) -> list[Product]:
        for attempt in range(self.max_retries):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return self._parse_listing_page(resp.text, category)
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429:
                    await asyncio.sleep(self.delay * (2 ** attempt))
                elif code in (403, 404):
                    logger.error(f"HTTP {code} for {url}")
                    return []
                else:
                    logger.warning(f"HTTP {code} attempt {attempt+1}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.delay)
        return []

    def _parse_listing_page(self, html: str, category: str) -> list[Product]:
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("article", class_=re.compile(r"prd"))
        if not cards:
            cards = soup.find_all("article", attrs={"data-id": True})
        return [p for card in cards if (p := self._parse_card(card, category))]

    def _parse_card(self, card, category: str) -> Optional[Product]:
        try:
            name_el = card.find("h3", class_="name") or card.find("h3")
            if not name_el:
                return None
            name = name_el.get_text(strip=True)

            link_el = card.find("a", class_=re.compile(r"core")) or card.find("a", href=True)
            product_url = urljoin(BASE_URL, link_el["href"]) if link_el else None

            price_el = card.find("div", class_="prc") or card.find(class_=re.compile(r"prc"))
            price = self._parse_naira(price_el.get_text(strip=True) if price_el else "0")
            if price <= 0:
                return None

            old_el = card.find("div", class_="old") or card.find(class_=re.compile(r"\bold\b"))
            old_price = self._parse_naira(old_el.get_text(strip=True)) if old_el else None
            discount_pct = None
            if old_price and old_price > price:
                discount_pct = round((old_price - price) / old_price * 100, 1)

            rating_el = card.find("div", class_="stars _s") or card.find(class_=re.compile(r"stars"))
            rating = None
            if rating_el:
                aria = rating_el.get("aria-label", "")
                m = re.search(r"([\d.]+)\s*out of", aria)
                if m:
                    rating = float(m.group(1))

            reviews_el = card.find("div", class_="rev") or card.find(class_=re.compile(r"\brev\b"))
            num_reviews = None
            if reviews_el:
                m = re.search(r"\((\d+)\)", reviews_el.get_text(strip=True))
                if m:
                    num_reviews = int(m.group(1))

            img_el = card.find("img")
            image_url = (img_el.get("data-src") or img_el.get("src")) if img_el else None

            pid = hashlib.md5(f"{name}{price}".encode()).hexdigest()[:16]

            return Product(
                id=pid,
                external_id=card.get("data-id"),
                source="jumia",
                name=name,
                category=category,
                description="",
                price=int(price),
                old_price=int(old_price) if old_price else None,
                discount_pct=discount_pct,
                rating=rating,
                num_reviews=num_reviews,
                product_url=product_url,
                image_url=image_url,
                stock_info="In stock",
                brand=self._extract_brand(name),
                tags=[category, "jumia"],
            )
        except Exception as e:
            logger.debug(f"Card parse error: {e}")
            return None

    def _parse_naira(self, text: str) -> float:
        cleaned = re.sub(r"[^\d.]", "", text)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def _extract_brand(self, name: str) -> Optional[str]:
        name_up = name.upper()
        for brand in NIGERIAN_BRANDS:
            if brand.upper() in name_up:
                return brand
        first = name.split()[0] if name.split() else None
        return first if first and len(first) > 2 else None

    async def scrape_multiple_categories(
        self,
        categories: list[str],
        max_pages: int = 2,
        max_per_cat: int = 30,
    ) -> dict[str, list[Product]]:
        results: dict[str, list[Product]] = {}
        for cat in categories:
            results[cat] = await self.scrape_category(cat, max_pages, max_per_cat)
            await asyncio.sleep(self.delay)
        return results
