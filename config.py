"""Revolve scraper configuration — EN+USD via mobile PLP (desktop Akamai-blocked)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# Top-level PLPs (non-overlapping). Skip broken beauty slug (falls through to junk totals).
# pageNum pagination; ~50 products/page. Dedup by product code.
CATEGORIES: list[tuple[str, str, str]] = [
    # (label, gender, url)
    ("Clothing", "Women", "https://www.revolve.com/mobile/clothing/br/3699fc/?s=c&c=Clothing&d=Womens"),
    ("Shoes", "Women", "https://www.revolve.com/mobile/shoes/br/3f40a9/?s=c&c=Shoes&d=Womens"),
    ("Bags", "Women", "https://www.revolve.com/mobile/bags/br/2df9df/?s=c&c=Bags&d=Womens"),
    ("Jewelry", "Women", "https://www.revolve.com/mobile/jewelry/br/5d8a4a/?s=c&c=Jewelry&d=Womens"),
    ("Accessories", "Women", "https://www.revolve.com/mobile/accessories/br/2fa629/?s=c&c=Accessories&d=Womens"),
    ("Clothing", "Men", "https://www.revolve.com/mobile/mens/clothing/br/a8d011/?d=Mens"),
    ("Shoes", "Men", "https://www.revolve.com/mobile/mens/shoes/br/3f40a9/?d=Mens"),
    ("Accessories", "Men", "https://www.revolve.com/mobile/mens/accessories/br/2fa629/?d=Mens"),
]


@dataclass
class Config:
    BRAND_NAME: str = "REVOLVE"
    SOURCE: str = "scraper-revolve"
    BRAND_COLUMN: str = "REVOLVE"  # overridden per-product with designer brand
    SECOND_HAND: bool = False
    LANDING_PAGE: str = "https://www.revolve.com/"
    BASE_URL: str = "https://www.revolve.com"
    CURRENCY: str = "USD"
    COUNTRY: str = "US"
    LANG: str = "en-US"

    SUPABASE_URL: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    SUPABASE_KEY: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))

    EMBEDDING_MODEL: str = "google/siglip-base-patch16-384"
    EMBEDDING_DIM: int = 768
    EMBEDDING_VERSION: int = 2
    RATE_LIMIT_DELAY: float = 0.35
    BATCH_SIZE: int = 5
    STALE_MISS_THRESHOLD: int = 2
    REQUEST_TIMEOUT: int = 45
    SCRAPE_WORKERS: int = 2
    DOWNLOAD_WORKERS: int = 30
    TEXT_EMBED_BATCH_SIZE: int = 32
    PAGE_SIZE: int = 50
    USER_AGENT: str = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    )
    GENDER_DEFAULT: str = "Unisex"
    CURL_IMPERSONATE: str = "chrome131"
    MAX_PAGES_PER_CATEGORY: int = 2000


cfg = Config()
