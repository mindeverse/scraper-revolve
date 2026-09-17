"""Revolve catalog scraper — mobile PLP HTML via curl_cffi (desktop Akamai-blocked)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

from config import CATEGORIES, cfg

logger = logging.getLogger(__name__)


def _session() -> cffi_requests.Session:
    sess = cffi_requests.Session(impersonate=cfg.CURL_IMPERSONATE)
    sess.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": cfg.USER_AGENT,
        }
    )
    return sess


def _with_page(url: str, page_num: int) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["pageNum"] = [str(page_num)]
    query = urlencode({k: v[-1] for k, v in qs.items()})
    return urlunparse(parsed._replace(query=query))


def _money(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"([\d,]+\.?\d*)", text.replace(",", ""))
    if not m:
        return None
    amount = m.group(1)
    # Detect currency symbol
    if "€" in text or "EUR" in text.upper():
        return f"{amount}EUR"
    return f"{amount}USD"


def _canonical_product_url(href: str, code: str) -> str:
    if href.startswith("http"):
        path = urlparse(href).path
    else:
        path = href.split("?")[0]
    # Prefer desktop-ish canonical without /mobile prefix for stability
    path = path.replace("/mobile/", "/", 1) if path.startswith("/mobile/") else path
    if "/dp/" not in path:
        path = f"/dp/{code}/"
    return urljoin(cfg.BASE_URL, path)


def _product_id(code: str, product_url: str) -> str:
    raw = f"{cfg.SOURCE}:{code}:{product_url}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _parse_cards(html: str, category: str, gender: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    products: list[dict[str, Any]] = []
    for li in soup.select("li.js-plp-container"):
        code = (li.get("id") or "").strip()
        if not code:
            sw = li.select_one("[data-code]")
            code = (sw.get("data-code") if sw else "") or ""
        if not code:
            continue

        a = li.select_one("a.js-plp-pdp-link[href*='/dp/'], a[href*='/dp/']")
        href = a.get("href") if a else None
        if not href:
            continue

        title_el = li.select_one(".js-plp-name, .product-name")
        brand_el = li.select_one(".js-plp-brand, .product-brand")
        img = li.select_one("img.js-plp-image")
        retail_el = li.select_one(".price__retail, .js-plp-price")
        sale_el = li.select_one(".price__sale, .js-plp-price-sale, .price__markdown")

        title = (title_el.get_text(strip=True) if title_el else None) or (
            img.get("alt") if img else None
        )
        if not title:
            continue

        brand = brand_el.get_text(strip=True) if brand_el else cfg.BRAND_COLUMN
        image_url = None
        if img:
            image_url = img.get("src") or img.get("data-src")
        if not image_url:
            # Construct from known CDN pattern
            image_url = f"https://is4.revolveassets.com/images/p4/n/tv/{code}_V1.jpg"

        # Prefer larger image for embedding
        if image_url and "/p4/n/tv/" in image_url:
            image_url = image_url.replace("/p4/n/tv/", "/p4/n/z/")
        elif image_url and "/p4/n/uv/" in image_url:
            image_url = image_url.replace("/p4/n/uv/", "/p4/n/z/")

        price = _money(retail_el.get_text(strip=True) if retail_el else None)
        sale = _money(sale_el.get_text(strip=True) if sale_el else None)
        if sale and price and sale == price:
            sale = None

        product_url = _canonical_product_url(href, code)
        colors = [
            el.get_text(strip=True)
            for el in li.select(".product-swatches__button span")
            if el.get_text(strip=True)
        ]

        products.append(
            {
                "id": _product_id(code, product_url),
                "source": cfg.SOURCE,
                "product_url": product_url,
                "affiliate_url": None,
                "image_url": image_url,
                "compressed_image_url": None,
                "back_image_url": f"https://is4.revolveassets.com/images/p4/n/z/{code}_V2.jpg",
                "brand": brand or cfg.BRAND_COLUMN,
                "title": title,
                "description": None,
                "category": category,
                "gender": gender,
                "price": price,
                "sale": sale,
                "metadata": json.dumps(
                    {
                        "product_code": code,
                        "colors": colors[:12],
                        "storefront": "revolve-mobile",
                    },
                    ensure_ascii=False,
                ),
                "size": None,
                "second_hand": False,
                "country": cfg.COUNTRY,
                "tags": [category, gender] if category else [gender],
                "additional_images": None,
                "other": None,
            }
        )
    return products


def _page_total(html: str) -> int | None:
    m = re.search(r"plp-item-count[^>]*>\s*([\d,]+)", html)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def scrape_category(
    sess: cffi_requests.Session,
    label: str,
    gender: str,
    url: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    empty_streak = 0
    total_hint: int | None = None

    for page in range(1, cfg.MAX_PAGES_PER_CATEGORY + 1):
        page_url = _with_page(url, page)
        try:
            r = sess.get(page_url, timeout=cfg.REQUEST_TIMEOUT)
            if r.status_code == 403:
                logger.error("Akamai 403 on %s page %d — abort category", label, page)
                break
            r.raise_for_status()
        except Exception as e:
            logger.warning("%s page %d fetch failed: %s", label, page, e)
            empty_streak += 1
            if empty_streak >= 3:
                break
            time.sleep(cfg.RATE_LIMIT_DELAY * 2)
            continue

        if total_hint is None:
            total_hint = _page_total(r.text)
            logger.info("%s (%s): reported total=%s", label, gender, total_hint)

        batch = _parse_cards(r.text, category=label, gender=gender)
        new = 0
        for p in batch:
            meta = json.loads(p["metadata"])
            code = meta.get("product_code")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            results.append(p)
            new += 1

        logger.info(
            "%s (%s) page %d: +%d (unique=%d)%s",
            label,
            gender,
            page,
            new,
            len(results),
            f" / ~{total_hint}" if total_hint else "",
        )

        if new == 0:
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0

        if total_hint and len(results) >= total_hint:
            break

        time.sleep(cfg.RATE_LIMIT_DELAY)

    return results


def scrape_all_categories() -> list[dict[str, Any]]:
    sess = _session()
    by_code: dict[str, dict[str, Any]] = {}

    for label, gender, url in CATEGORIES:
        logger.info("Scraping category %s / %s", gender, label)
        try:
            products = scrape_category(sess, label, gender, url)
        except Exception as e:
            logger.exception("Category %s/%s failed: %s", gender, label, e)
            continue
        for p in products:
            code = json.loads(p["metadata"]).get("product_code")
            if not code:
                continue
            # Prefer first-seen category; keep if new
            if code not in by_code:
                by_code[code] = p

    products = list(by_code.values())
    logger.info("Scrape complete: %d unique products", len(products))
    return products
