"""Supabase client — batch upsert, diff helpers, stale cleanup."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import create_client

from config import cfg

logger = logging.getLogger(__name__)

STALE_TRACKER_FILE = Path("logs") / "stale_tracker.json"
FAILED_PRODUCTS_LOG = Path("logs") / "failed_products.log"

UPSERT_WORKERS = 4


def _with_retries(label: str, fn, attempts: int = 6, base_sleep: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep_s = base_sleep * (2 ** i)
            logger.warning("%s attempt %d/%d failed: %s — sleep %.1fs", label, i + 1, attempts, e, sleep_s)
            time.sleep(sleep_s)
    raise last


class SupabaseClient:
    def __init__(self):
        if not cfg.SUPABASE_URL or not cfg.SUPABASE_KEY:
            raise ValueError("Set SUPABASE_URL and SUPABASE_KEY env vars")
        self.client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_KEY)

    def fetch_existing_products(self, source: str) -> dict[str, dict[str, Any]]:
        """Load existing product scalars for source (no vector columns — avoids statement timeouts)."""
        page_size = 1000
        select_cols = (
            "id,product_url,title,price,sale,category,description,"
            "image_url,back_image_url,additional_images,size,tags,metadata,gender"
        )

        def _load() -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            start = 0
            while True:
                response = (
                    self.client.table("products")
                    .select(select_cols)
                    .eq("source", source)
                    .range(start, start + page_size - 1)
                    .execute()
                )
                rows = response.data or []
                if not rows:
                    break
                for row in rows:
                    result[row["product_url"]] = row
                if len(rows) < page_size:
                    break
                start += page_size
            return result

        try:
            result = _with_retries("fetch_existing_products", _load)
            logger.info("Fetched %d existing products from DB", len(result))
            return result
        except Exception as e:
            logger.error("Failed to fetch existing products after retries: %s", e)
            return {}

    def fetch_urls_with_embeddings(self, source: str) -> dict[str, set[str]]:
        """Return sets of product_urls that already have image / info / back embeddings."""
        page_size = 1000
        col_map = {
            "image": "image_embedding",
            "info": "info_embedding",
            "back": "back_image_embedding",
        }

        def _load_kind(col: str) -> set[str]:
            urls: set[str] = set()
            start = 0
            while True:
                response = (
                    self.client.table("products")
                    .select("product_url")
                    .eq("source", source)
                    .not_.is_(col, "null")
                    .range(start, start + page_size - 1)
                    .execute()
                )
                rows = response.data or []
                if not rows:
                    break
                for row in rows:
                    urls.add(row["product_url"])
                if len(rows) < page_size:
                    break
                start += page_size
            return urls

        out: dict[str, set[str]] = {"image": set(), "info": set(), "back": set()}
        try:
            for kind, col in col_map.items():
                out[kind] = _with_retries(f"fetch_urls_with_embeddings:{kind}", lambda c=col: _load_kind(c))
                logger.info("URLs with %s embedding: %d", kind, len(out[kind]))
        except Exception as e:
            logger.error("Failed to fetch embedding URL sets after retries: %s", e)
        return out

    def _upsert_single_batch(self, batch: list[dict[str, Any]], batch_idx: int) -> tuple[int, int]:
        """Upsert a single batch with retries."""
        ok = 0
        fail = 0
        batch_ok = False
        for attempt in range(3):
            try:
                self.client.table("products").upsert(
                    batch, on_conflict="source,product_url"
                ).execute()
                ok += len(batch)
                batch_ok = True
                break
            except Exception as e:
                logger.warning("Batch %d attempt %d/3 failed: %s", batch_idx + 1, attempt + 1, e)
                time.sleep(2 ** (attempt + 1))

        if not batch_ok:
            # Fall back to single-row upserts
            logger.warning("Falling back to single-row upsert for batch %d", batch_idx + 1)
            for row in batch:
                row_ok = False
                for attempt in range(3):
                    try:
                        self.client.table("products").upsert(
                            row, on_conflict="source,product_url"
                        ).execute()
                        ok += 1
                        row_ok = True
                        break
                    except Exception as e:
                        logger.warning("Single upsert attempt %d/3 failed: %s", attempt + 1, e)
                        time.sleep(2 ** (attempt + 1))
                if not row_ok:
                    fail += 1
                    self._log_failed_products([row])
        return ok, fail

    def upsert_products(self, products: list[dict[str, Any]], batch_size: int = 50) -> tuple[int, int]:
        total = len(products)
        batches = [products[i : i + batch_size] for i in range(0, total, batch_size)]
        ok = 0
        fail = 0

        if len(batches) <= UPSERT_WORKERS:
            # Few batches — run sequentially to avoid connection issues
            for i, batch in enumerate(batches):
                logger.info("Upserting batch %d/%d (%d products)", i + 1, len(batches), len(batch))
                b_ok, b_fail = self._upsert_single_batch(batch, i)
                ok += b_ok
                fail += b_fail
        else:
            # Many batches — parallel upserts for throughput
            logger.info("Parallel upsert: %d batches with %d workers", len(batches), UPSERT_WORKERS)
            with ThreadPoolExecutor(max_workers=UPSERT_WORKERS) as executor:
                futures = {
                    executor.submit(self._upsert_single_batch, batch, i): i
                    for i, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        b_ok, b_fail = future.result()
                        ok += b_ok
                        fail += b_fail
                        logger.info("  Batch %d/%d done (ok=%d, fail=%d)", i + 1, len(batches), b_ok, b_fail)
                    except Exception as e:
                        logger.error("Batch %d raised exception: %s", i + 1, e)
                        fail += len(batches[i])

        return ok, fail

    def delete_product(self, product_id: str) -> None:
        self.client.table("products").delete().eq("id", product_id).execute()

    def _log_failed_products(self, products: list[dict[str, Any]]) -> None:
        try:
            FAILED_PRODUCTS_LOG.parent.mkdir(exist_ok=True)
            with open(FAILED_PRODUCTS_LOG, "a") as f:
                for p in products:
                    f.write(
                        f"{datetime.now(timezone.utc).isoformat()} | "
                        f"{p.get('source', '')} | {p.get('product_url', '')}\n"
                    )
        except Exception as e:
            logger.error("Failed to log failed products: %s", e)


def load_stale_tracker() -> dict[str, int]:
    if STALE_TRACKER_FILE.exists():
        try:
            return json.loads(STALE_TRACKER_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load stale tracker: %s", e)
    return {}


def save_stale_tracker(tracker: dict[str, int]) -> None:
    try:
        STALE_TRACKER_FILE.parent.mkdir(exist_ok=True)
        STALE_TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save stale tracker: %s", e)


def handle_stale_products(
    supa: SupabaseClient,
    source: str,
    seen_urls: set[str],
    existing_products: dict[str, dict[str, Any]],
    stale_tracker: dict[str, int],
    threshold: int = 2,
) -> tuple[int, dict[str, int]]:
    deleted = 0
    updated_tracker: dict[str, int] = {}
    for product_url, row in existing_products.items():
        if product_url in seen_urls:
            updated_tracker[product_url] = 0
        else:
            current_misses = stale_tracker.get(product_url, 0) + 1
            updated_tracker[product_url] = current_misses
            if current_misses >= threshold:
                try:
                    supa.delete_product(row["id"])
                    logger.info("Deleted stale product: %s", product_url[:80])
                    deleted += 1
                    updated_tracker.pop(product_url, None)
                except Exception as e:
                    logger.error("Failed to delete stale %s: %s", product_url[:60], e)
    return deleted, updated_tracker
