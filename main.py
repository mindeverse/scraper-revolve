"""Revolve scraper — scrape Revolve mobile PLP, embed with local SigLIP, upsert Supabase."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg
from embeddings import embed_products
from scraper import scrape_all_categories
from supabase_client import (
    SupabaseClient,
    handle_stale_products,
    load_stale_tracker,
    save_stale_tracker,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scraper-revolve")


def _needs_update(existing: dict[str, Any], scraped: dict[str, Any]) -> bool:
    compare_fields = [
        "title",
        "price",
        "sale",
        "category",
        "description",
        "image_url",
        "back_image_url",
        "additional_images",
        "size",
        "tags",
        "metadata",
        "gender",
    ]
    for field in compare_fields:
        if str(existing.get(field) or "") != str(scraped.get(field) or ""):
            return True
    # Embedding presence is attached as boolean flags when vectors are not fetched.
    has_image = existing.get("image_embedding") or existing.get("_has_image_embedding")
    has_info = existing.get("info_embedding") or existing.get("_has_info_embedding")
    if scraped.get("image_url") and not has_image:
        return True
    if not has_info:
        return True
    return False


def _to_db_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": record["id"],
        "source": record["source"],
        "product_url": record["product_url"],
        "affiliate_url": record.get("affiliate_url"),
        "image_url": record["image_url"],
        "compressed_image_url": record.get("compressed_image_url"),
        "back_image_url": record.get("back_image_url"),
        "brand": record.get("brand"),
        "title": record["title"],
        "description": record.get("description"),
        "category": record.get("category"),
        "gender": record.get("gender"),
        "price": record.get("price"),
        "sale": record.get("sale"),
        "metadata": record.get("metadata"),
        "size": record.get("size"),
        "second_hand": record.get("second_hand", False),
        "country": record.get("country"),
        "tags": record.get("tags"),
        "additional_images": record.get("additional_images"),
        "other": record.get("other"),
    }
    if record.get("image_embedding"):
        row["image_embedding"] = record["image_embedding"]
    if record.get("back_image_embedding") is not None:
        row["back_image_embedding"] = record["back_image_embedding"]
    if record.get("info_embedding"):
        row["info_embedding"] = record["info_embedding"]
    return row


def _assign_chunk(products: list[dict[str, Any]], chunk_index: int, total_chunks: int) -> list[dict[str, Any]]:
    """Deterministically assign products to chunks based on URL hash."""
    assigned = []
    for p in products:
        url = p.get("product_url", "")
        h = int(hashlib.md5(url.encode()).hexdigest(), 16)
        if h % total_chunks == chunk_index:
            assigned.append(p)
    return assigned


def run_scrape() -> dict[str, Any]:
    """Phase 1: scrape only — no embedding. Saves scraped products to JSON artifact."""
    logger.info("=== Scrape phase start ===")
    if not cfg.SUPABASE_URL or not cfg.SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY missing")
        sys.exit(1)

    supa = SupabaseClient()

    # Scrape first, then DB reads (sequential). Parallel DB+scrape competed with
    # dying embed workers and caused "Server disconnected" → empty existing → full re-embed.
    scraped = scrape_all_categories()
    existing = supa.fetch_existing_products(cfg.SOURCE)
    emb_urls = supa.fetch_urls_with_embeddings(cfg.SOURCE)

    if not scraped:
        logger.error("No products scraped")
        return {"total": 0}

    # Attach cheap embedding-presence flags (vectors intentionally not loaded).
    for url, row in existing.items():
        row["_has_image_embedding"] = url in emb_urls.get("image", set())
        row["_has_info_embedding"] = url in emb_urls.get("info", set())
        row["_has_back_embedding"] = url in emb_urls.get("back", set())

    to_embed: list[dict[str, Any]] = []
    unchanged = 0
    for record in scraped:
        url = record["product_url"]
        if url not in existing:
            to_embed.append(record)
        elif _needs_update(existing[url], record):
            prev = existing[url]
            # No vector payloads in artifact — embed workers regenerate as needed.
            record["_existing"] = {
                "image_url": prev.get("image_url"),
                "back_image_url": prev.get("back_image_url"),
                "_has_image_embedding": prev.get("_has_image_embedding"),
                "_has_info_embedding": prev.get("_has_info_embedding"),
                "_has_back_embedding": prev.get("_has_back_embedding"),
            }
            to_embed.append(record)
        else:
            unchanged += 1

    # Lightweight map for embed_products (no vectors).
    existing_embeddings = {}
    for purl, row in existing.items():
        existing_embeddings[purl] = {
            "image_url": row.get("image_url"),
            "back_image_url": row.get("back_image_url"),
            "_has_image_embedding": row.get("_has_image_embedding"),
            "_has_info_embedding": row.get("_has_info_embedding"),
            "_has_back_embedding": row.get("_has_back_embedding"),
        }

    Path("logs").mkdir(exist_ok=True)
    # Drop heavy/unused keys from existing before serializing artifact
    existing_light = {
        u: {k: v for k, v in row.items() if not k.startswith("image_embedding") and not k.startswith("back_image_embedding") and not k.startswith("info_embedding")}
        for u, row in existing.items()
    }
    # Keep artifact lean: full to_embed rows + URL list for stale detection (not full scraped rows).
    output = {
        "scraped_urls": [r["product_url"] for r in scraped],
        "to_embed": to_embed,
        "existing_embeddings": existing_embeddings,
        "existing": existing_light,
        "unchanged": unchanged,
        "scraped_count": len(scraped),
    }
    out_path = Path("logs/scrape_output.json")
    out_path.write_text(json.dumps(output), encoding="utf-8")
    logger.info("Saved scrape output: %d products (%d to embed, %d unchanged)",
                len(scraped), len(to_embed), unchanged)
    return {"total": len(scraped), "to_embed": len(to_embed), "unchanged": unchanged}


def run_embed_only(chunk_index: int, total_chunks: int) -> dict[str, Any]:
    """Phase 2: embed + upsert a single chunk. Reads scrape_output.json from artifact."""
    logger.info("=== Embed phase start (chunk %d/%d) ===", chunk_index + 1, total_chunks)

    scrape_path = Path("logs/scrape_output.json")
    if not scrape_path.exists():
        logger.error("scrape_output.json not found — scrape phase must run first")
        sys.exit(1)

    data = json.loads(scrape_path.read_text(encoding="utf-8"))
    to_embed = data["to_embed"]
    existing_embeddings = data["existing_embeddings"]
    existing = data["existing"]

    chunk = _assign_chunk(to_embed, chunk_index, total_chunks)
    logger.info("Chunk %d/%d: %d products to embed", chunk_index + 1, total_chunks, len(chunk))

    if not chunk:
        logger.info("No products in this chunk, done")
        return {"embedded": 0}

    products_embedded, embed_stats = embed_products(
        chunk,
        existing_embeddings=existing_embeddings,
        source=f"{cfg.SOURCE}-chunk{chunk_index}",
    )

    supa = SupabaseClient()
    rows = [_to_db_row(r) for r in products_embedded]
    ok, fail = supa.upsert_products(rows, batch_size=cfg.BATCH_SIZE)

    seen_urls = set(data.get("scraped_urls") or [p["product_url"] for p in data.get("scraped", [])])
    stale_tracker = load_stale_tracker()
    deleted, updated_tracker = handle_stale_products(
        supa,
        cfg.SOURCE,
        seen_urls,
        existing,
        stale_tracker,
        threshold=cfg.STALE_MISS_THRESHOLD,
    )
    save_stale_tracker(updated_tracker)

    summary = {
        "chunk": f"{chunk_index + 1}/{total_chunks}",
        "embedded": len(products_embedded),
        "front_embeddings": embed_stats.get("front_embeddings", 0),
        "back_embeddings": embed_stats.get("back_embeddings", 0),
        "text_embeddings": embed_stats.get("text_embeddings", 0),
        "upsert_ok": ok,
        "upsert_fail": fail,
        "stale_deleted": deleted,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    Path("logs").mkdir(exist_ok=True)
    summary_path = Path(f"logs/chunk{chunk_index}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=== Chunk %d/%d summary ===", chunk_index + 1, total_chunks)
    for k, v in summary.items():
        logger.info("%s: %s", k, v)
    return summary


def run() -> dict[str, Any]:
    """Default single-process mode (no chunking)."""
    logger.info("=== Revolve scraper start (single process) ===")
    if not cfg.SUPABASE_URL or not cfg.SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY missing")
        sys.exit(1)

    supa = SupabaseClient()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_scrape = executor.submit(scrape_all_categories)
        future_existing = executor.submit(supa.fetch_existing_products, cfg.SOURCE)
        scraped = future_scrape.result()
        existing = future_existing.result()

    if not scraped:
        logger.error("No products scraped")
        return {"new": 0, "updated": 0, "unchanged": 0}

    to_embed: list[dict[str, Any]] = []
    unchanged = 0
    for record in scraped:
        url = record["product_url"]
        if url not in existing:
            to_embed.append(record)
        elif _needs_update(existing[url], record):
            prev = existing[url]
            record["_existing"] = {
                "image_url": prev.get("image_url"),
                "back_image_url": prev.get("back_image_url"),
                "image_embedding": prev.get("image_embedding"),
                "back_image_embedding": prev.get("back_image_embedding"),
                "info_embedding": prev.get("info_embedding"),
            }
            to_embed.append(record)
        else:
            unchanged += 1

    logger.info(
        "Diff: %d to process, %d unchanged, %d existing in DB",
        len(to_embed),
        unchanged,
        len(existing),
    )

    existing_embeddings = {}
    for purl, row in existing.items():
        existing_embeddings[purl] = {
            "image_url": row.get("image_url"),
            "back_image_url": row.get("back_image_url"),
            "image_embedding": row.get("image_embedding"),
            "back_image_embedding": row.get("back_image_embedding"),
            "info_embedding": row.get("info_embedding"),
        }

    products_embedded, embed_stats = embed_products(
        to_embed,
        existing_embeddings=existing_embeddings,
        source=cfg.SOURCE,
    )

    rows = [_to_db_row(r) for r in products_embedded]
    ok, fail = supa.upsert_products(rows, batch_size=cfg.BATCH_SIZE)

    seen_urls = {p["product_url"] for p in scraped}
    stale_tracker = load_stale_tracker()
    deleted, updated_tracker = handle_stale_products(
        supa,
        cfg.SOURCE,
        seen_urls,
        existing,
        stale_tracker,
        threshold=cfg.STALE_MISS_THRESHOLD,
    )
    save_stale_tracker(updated_tracker)

    summary = {
        "new": sum(1 for r in products_embedded if r["product_url"] not in existing),
        "updated": sum(1 for r in products_embedded if r["product_url"] in existing),
        "unchanged": unchanged,
        "upsert_ok": ok,
        "upsert_fail": fail,
        "front_embeddings": embed_stats.get("front_embeddings", 0),
        "back_embeddings": embed_stats.get("back_embeddings", 0),
        "text_embeddings": embed_stats.get("text_embeddings", 0),
        "stale_deleted": deleted,
        "total_scraped": len(scraped),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    Path("logs").mkdir(exist_ok=True)
    Path("logs/last_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=== Run summary ===")
    for k, v in summary.items():
        logger.info("%s: %s", k, v)
    logger.info("=== Revolve scraper complete ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revolve scraper")
    parser.add_argument("--mode", choices=["scrape", "embed", "full"], default="full",
                        help="scrape=save products only, embed=process chunk only, full=single-process (default)")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Chunk index (0-based) for embed mode")
    parser.add_argument("--total-chunks", type=int, default=1,
                        help="Total number of chunks for embed mode")
    args = parser.parse_args()

    if args.mode == "scrape":
        run_scrape()
    elif args.mode == "embed":
        run_embed_only(args.chunk, args.total_chunks)
    else:
        run()
