"""Image and text embeddings using SigLIP (768-dim) - local model with multiprocessing."""
import gc
import io
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import requests
import torch
from PIL import Image

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor
from transformers import SiglipModel, SiglipTokenizer

from config import cfg

logger = logging.getLogger(__name__)

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
INFERENCE_BATCH_SIZE = 64
CHECKPOINT_DIR = "logs"


# ── Worker process function ────────────────────────────────────────

def _worker_embed(
    worker_id: int,
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    num_workers: int,
    shutdown_event: mp.Event,
):
    """Worker process: loads its own model, embeds images from queue."""
    device = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() // num_workers))

    logger.info("[Worker %d] Loading SigLIP model...", worker_id)
    processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    logger.info("[Worker %d] Model loaded.", worker_id)

    while not shutdown_event.is_set():
        try:
            item = input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if item is None:  # Poison pill
            break

        idx, url, view_type, image_bytes = item

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb_tensor = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                emb_tensor = outputs.last_hidden_state[:, 0, :]
            else:
                emb_tensor = outputs
            embedding = emb_tensor.cpu().float().numpy().flatten().tolist()
            result_queue.put((idx, url, view_type, embedding, None))
        except Exception as e:
            result_queue.put((idx, url, view_type, None, str(e)))

    logger.info("[Worker %d] Shutting down.", worker_id)


def _download_image_bytes(image_url: str) -> Optional[bytes]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    try:
        resp = requests.get(image_url, timeout=15, headers=headers)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning("Failed to download image %s: %s", image_url, e)
        return None


def _embed_images_batch(images: list[Image.Image]) -> list[Optional[list[float]]]:
    """Fallback single-process batch embedding (for small batches / text)."""
    processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    device = "cpu"
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    results: list[Optional[list[float]]] = [None] * len(images)
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        for i in range(emb_tensor.shape[0]):
            results[i] = emb_tensor[i].cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Batch embed failed (size %d): %s", len(images), e)
    return results


def _embed_single(image: Image.Image) -> Optional[list[float]]:
    """Fallback single-process single-image embedding."""
    processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    device = "cpu"
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed image: %s", e)
        return None


def get_text_embedding(text: str) -> Optional[list[float]]:
    if not text or not str(text).strip():
        return None
    processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
    device = "cpu"
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    try:
        inputs = tokenizer(
            text=[str(text).strip()],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb_tensor = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb_tensor = outputs.last_hidden_state[:, 0, :]
        else:
            emb_tensor = outputs
        return emb_tensor.cpu().float().numpy().flatten().tolist()
    except Exception as e:
        logger.warning("Failed to embed text: %s", e)
        return None


def get_text_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[Optional[list[float]]]:
    """Embed multiple texts in batches for much faster throughput."""
    if not texts:
        return []
    tokenizer = SiglipTokenizer.from_pretrained(MODEL_NAME)
    device = "cpu"
    model = SiglipModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    all_results: list[Optional[list[float]]] = [None] * len(texts)

    valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
    if not valid_indices:
        return all_results

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start: batch_start + batch_size]
        batch_texts = [str(texts[i]).strip() for i in batch_indices]

        try:
            inputs = tokenizer(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb_tensor = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                emb_tensor = outputs.last_hidden_state[:, 0, :]
            else:
                emb_tensor = outputs
            for j, idx in enumerate(batch_indices):
                all_results[idx] = emb_tensor[j].cpu().float().numpy().flatten().tolist()
        except Exception as e:
            logger.warning("Batch text embed failed (size %d): %s", len(batch_texts), e)
            for idx in batch_indices:
                try:
                    all_results[idx] = get_text_embedding(texts[idx])
                except Exception:
                    pass

    return all_results


def _build_info_text(product: dict[str, Any]) -> str:
    parts = []
    parts.append(f"Brand: {product.get('brand', 'Reserved')}")
    parts.append(f"Product: {product.get('title', '')}")
    if product.get("category"):
        parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        parts.append(f"Gender: {product['gender']}")
    if product.get("price"):
        parts.append(f"Price: {product['price']}")
    if product.get("sale"):
        parts.append(f"Sale price: {product['sale']}")
    if product.get("description"):
        parts.append(f"Description: {product['description']}")
    try:
        metadata = json.loads(product.get("metadata") or "{}")
        if metadata.get("color_name"):
            parts.append(f"Color: {metadata['color_name']}")
        if metadata.get("materials"):
            parts.append(f"Materials: {', '.join(metadata['materials'])}")
    except (json.JSONDecodeError, TypeError):
        pass
    return " | ".join(parts)


# ── Checkpoint helpers ──────────────────────────────────────────────

_checkpoint_source: str = ""
_checkpoint_data: dict[str, dict[str, Any]] = {}


def _checkpoint_path(source: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"embed_checkpoint_{source}.pkl")


def load_checkpoint(source: str) -> dict[str, dict[str, Any]]:
    path = _checkpoint_path(source)
    if not os.path.exists(path):
        json_path = os.path.join(CHECKPOINT_DIR, f"embed_checkpoint_{source}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                logger.info("Migrated legacy JSON checkpoint: %d products", len(data))
                _save_checkpoint(source, data)
                os.remove(json_path)
                return data
            except Exception as e:
                logger.warning("Failed to load legacy checkpoint: %s", e)
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("Loaded checkpoint: %d products already embedded", len(data))
        return data
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return {}


def _save_checkpoint(source: str, data: dict[str, dict[str, Any]]):
    path = _checkpoint_path(source)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def clear_checkpoint(source: str):
    path = _checkpoint_path(source)
    if os.path.exists(path):
        os.remove(path)
        logger.info("Cleared embedding checkpoint")


def _log_memory(label: str):
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if raw > 1_000_000:
            mb = raw / (1024 * 1024)
        else:
            mb = raw / 1024
        logger.info("  [MEM %s] RSS: %.0f MB", label, mb)
    except Exception:
        pass


def _setup_signal_handlers(source: str):
    global _checkpoint_source, _checkpoint_data
    _checkpoint_source = source

    def handler(signum, frame):
        logger.warning("Received signal %d, saving checkpoint before exit...", signum)
        if _checkpoint_data:
            try:
                _save_checkpoint(_checkpoint_source, _checkpoint_data)
                logger.warning("Checkpoint saved (%d products)", len(_checkpoint_data))
            except Exception as e:
                logger.error("Failed to save checkpoint: %s", e)
        sys.exit(143)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ── Main embedding pipeline ────────────────────────────────────────

def embed_products(
    products: list[dict[str, Any]],
    existing_embeddings: dict[str, dict[str, Any]] | None = None,
    source: str = "scraper-reserved",
    on_batch_done: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if existing_embeddings is None:
        existing_embeddings = {}

    stats = {
        "front_embeddings": 0,
        "back_embeddings": 0,
        "text_embeddings": 0,
        "skipped": 0,
    }

    checkpoint = load_checkpoint(source)
    for url, ckpt_data in checkpoint.items():
        if url not in existing_embeddings:
            existing_embeddings[url] = ckpt_data

    _log_memory("after model load")
    _setup_signal_handlers(source)

    # ── Determine which images need embedding ──────────────────────
    all_needed: list[tuple[int, str, str]] = []
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if image_url and (
            not existing
            or image_url != existing_image_url
            or not existing.get("image_embedding")
        ):
            all_needed.append((i, image_url, "front"))

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if back_url and (
            not existing
            or back_url != existing_back_url
            or not existing.get("back_image_embedding")
        ):
            all_needed.append((i, back_url, "back"))

    logger.info("Need to embed %d images total", len(all_needed))
    _log_memory("before embedding loop")

    # ── Restore existing embeddings from checkpoint ────────────────
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})
        if existing:
            if existing.get("image_embedding") and not product.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]
            if existing.get("back_image_embedding") and not product.get("back_image_embedding"):
                product["back_image_embedding"] = existing["back_image_embedding"]
            if existing.get("info_embedding") and not product.get("info_embedding"):
                product["info_embedding"] = existing["info_embedding"]

    global _checkpoint_data
    checkpoint_data: dict[str, dict[str, Any]] = dict(checkpoint)
    _checkpoint_data = checkpoint_data
    completed_urls: set[str] = set(checkpoint.keys())

    if not all_needed:
        logger.info("All images already embedded (from checkpoint/existing), skipping image embedding")
    else:
        # ── Determine worker count ─────────────────────────────────
        num_cpus = os.cpu_count() or 2
        num_workers = min(num_cpus, cfg.DOWNLOAD_WORKERS, 8)
        logger.info("  Using %d inference workers (%d CPUs available)", num_workers, num_cpus)

        # ── Shared queues ──────────────────────────────────────────
        input_queue: mp.Queue = mp.Queue(maxsize=500)
        result_queue: mp.Queue = mp.Queue(maxsize=500)
        shutdown_event = mp.Event()

        # ── Start worker processes ─────────────────────────────────
        workers = []
        for w in range(num_workers):
            p = mp.Process(
                target=_worker_embed,
                args=(w, input_queue, result_queue, num_workers, shutdown_event),
                daemon=True,
            )
            p.start()
            workers.append(p)
        logger.info("  Started %d worker processes", num_workers)

        # ── Download thread: fetch images, put bytes on input queue ─
        total_images = len(all_needed)
        downloaded_count = [0]
        download_done = threading.Event()

        # Deduplicate URLs
        url_to_items: dict[str, list[tuple[int, str, str]]] = {}
        for item in all_needed:
            idx, url, view = item
            url_to_items.setdefault(url, []).append(item)

        unique_urls = list(url_to_items.keys())
        logger.info("  %d unique image URLs to download (%d total embeddings)",
                     len(unique_urls), total_images)

        def _download_worker():
            with ThreadPoolExecutor(max_workers=cfg.DOWNLOAD_WORKERS) as pool:
                futures = {pool.submit(_download_image_bytes, url): url for url in unique_urls}
                for fut in as_completed(futures):
                    url = futures[fut]
                    try:
                        img_bytes = fut.result()
                    except Exception:
                        img_bytes = None
                    if img_bytes is not None:
                        for idx, url_val, view_type in url_to_items[url]:
                            try:
                                input_queue.put((idx, url, view_type, img_bytes), timeout=5)
                            except Exception:
                                pass
                    downloaded_count[0] += 1
                    if downloaded_count[0] % 100 == 0:
                        logger.info("    Downloads: %d/%d", downloaded_count[0], len(unique_urls))
            download_done.set()
            # Send poison pills
            for _ in range(num_workers):
                try:
                    input_queue.put(None, timeout=5)
                except Exception:
                    pass

        dl_thread = threading.Thread(target=_download_worker, daemon=True)
        dl_thread.start()

        # ── Main thread: collect results from workers ──────────────
        total_inferred = 0
        checkpoint_interval = 500
        last_checkpoint_count = 0
        workers_alive = num_workers
        failed_count = 0

        while workers_alive > 0:
            try:
                result = result_queue.get(timeout=2.0)
            except queue.Empty:
                # Check if all workers are done
                workers_alive = sum(1 for p in workers if p.is_alive())
                continue

            if result is None:
                workers_alive -= 1
                continue

            idx, url, view_type, embedding, error = result
            product = products[idx]
            purl = product.get("product_url", "")

            if embedding is not None:
                if view_type == "front":
                    product["image_embedding"] = embedding
                    stats["front_embeddings"] += 1
                else:
                    product["back_image_embedding"] = embedding
                    stats["back_embeddings"] += 1
                if purl not in checkpoint_data:
                    checkpoint_data[purl] = {}
                if view_type == "front":
                    checkpoint_data[purl]["image_embedding"] = embedding
                    checkpoint_data[purl]["image_url"] = product.get("image_url", "")
                else:
                    checkpoint_data[purl]["back_image_embedding"] = embedding
                    checkpoint_data[purl]["back_image_url"] = product.get("back_image_url", "")
            else:
                stats["skipped"] += 1
                failed_count += 1

            total_inferred += 1

            if total_inferred % 100 == 0:
                logger.info("    Embedded %d/%d images (front=%d, back=%d, failed=%d)",
                            total_inferred, total_images,
                            stats["front_embeddings"], stats["back_embeddings"], failed_count)

            if total_inferred - last_checkpoint_count >= checkpoint_interval:
                _save_checkpoint(source, checkpoint_data)
                last_checkpoint_count = total_inferred

        # Wait for download thread
        dl_thread.join(timeout=10)

        # Final checkpoint
        _save_checkpoint(source, checkpoint_data)
        completed_urls.update(checkpoint_data.keys())

        logger.info(
            "  Image embedding done: front=%d, back=%d, skipped=%d (checkpoint saved)",
            stats["front_embeddings"], stats["back_embeddings"], stats["skipped"],
        )

    # ── Text embeddings (single process, batched) ──────────────────
    checkpoint_data = load_checkpoint(source)
    _checkpoint_data = checkpoint_data

    logger.info("  Generating text embeddings for %d products...", len(products))

    product_texts: list[tuple[int, str, str, dict[str, Any]]] = []
    for i, product in enumerate(products):
        product_url = product.get("product_url", "")
        existing = existing_embeddings.get(product_url, {})

        image_url = product.get("image_url", "")
        existing_image_url = existing.get("image_url", "")
        if not (image_url and (not existing or image_url != existing_image_url)):
            if existing and existing.get("image_embedding"):
                product["image_embedding"] = existing["image_embedding"]

        back_url = product.get("back_image_url")
        existing_back_url = existing.get("back_image_url")
        if not back_url and not (not existing or back_url != existing_back_url):
            if existing and existing.get("back_image_embedding"):
                product["back_image_embedding"] = existing["back_image_embedding"]

        info_text = _build_info_text(product)
        existing_info_text = ""
        if existing:
            existing_info_text = _build_info_text(existing)

        if not existing or info_text != existing_info_text:
            product_texts.append((i, product_url, info_text, existing))
        else:
            if existing and existing.get("info_embedding"):
                product["info_embedding"] = existing["info_embedding"]

    if product_texts:
        batch_texts = [t for _, _, t, _ in product_texts]
        batch_embeddings = get_text_embeddings_batch(batch_texts, batch_size=cfg.TEXT_EMBED_BATCH_SIZE)

        for (i, product_url, _, existing), text_emb in zip(product_texts, batch_embeddings):
            product = products[i]
            if text_emb:
                product["info_embedding"] = text_emb
                stats["text_embeddings"] += 1
                if product_url not in checkpoint_data:
                    checkpoint_data[product_url] = {}
                checkpoint_data[product_url]["info_embedding"] = text_emb

        _save_checkpoint(source, checkpoint_data)
        logger.info("  Text embedding complete: %d/%d embedded", stats["text_embeddings"], len(product_texts))
    else:
        logger.info("  All text embeddings up to date, skipping")

    _save_checkpoint(source, checkpoint_data)
    _checkpoint_data.clear()

    return products, stats
