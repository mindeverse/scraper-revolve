# scraper-revolve

REVOLVE aggregator scraper for Finds.

- **Source:** `scraper-revolve`
- **Locale:** EN+USD (EUR cookie ignored; ship USD)
- **Access:** mobile PLP HTML (`/mobile/...`) via iPhone UA + `curl_cffi` (desktop PLP Akamai 403)
- **Architecture:** scrape job → `scrape_output.json` artifact → 30 parallel embed chunks (local SigLIP)
- **Cron:** `23 8 * * 2,4,6` UTC (Tue/Thu/Sat)

```bash
python main.py --mode scrape
python main.py --mode embed --chunk 0 --total-chunks 30
python main.py --mode full
```
