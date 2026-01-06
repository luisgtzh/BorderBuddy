# BorderBuddy Services

BorderBuddy ingests CBP RSS feeds, normalizes lane-level border crossing data, and updates Supabase tables that power downstream analytics and applications. This repository contains the production ETL service and supporting assets.

## How it works
- Fetch the CBP RSS feed (`RSS_FEED_URL`).
- Flatten the XML into tabular data (automation type, lane counts, delays, statuses, timestamps).
- Normalize time fields and derive Mountain Time for reporting.
- Upsert fresh metrics into Supabase (`cruces`) and, when `ENV=prod`, bulk insert snapshots into `cruces_fronterizos`.
- Archive raw RSS pulls in `xml_files/` (optional if you enable it).

## Services
- `services/cbp_service.py`: Core ETL logic (fetch, parse, normalize, database writes). Safe to import; only runs when called explicitly.
- `services/predictions_service.py`: Entry point that invokes `cbp_service.run()`. Use this for local runs and in automation.
- `deprecated/`: Historical scripts kept for reference only.

## Requirements
- Python 3.9+ (matches GitHub Actions runner)
- Dependencies: `pip install -r requirements.txt`

## Configuration
Provide these environment variables (e.g., via `.env`, shell exports, or CI secrets):
- `RSS_FEED_URL` – CBP RSS endpoint
- `ENV` – `prod` to insert into `cruces_fronterizos`; any other value skips that insert
- `SUPABASE_USER`, `SUPABASE_PASSWORD`, `SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_DBNAME` – database connection

## Running locally
```bash
python services/predictions_service.py
```
Logs print a preview of the processed dataframe when `ENV` is not `prod`.

## Automation
GitHub Actions workflow `.github/workflows/rss_pull_push_hourly_job.yml` runs hourly and on manual dispatch:
- Installs dependencies.
- Exposes secrets for the RSS feed and Supabase.
- Executes `python services/predictions_service.py`.

## Project layout (high level)
- `services/` – production ETL code (`cbp_service.py`, `predictions_service.py`)
- `xml_files/` – optional archived RSS pulls
- `analysis/` – exploratory notebooks/visuals
- `deprecated/` – legacy scripts for reference
- `requirements.txt` – Python dependencies
