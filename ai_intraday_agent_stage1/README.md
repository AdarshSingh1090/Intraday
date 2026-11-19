# AI Intraday Agent — Stage 1: Market Data Infrastructure

This is a new project, separate from the V8 paper-trading agent. It is data collection/validation only.
No broker order endpoints, no live trading.

## Included
- Provider interface + Upstox V3 adapter
- CSV provider for local historical files
- Demo mode (synthetic data, clearly labelled)
- OHLCV validation and freshness checks
- SQLite candle store with deduplication
- CLI diagnostics and data exports

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Demo smoke test (no API credentials)
```powershell
python main.py demo --symbol DEMO
python main.py status
python main.py export --symbol DEMO
```
Synthetic demo candles are not market data and must not be used for strategy testing.

## Upstox setup
1. Create an Upstox Developer App and generate a market-data access token.
2. Put it in `.env` as `UPSTOX_ACCESS_TOKEN=...` (do not paste it into chat or commit it).
3. Confirm the instrument keys in `config.yaml` against the latest Upstox instrument master.
4. Run:
```powershell
python main.py doctor
python main.py fetch-intraday --symbol RELIANCE --interval 1
python main.py fetch-intraday --symbol RELIANCE --interval 5
python main.py status
python main.py export --symbol RELIANCE
```
The Upstox V3 intraday endpoint fetches the current trading day's candles. Access token availability/expiry and API limits are provider-controlled. If authentication fails, the tool reports the response but never prints the token.

## Historical request
```powershell
python main.py fetch-history --symbol RELIANCE --interval 1 --from-date 2026-11-01 --to-date 2026-11-20
```
V3 minute-history request limits apply; this starter fetches one requested date range and does not automatically paginate/chunk.

## CSV import
Expected columns (case-insensitive aliases accepted):
`timestamp, open, high, low, close, volume`
```powershell
python main.py import-csv --symbol RELIANCE --file .\my_candles.csv
```

## Commands
- `python main.py doctor`
- `python main.py demo --symbol DEMO`
- `python main.py fetch-intraday --symbol RELIANCE --interval 1`
- `python main.py fetch-history --symbol RELIANCE --interval 1 --from-date YYYY-MM-DD --to-date YYYY-MM-DD`
- `python main.py import-csv --symbol SYMBOL --file path.csv`
- `python main.py status`
- `python main.py export --symbol SYMBOL`

## Data quality rules
- Timestamps parsed and normalized to Asia/Kolkata
- Required OHLCV fields
- Positive OHLC values
- Non-negative volume
- High >= max(open, close, low), low <= min(open, close, high)
- Duplicate timestamp detection
- Provider/source and ingestion timestamp retained
- Data quality report retained for rejected rows

## Database
`database/market_data.sqlite` stores candles and quality reports. CSV exports go to `data/processed/`.

## Safety
The access token is a credential. Do not send it to anyone or include it in screenshots. This project only requests market data and does not contain order-placement code.
