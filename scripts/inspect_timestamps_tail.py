from pathlib import Path
import pandas as pd
import sys

p = Path(__file__).resolve().parents[1] / 'dataset' / 'binance_ohlcv.csv'
print('CSV path:', p)

# read last N rows similarly to _read_last_rows
N = 10000
chunksize = max(N, 200_000)
reader = pd.read_csv(p, chunksize=chunksize, low_memory=False)
recent = []
rows = 0
columns = None
for chunk in reader:
    if columns is None:
        columns = chunk.columns
    recent.append(chunk)
    rows += len(chunk)
    while recent and rows - len(recent[0]) >= N:
        rows -= len(recent[0])
        recent.pop(0)

if not recent:
    print('No recent')
    sys.exit(0)
combined = pd.concat(recent, ignore_index=True)
if N < len(combined):
    combined = combined.tail(N)

print('Loaded rows:', len(combined))
print('Columns:', combined.columns.tolist())
print('Sample timestamp values:')
print(combined['timestamp'].head(20).to_list())

# detect numeric
numeric = pd.to_numeric(combined['timestamp'], errors='coerce')
print('Numeric sample non-nulls:', numeric.dropna().shape[0])

# try detect unit like module
median = numeric.dropna().median() if numeric.dropna().size>0 else None
print('Median numeric sample:', median)

# try parsing with auto and with ms
auto_parsed = pd.to_datetime(combined['timestamp'].astype(str), errors='coerce')
ms_parsed = pd.to_datetime(combined['timestamp'], unit='ms', errors='coerce')
print('Auto parsed non-nulls:', auto_parsed.notna().sum())
print('MS parsed non-nulls:', ms_parsed.notna().sum())

# show first 10 auto_parsed
print('Auto parse first 10:', auto_parsed.head(10).to_list())
print('MS parse first 10:', ms_parsed.head(10).to_list())

# write small sample
out = p.parent / 'inspect_tail_sample.csv'
combined.head(50).to_csv(out, index=False)
print('Wrote sample to', out)
