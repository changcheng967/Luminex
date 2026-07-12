#!/usr/bin/env python3
"""C500: Decompress sf_dataset_big.txt.gz -> sf_dataset_big.txt
Finds the .gz in the c2net dataset path."""
import os, glob, gzip, shutil
from c2net.context import prepare
ctx = prepare()
ds = ctx.dataset_path

# Find the compressed file
gz = None
for name in ["sf_dataset_big.txt.gz", "sf_dataset.txt.gz"]:
    matches = glob.glob(os.path.join(ds, "**", name), recursive=True)
    if matches:
        gz = matches[0]
        break
if not gz:
    # Try uncompressed
    for name in ["sf_dataset_big.txt", "sf_dataset.txt"]:
        matches = glob.glob(os.path.join(ds, "**", name), recursive=True)
        if matches:
            print(f"Already decompressed: {matches[0]}", flush=True)
            exit(0)
    raise FileNotFoundError("sf_dataset_big.txt(.gz) not found in dataset")

out = "/tmp/code/sf_dataset_big.txt"
os.makedirs("/tmp/code", exist_ok=True)
print(f"Decompressing {gz} -> {out}...", flush=True)
with gzip.open(gz, 'rb') as f_in, open(out, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
print(f"Done: {os.path.getsize(out)/1e9:.1f} GB", flush=True)
