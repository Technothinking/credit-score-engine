import os, urllib.request, zipfile, hashlib

RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

datasets = {
    "UCI Default (Taiwan credit)": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "dest": f"{RAW}/uci_credit.zip"
    }
}

for name, info in datasets.items():
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(info["url"], info["dest"])
    with zipfile.ZipFile(info["dest"], 'r') as z:
        z.extractall(RAW)
    print(f"  Saved to {info['dest']}")

print("\nDone. Check data/raw/ for extracted files.")