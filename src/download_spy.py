import argparse
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://raw.githubusercontent.com/anahatsingh-ui/options-dataset-hist/main/spy"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "VolatilitySurfaceIntelligenceEngine/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total:.0%}", end="", flush=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Download free historical SPY option Parquet data.")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to download, e.g. 2021 2022 2023 2024 2025")
    parser.add_argument("--output", default="data/raw/spy", help="Output directory")
    parser.add_argument("--underlying-only", action="store_true", help="Only download underlying_prices.parquet")
    args = parser.parse_args()

    output = Path(args.output)
    if not args.underlying_only:
        for year in sorted(set(args.years)):
            download(
                f"{BASE}/options_{year}.parquet",
                output / f"options_{year}.parquet",
            )

    download(
        f"{BASE}/underlying_prices.parquet",
        output / "underlying_prices.parquet",
    )

    print(f"\nData saved under {output.resolve()}")


if __name__ == "__main__":
    main()
