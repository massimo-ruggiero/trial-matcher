import zipfile
from pathlib import Path

import requests
import typer
from tqdm import tqdm

from src.config import DATA_RAW

app = typer.Typer()

NIST = "https://trec.nist.gov/data/trials"
CDS = "https://www.trec-cds.org/2021_data"
TARGET = DATA_RAW / "trec-ct"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def get(url: str) -> Path:
    """Download `url` into TARGET with a progress bar, skipping existing files."""
    dest = TARGET / url.rsplit("/", 1)[-1]
    if dest.exists():
        print(f"skip {dest.name}")
        return dest

    with requests.get(url, stream=True, timeout=120, headers=HEADERS) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))

    return dest


@app.command()
def main(corpus: bool = False) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)

    for year in (2021, 2022):
        get(f"{NIST}/topics{year}.xml")
        get(f"{NIST}/qrels{year}.txt")

    if not corpus:
        return

    for i in range(1, 6):
        archive = get(f"{CDS}/ClinicalTrials.2021-04-27.part{i}.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(TARGET / "corpus")
        print(f"extracted {archive.name}")


if __name__ == "__main__":
    app()
