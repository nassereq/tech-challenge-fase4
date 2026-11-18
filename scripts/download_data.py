"""Baixa o Brazilian E-Commerce Public Dataset by Olist (Kaggle) para data/raw.

Uso:  python scripts/download_data.py

Alternativa manual: baixe o .zip em
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data
e extraia os CSVs em data/raw/.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, KAGGLE_DATASET  # noqa: E402


def main() -> None:
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for csv in src.glob("*.csv"):
        shutil.copy2(csv, DATA_RAW / csv.name)
        print(f"  {csv.name}")
    print(f"CSVs copiados para {DATA_RAW}")


if __name__ == "__main__":
    main()
