"""Script to download and verify IEEE-CIS Fraud Detection dataset files.

Usage:
    python scripts/download_data.py

Requires either:
1. `~/.kaggle/kaggle.json` configured with your Kaggle API key, OR
2. `KAGGLE_USERNAME` and `KAGGLE_KEY` set as environment variables, OR
3. Manual placement of `train_transaction.csv` and `train_identity.csv` into `data/raw/`.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR, TRAIN_TRANSACTION_FILE, TRAIN_IDENTITY_FILE

COMPETITION_NAME = "ieee-fraud-detection"
REQUIRED_FILES = [TRAIN_TRANSACTION_FILE, TRAIN_IDENTITY_FILE]


def check_existing_data() -> bool:
    """Check if all required raw CSV files already exist."""
    missing = [f for f in REQUIRED_FILES if not (RAW_DATA_DIR / f).is_file()]
    if not missing:
        print(f"All required raw dataset files are already present in {RAW_DATA_DIR}:")
        for f in REQUIRED_FILES:
            size_mb = (RAW_DATA_DIR / f).stat().st_size / (1024 * 1024)
            print(f"  - {f}: {size_mb:.1f} MB")
        return True
    return False


def download_with_kaggle_api() -> bool:
    """Attempt downloading required files using the Kaggle Python API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("Successfully authenticated with Kaggle API.")

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        for filename in REQUIRED_FILES:
            target_path = RAW_DATA_DIR / filename
            if target_path.is_file():
                print(f"{filename} already exists. Skipping download.")
                continue

            print(f"Downloading {filename} from Kaggle competition '{COMPETITION_NAME}'...")
            # Kaggle API downloads file and may save it with or without .zip extension
            api.competition_download_file(
                competition=COMPETITION_NAME,
                file_name=filename,
                path=str(RAW_DATA_DIR),
                force=False,
                quiet=False,
            )

            zip_path = RAW_DATA_DIR / f"{filename}.zip"
            if zip_path.is_file():
                print(f"Extracting {zip_path.name}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(RAW_DATA_DIR)
                zip_path.unlink()
                print(f"Successfully extracted {filename}.")

        return check_existing_data()
    except Exception as e:
        print(f"\nKaggle API download failed or credentials not found: {e}")
        return False


def print_manual_instructions():
    """Print step-by-step instructions for obtaining the dataset."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Visit: https://www.kaggle.com/c/ieee-fraud-detection/data")
    print("2. Accept competition rules if you haven't already.")
    print("3. Download these two files:")
    print("   - train_transaction.csv.zip (unzip to get train_transaction.csv)")
    print("   - train_identity.csv.zip (unzip to get train_identity.csv)")
    print(f"4. Move the extracted CSV files into: {RAW_DATA_DIR.resolve()}/")
    print("\nOr configure your Kaggle API key:")
    print("1. Go to https://www.kaggle.com/settings -> 'Create New Token'")
    print("2. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("3. chmod 600 ~/.kaggle/kaggle.json")
    print("4. Re-run this script: python scripts/download_data.py")
    print("=" * 70 + "\n")


def main():
    print(f"Checking dataset in {RAW_DATA_DIR}...")
    if check_existing_data():
        return

    print("Data not found locally. Attempting automatic download via Kaggle API...")
    success = download_with_kaggle_api()
    if not success:
        print_manual_instructions()


if __name__ == "__main__":
    main()
