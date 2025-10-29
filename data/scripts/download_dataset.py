"""
Download the MyData mycetoma dataset from Zenodo.

The MyData dataset contains 864 histopathology images from 142 patients
collected at the Mycetoma Research Centre, University of Khartoum, Sudan.

Zenodo DOI: 10.5281/zenodo.13655082
ArXiv Paper: https://arxiv.org/abs/2410.12833

Note: This dataset requires request access from Zenodo.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import argparse


# MyData Zenodo Record
ZENODO_RECORD_ID = "13655082"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def download_file(url: str, destination: Path, filename: str):
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Destination directory
        filename: Name of file to save
    """
    destination.mkdir(parents=True, exist_ok=True)
    filepath = destination / filename
    
    # Check if file already exists
    if filepath.exists():
        print(f"✓ {filename} already exists, skipping...")
        return filepath
    
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✓ Downloaded {filename}")
    return filepath


def get_zenodo_files():
    """
    Get list of files from Zenodo record.
    
    Returns:
        List of file metadata dictionaries
    """
    response = requests.get(ZENODO_API_URL)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Zenodo record: {response.status_code}")
    
    data = response.json()
    files = data.get('files', [])
    
    return files


def download_mydata_dataset(output_dir: str = "data/"):
    """
    Download the complete MyData dataset from Zenodo.
    
    Args:
        output_dir: Directory to save downloaded files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MYDATA MYCETOMA DATASET DOWNLOADER")
    print("="*60)
    print(f"Zenodo Record: {ZENODO_RECORD_ID}")
    print(f"Output Directory: {output_path.absolute()}")
    print()
    
    # Get file list
    print("Fetching file list from Zenodo...")
    try:
        files = get_zenodo_files()
        print(f"✓ Found {len(files)} files")
    except Exception as e:
        print(f"✗ Error fetching files: {e}")
        print()
        print("NOTE: The MyData dataset may require access request.")
        print("Please visit: https://zenodo.org/records/13655082")
        print("And request access from the authors.")
        return
    
    # Download each file
    print()
    print("Downloading files...")
    print("-"*60)
    
    for file_info in files:
        filename = file_info['key']
        file_url = file_info['links']['self']
        size_mb = file_info['size'] / (1024 * 1024)
        
        print(f"\n{filename} ({size_mb:.1f} MB)")
        download_file(file_url, output_path, filename)
    
    print()
    print("="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Files saved to: {output_path.absolute()}")
    
    # Extract if zip files
    print("\nExtracting archives...")
    for file_path in output_path.glob("*.zip"):
        print(f"Extracting {file_path.name}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"✓ Extracted {file_path.name}")
    
    print("\n✓ Dataset ready for use!")


def download_sample_data(output_dir: str = "data/sample/"):
    """
    Download a small sample of the dataset for testing.
    
    Args:
        output_dir: Directory to save sample data
    """
    print("NOTE: Sample data download not yet implemented.")
    print("The full MyData dataset must be requested from Zenodo.")
    print()
    print("To request access:")
    print("1. Visit: https://zenodo.org/records/13655082")
    print("2. Click 'Request Access'")
    print("3. Provide justification for research use")
    print("4. Wait for approval from dataset authors")


def print_dataset_info():
    """Print information about the MyData dataset."""
    print("="*60)
    print("MYDATA MYCETOMA DATASET INFORMATION")
    print("="*60)
    print()
    print("Dataset: MyData - Mycetoma Tissue Microscopic Images")
    print("Source: Mycetoma Research Centre, University of Khartoum, Sudan")
    print("Published: October 2024")
    print()
    print("Content:")
    print("  - 864 histopathology images (H&E stained)")
    print("  - 142 patients")
    print("  - Binary classification: Eumycetoma vs Actinomycetoma")
    print("  - Annotations with grain segmentation masks")
    print()
    print("Causative Organisms Included:")
    print("  - Madurella mycetomatis (most common)")
    print("  - Streptomyces somaliensis")
    print("  - Actinomadura madurae")
    print("  - Actinomadura pelletieri")
    print("  - Nocardia species")
    print()
    print("Access:")
    print("  - License: CC BY (Creative Commons Attribution)")
    print("  - Access: Restricted - requires formal request")
    print("  - Zenodo: https://zenodo.org/records/13655082")
    print("  - Paper: https://arxiv.org/abs/2410.12833")
    print()
    print("Citation:")
    print("  Omar Ali, H., Abraham, R., Desoubeaux, G., Fahal, A., & Tauber, C. (2024).")
    print("  MyData: A Comprehensive Database of Mycetoma Tissue Microscopic Images")
    print("  for Histopathological Analysis. arXiv:2410.12833")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download MyData mycetoma dataset from Zenodo"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download only sample data for testing"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        return
    
    if args.sample:
        download_sample_data(args.output)
    else:
        download_mydata_dataset(args.output)


if __name__ == "__main__":
    main()
