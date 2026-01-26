import os
import requests
import time

"""
Data Acquisition Script for Galaxy Rotation Curves (SPARC)
==========================================================
This script downloads rotation curve data (.dat files) from the SPARC database mirrors.
It automatically handles directory creation and manages connection timeouts.

Usage:
    Run this script within the 'data' directory.
    Output structure:
        ./galaxies/12_galaxies/ (Target subset)
        ./galaxies/65_galaxies/ (Full dataset)
"""

# ==============================================================================
# [1] Configuration: Directory Setup (Relative Paths)
# ==============================================================================
# Determine the directory where this script is located (e.g., .../data/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define base output directory relative to the script location
BASE_PATH = os.path.join(CURRENT_DIR, "galaxies")

# Define subdirectories
DIR_65 = os.path.join(BASE_PATH, "65_galaxies")
DIR_12 = os.path.join(BASE_PATH, "12_galaxies")

# Create directories if they do not exist
os.makedirs(DIR_65, exist_ok=True)
os.makedirs(DIR_12, exist_ok=True)

print(f"[Info] Current Script Directory: {CURRENT_DIR}")
print(f"[Info] Target Directory (65 set): {DIR_65}")
print(f"[Info] Target Directory (12 set): {DIR_12}")

# ==============================================================================
# [2] Target Galaxy Lists
# ==============================================================================
# Subset for primary analysis
SAMPLE_12_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", 
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885"
]

# Full dataset for statistical validation
SAMPLE_65_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", 
    "NGC3521", "NGC5055", "NGC7331", "NGC6946", "NGC7793", 
    "NGC1560", "UGC02885", "NGC0801", "NGC2998", "NGC5033", 
    "NGC5533", "NGC5907", "NGC6674", "UGC06614", "UGC06786", 
    "F568-3", "F571-8", "NGC0055", "NGC0247", "NGC0300", 
    "NGC1003", "NGC1365", "NGC2541", "NGC2683", "NGC2915", 
    "NGC3109", "NGC3621", "NGC3726", "NGC3741", "NGC3769", 
    "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC3953", 
    "NGC3972", "NGC3992", "NGC4013", "NGC4051", "NGC4085", 
    "NGC4088", "NGC4100", "NGC4138", "NGC4157", "NGC4183", 
    "NGC4217", "NGC4559", "NGC5585", "NGC5985", "NGC6015", 
    "NGC6195", "UGC06399", "UGC06446", "UGC06667", "UGC06818", 
    "UGC06917", "UGC06923", "UGC06930", "UGC06983", "UGC07089"
]

# ==============================================================================
# [3] Core Function: Data Retrieval
# ==============================================================================
def download_file(gal_name, target_dir):
    """
    Downloads the rotation curve data file for a given galaxy.
    Tries multiple mirrors to ensure availability.
    """
    filename = f"{gal_name}_rotmod.dat"
    save_path = os.path.join(target_dir, filename)
    
    # Check if file already exists to avoid redundant downloads
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return "Skip (File Exists)"

    # User-Agent header to mimic a standard browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # List of mirrors (GitHub Raw is prioritized for stability)
    urls = [
        f"https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/master/data/{filename}",
        f"https://raw.githubusercontent.com/carsondowns-cte/Rotmod_LTG/main/{filename}",
        f"http://astroweb.cwru.edu/SPARC/data/{filename}"
    ]

    for i, url in enumerate(urls):
        try:
            # Request with timeout settings
            r = requests.get(url, headers=headers, timeout=10)
            
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return f"Success (Source {i+1})"
            elif r.status_code == 404:
                continue # File not found on this mirror, try next
                
        except Exception as e:
            continue # Connection error, try next mirror

    return "Failed (All Mirrors Unreachable)"

# ==============================================================================
# [4] Main Execution
# ==============================================================================
def main():
    print("="*60)
    print("SPARC Data Downloader Initialized")
    print("="*60)

    # 1. Download Full Dataset (65 Galaxies)
    print(f"\n[Task 1] Processing Full Dataset (N={len(SAMPLE_65_GALAXIES)})...")
    for i, gal in enumerate(SAMPLE_65_GALAXIES):
        status = download_file(gal, DIR_65)
        print(f"   [{i+1:02d}/{len(SAMPLE_65_GALAXIES)}] {gal:<10} : {status}")
        
    # 2. Download Core Subset (12 Galaxies)
    print(f"\n[Task 2] Processing Core Subset (N={len(SAMPLE_12_GALAXIES)})...")
    for i, gal in enumerate(SAMPLE_12_GALAXIES):
        status = download_file(gal, DIR_12)
        print(f"   [{i+1:02d}/{len(SAMPLE_12_GALAXIES)}] {gal:<10} : {status}")

    print("\n" + "="*60)
    print("Download Process Completed.")
    print(f"Verify Data at: {BASE_PATH}")

if __name__ == "__main__":
    main()