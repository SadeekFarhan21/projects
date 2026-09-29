# Airport Network Analysis Project

## Data Files

This repository contains monthly flight list data from March 2020 to October 2021.

### File Information
- **Total files**: 20 valid flight data files
- **Date range**: March 2020 - October 2021
- **Compressed size**: ~3.2GB (.csv.gz files)
- **Uncompressed size**: ~8.4GB (.csv files)
- **File format**: CSV files with flight tracking data

### Data Columns
Each CSV file contains the following columns:
- `callsign`, `number`, `aircraft_uid`, `typecode`
- `origin`, `destination`
- `firstseen`, `lastseen`, `day`
- Location data: `latitude_1`, `longitude_1`, `altitude_1`, `latitude_2`, `longitude_2`, `altitude_2`

### Git LFS Configuration

All large data files are tracked using Git LFS to ensure efficient version control:
- `*.csv` - Uncompressed flight data files (~8.4GB total)
- `*.csv.gz` - Compressed flight data files (~3.2GB total)
- `*.zip` - Archive files

### Notes
- One file (`flightlist_20211101_20211130.csv.gz`) was corrupted during extraction and is excluded from the repository
- The original zip archive (`5815448.zip`) is also tracked with Git LFS

## Getting Started

If you clone this repository, make sure to have Git LFS installed:

```bash
brew install git-lfs
git lfs install
git lfs pull
```
