# AI-Powered Cross-Layer IDS — Dataset Download Guide

## Supported Datasets

### 1. CICIDS2017 (Recommended)
- URL: https://www.unb.ca/cic/datasets/ids-2017.html
- Download the CSV files (Friday-WorkingHours-*.csv)
- Place in: `datasets/cicids2017/`

### 2. NSL-KDD
- URL: https://www.unb.ca/cic/datasets/nsl.html
- Files: KDDTrain+.txt, KDDTest+.txt
- Place in: `datasets/nslkdd/`

### 3. UNSW-NB15
- URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Files: UNSW_NB15_training-set.csv
- Place in: `datasets/unsw_nb15/`

## No Dataset? Use Demo Mode
Run without any dataset:
```
python demo_mode.py --seed 300
streamlit run dashboard.py
```
