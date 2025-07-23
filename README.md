# finding-a-gold-standard
This Repo is for the codebase adjacent to the Bachelor Thesis from me - Jesús Nieto Reyes. The 
Thesis revolves around the question of how to improve the detection and analysis of Burst 
Suppression Patterns. The investigation here will focus on the connection between unusually 
high BIS Values, which are sometimes associated with Burst Suppression Patterns.

## Roadmap
### Dataset
Dataset from [VitalDB](https://vitaldb.net/dataset/)
- [x] Skript for Downloading Data
- [x] Load Skript into this Repo

### Misc
- [x] Use consistently english in code and comments; change existing german code
- [x] Make Code readable
- [x] revise Codebase for code with no usage

# Python Section
**TO-DO**
- [ ] Find solution for organization of filtered and raw EEG PSDs
- [x] Plot it to find filtering sweet spot
  - [ ] (Optional) Implement Method to get filtering Parameters based on Ringing and Attenuation slope
- [ ] Implement the Rest of the Features
  - [x] Add missing docstrings
  - [x] Make helper functions for code block, that loops through epochs in feature_extractor functions
- [x] Implement Epoching of awake episodes
- [x] Implement train and test set functions
- [x] Implement SVM
- [ ] Implement modular Workflow
  - [x] Implement Pipeline subworkflows
  - [x] Implement coordinating method for workflows
- [ ] Implement Run-metadata functionalities
  - [x] Create Container class for collection and management of all relevant metadata
  - [ ] Create Reader functionalities to retrieve and compare information in run metadata
  - [x] Saving Functions
    - [x] Metadata file
    - [x] Save all Data from combined feature files downstream in unique run folders
      - [x] Splits
      - [x] Results
      - [x] Metrics
      - [x] Analysis
  - [ ] Check for unique Name when created (with automatic uniquifying with appended timestamp)
  - [x] Collect data from every step
    - [x] Filtering
    - [x] Transforms
    - [x] Feature extraction and selection
    - [x] Classification
      - [x] Classification data
      - [x] Model Data
    - [x] Metrics
      - [x] Metrics

# Matlab Section
**TO-DO**
Workflow
- [x] Build Utils for merging tables

## BIS_bumpSearch
This is a class, that provides functionalities around searching for Episodes, where the BIS is 
unusually high, with high MAC Values present. These episodes show a contradiction, since high
BIS values are typical for being awake but high MAC values are typical for being anesthesized.

## CSVMerger
This provides functionalities to merge CSVs retrieved from VitalDB

# Data and Folder Name Conventions
+ **Resultfolders with Name 'result_A_B_C_D':**
    These are folders for results of the Episode search with a combination of Parameters 
    represented in the name. Parameters are:
    1. A: BIS Threshold; default is 70
    2. B: MAC Threshold; current range is \[0.5,0.6,0.7,0.8\] represented without the dot. 
        0.5 -> 050, 0.6 -> 060 etc.
    3. C: Minimum Episode windowlength; current range is \[5,6,7,8,9,10,15,20\]
    4. D: Minimum refractory time between Episodes; current range is \[3,4,5\]
+ **CSV files with name 'Summary_Episodes_X_Y':**
    These are tables calculated from Summary files in above mentioned result folders in that
    way that fixed windowsizes of a specific length and a percentage of overlap is used to 
    extract all episodes with these parameters. The parameters in the name are:
    1. X: fixed windowsize; current range is \[5,6,7,8,9,10,15,20\]
    2. Y: overlap as fraction; current range is \[0,0.25,0.5\]
    
