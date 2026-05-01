# Data Acquisition
## Dataset 
+ [VitalDB](https://vitaldb.net/dataset/) containing synchronized biosignals and clinical data from 6388 surgical patients from 
Seoul National University Hospital [(Lee et al., 2022)](https://www.nature.com/articles/s41597-022-01411-5)

### Tracks of Interest
+ **BIS/BIS**: Bispectral Index [(BIS)](https://en.wikipedia.org/wiki/Bispectral_index) 
value sampled at 1 Hz
+ **BIS/SR**: Suppression ratio of the EEG calculated by BIS-Monitor sampled at 1 Hz
+ **Primus/MAC**: Minimum alveolar concentration [(MAC)](https://en.wikipedia.org/wiki/Minimum_alveolar_concentration) 
sampled once every 7 seconds ~ 0.14 Hz
+ **EEG1/EEG2**: 2-Channel-EEG track sampled at 128 Hz

*Not all patients have these tracks present (especially MAC is only available for inhalational anesthesia) leading to
a subset of 2849 patients*

## Merging and cleanup of downloaded tracks 
## BIS_bumpSearch
This is a class, that provides functionalities around searching for Episodes, where the BIS is 
unusually high, with high MAC Values present. These episodes show a contradiction, since high
BIS values are typical for being awake but high MAC values are typical for being anesthesized.

# Data Preprocessing
## Loading into the Pipeline
**Awake Episodes**: The main source of the Awake Episodes comes from the anestart annotations from the metadata which are
aggregated in _anestart_analysis_results.csv_. 
After inspecting the tracks visually, it turned out the annotations are not always correct, so another file was created 
named _awake_cleaned.txt_. That's the reason for the _awake_cleaned_ flag in the _load_awake_times_as_df_ function.

# Feature Extraction

# Split Creation

# Classification

# Post-hoc Analyses