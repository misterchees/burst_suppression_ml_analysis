# finding-a-gold-standard
This Repo is for the codebase adjacent to the Bachelor Thesis from me (Jesús Nieto Reyes). The 
Thesis revolves around the question of how to improve the detction and analysis of Burst 
Suppression Patterns. The investigation here will focus on the connection between unusually 
high BIS Values, which are sometimes associated with Burst Suppression Patterns.

## Roadmap
### Dataset
Dataset from [VitalDB](https://vitaldb.net/dataset/)
-[x] Skript for Downloading Data
-[ ] Load Skript into this Repo

### Workflow
-[x] Build Utils for merging tables
-[ ] Build Functions for 

### Misc
-[x] Use consistently english in code and comments; change existing german code
-[x] Make Code readable
-[x] revise Codebase for code with no usage
-[ ] search and follow conventions


## BIS_bumpSearch
This is a class, that provides functionalities around searching for Episodes, where the BIS is 
unusually high, with high MAC Values present. These episodes show a contradiction, since high
BIS values are typical for being awake but high MAC values are typical for being anesthesized.

## CSVMerger
This provides functionalities to merge CSVs retrieved from VitalDB


