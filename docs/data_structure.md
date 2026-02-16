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