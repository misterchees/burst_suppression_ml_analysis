psd_file = "D:\Daten\Features\PSD\Awake_20\PSD_0_20_91.csv";

T = readtable(psd_file, 'VariableNamingRule','preserve'); 
disp(T.Properties.VariableNames)