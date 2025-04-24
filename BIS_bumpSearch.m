% This class has the purpose to search in CSV-Tables provided by VitalDB
% for unusual Events, that hint to Burst Suppression Pattern. It's
% specifically about Episodes when BIS Values are high (typical for the
% awake state) but the MAC Values are also high (typical when patient is anesthesized)

classdef BIS_bumpSearch < handle

    properties
        % relevant paths
        inputFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed_BIS_BIS_SR_MAC\';
        metaDataFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\';
        resultsFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\results\';
        plotsFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\plots\';
        inputTablesField = 'inputTables'; % fieldname in data structure of tables with the BIS values
        episodesTablesField = 'episodeTables'; % fieldname with tables of found episodes
        metadataField = 'metadata'; % filedname with metadata from all patients
        BIS_col_name = 'BIS_BIS'; % Column name for BIS values in BIS tables
        BIS_SR_col_name = 'BIS_SR'; % Column name for BIS Suppression Rate values 
        MAC_col_name = 'Primus_MAC'; % Column for MAC values
        BIS_threshold = 70; % Threshold minimum BIS (range 0-100)
        MAC_threshold = 0.5; % Threshold minimum MAC (no unit)
        min_BIS_episodeTimeInSeconds = 5; % Threshold minimum time (s) for episodes
        refractoryTimeInSeconds = 5; % Threshold minimum for time (s) in between episodes
        data = struct; % Datenstruktur mit allen relevanten Daten
    end

    methods
        % Constructor
        function obj = BIS_bumpSearch (inputFolderPath, metaDataFolderPath, resultsFolderPath, plotsFolderPath)
            if nargin > 0
                obj.inputFolderPath = inputFolderPath;
            elseif nargin > 1
                obj.metaDataFolderPath = metaDataFolderPath;
            elseif nargin > 2
                obj.resultsFolderPath = resultsFolderPath;
            elseif nargin > 3
                obj.plotsFolderPath = plotsFolderPath;

            end
        end

        % change folder path for specified field to Path given in inputFolderPathValue
        function obj = setNewInputFolderPath(obj, inputFolderPathField, inputFolderPathValue)
            obj.(inputFolderPathField) = inputFolderPathValue;
        end

        % Set new thresholds for BIS and MAC values
        function obj = setNewThresholds(obj, BIS_threshold, MAC_threshold)
            obj.BIS_threshold = BIS_threshold;
            obj.MAC_threshold = MAC_threshold;
        end

        % Set new episode time and new refractory time
        function obj = setNewTimes(obj, episodeTime, refractoryTime)
            obj.min_BIS_episodeTimeInSeconds = episodeTime;
            obj.refractoryTimeInSeconds = refractoryTime;
        end

        function getAwakeTime(obj)
        % This function gathers all relevant caseids in the input folder
        % and then proceeds to sum all positive numbers of anestart in
        % metadata, outputting the overall sum of the time, where patients
        % were awake and also outputs which patients contributed to the
        % sum.
            
            % Declare variables
            csvFolderPath = obj.inputFolderPath;
            metadataFolderPath = obj.metaDataFolderPath;


            % List all CSV files in the target folder
            files = dir(fullfile(csvFolderPath, '*.csv'));
            
            % Extract numbers from filenames and store in list
            caseIds = [];
            for i = 1:length(files)
                [~, name, ~] = fileparts(files(i).name);
                id = str2double(name);
                % If number -> append, if not -> warning
                if ~isnan(id)
                    caseIds(end+1) = id; 
                else
                    warning("Number in csv-file %s could not be extracted" ,files(i).name);
                end
            end
        
            % Load metadata file
            metadataFile = fullfile(metadataFolderPath, 'metadata_vitaldb.csv');
            if ~isfile(metadataFile)
                error('Metadata file not found in %s', metadataFile);
            end
        
            metadata = readtable(metadataFile);
        
            % Filter metadata rows matching the extracted case IDs
            validIndices = ismember(metadata.caseid, caseIds);
            matchedCaseIds = metadata.caseid(validIndices);
            anestartValues = metadata.anestart(validIndices);
        
            % Further filter rows where anestart > 0
            positiveMask = anestartValues > 0;
            filteredCaseIds = matchedCaseIds(positiveMask);
            filteredAnestartValues = anestartValues(positiveMask);
            totalSum = sum(filteredAnestartValues);
        
            % Display results in the console
            fprintf('Found case IDs: %s\n', mat2str(filteredCaseIds'));
            fprintf('Sum of awake times: %.2f\n', totalSum);
        
            % Save results to a CSV file
            resultTable = table(filteredCaseIds, filteredAnestartValues, ...
                'VariableNames', {'caseid', 'anestart'});
            resultFile = fullfile(metadataFolderPath, 'anestart_analysis_results.csv');
            writetable(resultTable, resultFile);
        end

        function obj = detectEpisodes(obj, tableName)
        % Searches for Episodes where BIS values and MAC values
        % simultaniously exceed certain thresholds over a minimum duration
        % and flagging episodes if duration in between is below a
        % refractory time. Thresholds and durations are class properties
        % and can be changed through setters
            
            % Set function variables
            inputTables = obj.inputTablesField;
            inputTable = obj.data.(inputTables).(tableName);
            time = inputTable.Time;
            colBIS = inputTable.(obj.BIS_col_name);
            colMAC = inputTable.(obj.MAC_col_name);
            minDuration = obj.min_BIS_episodeTimeInSeconds;
            minRefractoryTime = obj.refractoryTimeInSeconds;
            
            % Find first and last time in which Data is nonzero and not NaN
            validIdx = find(~isnan(colBIS) & colBIS ~= 0, 1, 'first');
            firstTime = time(validIdx);
            validIdx = find(~isnan(colBIS) & colBIS ~= 0, 1, 'last');
            lastTime = time(validIdx);
            
            % replace NaN values in MAC-Column with linear interpolation
            nanIdx = isnan(colMAC);
            validX = time(~nanIdx);
            validY = colMAC(~nanIdx);
            colMAC(nanIdx) = interp1(validX, validY, time(nanIdx), 'linear', 'extrap');
            
            % Identify possible episode beginnings
            aboveThreshold1 = colBIS > obj.BIS_threshold; % Logical vector if above BIS threshold
            % create datastructures for episodes
            episodeStart = []; 
            episodeEnd = [];
            flagged = [];
            inEpisode = false;
            startIdx = 0;

            % Factor in minDuration of an Episode in Range end
            for i = 1:length(time) - minDuration
                % Search for episode beginning
                if ~inEpisode && all(aboveThreshold1(i:i+minDuration-1))
                    if all(colMAC(i:i+minDuration-1) > obj.MAC_threshold)
                        inEpisode = true;
                        startIdx = i;
                    end
                % Search episode end
                elseif inEpisode && ~aboveThreshold1(i)
                    inEpisode = false;
                    episodeStart = [episodeStart; time(startIdx)];
                    episodeEnd = [episodeEnd; time(i-1)];
                end
            end
            
            % Check if time between episodes is below refractory time
            flagged = false(size(episodeStart));
            % Flagging beginning of one episode and end of the other so
            % both episodes are flagged
            for j = 2:length(episodeStart)
                if (episodeStart(j) - episodeEnd(j-1)) < minRefractoryTime
                    flagged(j-1) = true;
                    flagged(j) = true;
                end
            end
            
            % make valid field name, deleting point of floating point number
            MAC_threshold_str = strrep(sprintf('%.2f', obj.MAC_threshold), '.', '');
            resultName = sprintf('result_%d_%s_%d_%d', obj.BIS_threshold, MAC_threshold_str, minDuration, minRefractoryTime);
            resultName = matlab.lang.makeValidName(resultName);

            % Save table and valid beginning and end 
            episodeTable = table(episodeStart, episodeEnd, flagged, 'VariableNames', {'Start', 'End', 'Flagged'});
            obj.data.(resultName).(['result_' extractAfter(tableName, 'BIS_ID_')])... 
                = struct('Episodes', episodeTable, 'FirstValidTime', firstTime, 'LastValidTime', lastTime);
        end
       
        % Detect Episodes for a range of tables
        function obj = detectEpisodesInRange(obj, range)
            for i = range
                fullTableName = strcat("BIS_ID_" + i);
                fullTableNameChar = convertStringsToChars(fullTableName);

                try
                    disp('Searching in table: ' + fullTableNameChar);
                    detectEpisodes(obj, fullTableNameChar);
                catch error
                    errorMessage = strcat('Error found in detectEpisodes for file ', fullTableNameChar, ...
                    ' error message: ', error.message);
                    warning(errorMessage);
                end
            end
        end

        % Detect Episodes in all tables of given tables field
        function obj = detectEpisodesInAllTables(obj)
            
            tables = fieldnames(obj.data.(obj.inputTablesField));
            for i = 1:length(tables)
                try
                    disp(strcat('Searching in table: ' , tables{i}));
                    detectEpisodes(obj, tables{i});
                catch error
                    errorMessage = strcat('Error found in detectEpisodes for file ', tables{i}, ...
                    ' error message: ', error.message);
                    warning(errorMessage);
                end
            end
        end

        % Generate a summary table with all tables from given resultsField
        % and a table that summarizes all first and last values per result
        function obj = generateSummaryTables(obj, resultsField)
            % initialize summary array
            episodeList = [];
            timeList = [];
            
            fields = fieldnames(obj.data.(resultsField));
            for i = 1:length(fields)
                if startsWith(fields{i}, 'result_') % Only evaluate fields starting with 'result_'
                    resultNum = str2double(extractAfter(fields{i}, 'result_')); % extract patient ID
                    resultStruct = obj.data.(resultsField).(fields{i}); % save result field in variable
                    
                    % If episodes are present, add to summary array and save
                    % first and last value of the table in timeList
                    if ~isempty(resultStruct.Episodes)
                        tempTable = resultStruct.Episodes;
                        tempTable.ResultID = repmat(resultNum, height(tempTable), 1); % Patientennummer hinzufügen
                        episodeList = [episodeList; tempTable]; % An Summary-Array anhängen
                        
                        % Anfangs- und Endzeiten der Aufzeichnung anhängen
                        timeList = [timeList; table(resultNum, resultStruct.FirstValidTime, resultStruct.LastValidTime, ...
                                    'VariableNames', {'ResultID', 'FirstValidTime', 'LastValidTime'})];
                    end
                end
            end
            
            % Save Summaries in Summary_Episodes und Summary_GlobalTimes
            obj.data.(resultsField).Summary_Episodes = episodeList;
            obj.data.(resultsField).Summary_GlobalTimes = timeList;
        end

        % Generate summary tables for every resultField in obj.data
        function obj = generateSumaryTablesForAll(obj)
            fields = fieldnames(obj.data);
            for i = 1:length(fields)
                if startsWith(fields{i}, 'result_') % Only evaluate fields starting with 'result_'
                    generateSummaryTables(obj, fields{i});
                end
            end
        end

        % Search in given folder for subfolders with Episode_Summary and
        % create new table with name Summary_Merged_Episodes.csv where all
        % flagged episodes were merged together depending on refractory
        % time
        function mergeFlaggedEpisodes(obj)
            % save resultsFolderPath in variable
            resultsFolder = obj.resultsFolderPath;

            % Search for subfolder in 'results'-directory
            subFolders = dir(resultsFolder);
            % filter all nonFolders out (hidden Folders, system folders etc.)
            subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));
            
            for i = 1:length(subFolders)
                % create path to CSV in subfolder
                curFolderPath = fullfile(resultsFolder, subFolders(i).name);
                summaryFile = fullfile(curFolderPath, 'Summary_Episodes.csv');
                
                % Errorhandling for not existing file
                if ~exist(summaryFile, 'file')
                    error('%s does not exist.',summaryFile);
                end

                % log
                disp(strcat('Merging flagged Episodes from ', summaryFile));

                % load CSV file
                episodes = readtable(summaryFile);
                
                % variables for merged episodes
                mergedEpisodes = [];
                currentStart = episodes.Start(1);
                currentEnd = episodes.End(1);
                currentResultID = episodes.ResultID(1);
                
                for j = 2:height(episodes)
                    episodeDistance = episodes.Start(j) - currentEnd;
                    % Merging when both episodes are flagged and patient ID
                    % is the same and time in between below refractoryTime
                    if episodes.Flagged(j-1) == 1 && episodes.Flagged(j) == 1 ...
                            && episodes.ResultID(j) == currentResultID ...
                            && episodeDistance < obj.refractoryTimeInSeconds
                        % expand current episode
                        currentEnd = episodes.End(j);
                    else
                        % save merged episode
                        mergedEpisodes = [mergedEpisodes; {currentStart, currentEnd, currentResultID}];
                        % start new episode
                        currentStart = episodes.Start(j);
                        currentEnd = episodes.End(j);
                        currentResultID = episodes.ResultID(j);
                    end
                end
                % save last episode
                mergedEpisodes = [mergedEpisodes; {currentStart, currentEnd, currentResultID}];
                
                % create new table
                mergedTable = cell2table(mergedEpisodes, 'VariableNames', {'Start', 'End', 'ResultID'});
                
                % save CSV file
                mergedFile = fullfile(curFolderPath, 'Summary_Merged_Episodes.csv');
                writetable(mergedTable, mergedFile);
                % finalization log
                disp(strcat('Merging complete. Episodes are in: ', mergedFile));
            end
        end

        function generate_windowed_episodes(obj, windowlength, overlap, mergedEpisodes, noOverwrite)
        % This function processes subfolders inside the given folderPath.
        % Each subfolder is named as 'result_A_B_C_D'.
        % It filters folders where C >= windowlength, then reads 'Summary_Episodes.csv',
        % and generates new episodes based on given windowlength and overlap.
        % The output is written to a new CSV file named 'Summary_Episodes_X_Y.csv'.
        % mergedEpisodes is a logical parameter, that decides if this looks
        % into 'Summary_Merged_Episodes.csv' or 'Summary_Episodes.csv'

            % input handling
            if windowlength < 1
                error("illegal window length:" + windowlength)
            end

            % use folderPath from class
            folderPath = obj.resultsFolderPath;
        
            % Format overlap for filename
            overlap_str = sprintf('%03d', round(overlap * 100));

            % Build output filename
            if mergedEpisodes
                outputFilename = sprintf('Summary_Merged_Episodes_%d_%s.csv', windowlength, overlap_str);
            else
                outputFilename = sprintf('Summary_Episodes_%d_%s.csv', windowlength, overlap_str);
            end
        
            % List all items in the folderPath
            folderList = dir(folderPath);
        
            for i = 1:length(folderList)
                folderName = folderList(i).name;

                % log current folder
                fprintf("Processing folder: %s \n", folderName);
        
                % Skip if it's not a folder or doesn't match the 'result_' pattern
                if ~folderList(i).isdir || ~startsWith(folderName, 'result_')
                    warning("Folder %s does not start with 'result_' and will be ignored ", folderName);
                    continue;
                end
        
                % Build the path to the episodes CSV depending on
                % mergedEpisodes Flag
                if mergedEpisodes
                    csvFile = 'Summary_Merged_Episodes.csv';
                else
                    csvFile = 'Summary_Episodes.csv';
                end
                csvPath = fullfile(folderPath, folderName, csvFile);

                % Build output path
                outputPath = fullfile(folderPath, folderName, outputFilename);

                % Skip if output File exists and noOverwrite is set 
                if isfile(outputPath) && noOverwrite
                    warning("The file in %s does exist and function is in no overwrite mode. Continuing with next file", outputPath);
                    continue;
                end

                % Check if file exists
                if ~isfile(csvPath)
                    warning("The file in %s does not exist. Continuing with next file", csvPath);
                    continue;
                end
        
                % Read the original CSV file
                csvTable = readtable(csvPath);
        
                % Check required columns exist
                if ~all(ismember({'Start', 'End', 'ResultID'}, csvTable.Properties.VariableNames))
                    warning('Missing required columns in %s', csvPath);
                    continue;
                end
        
                % Initialize output table
                newEpisodes = table();
        
                % Generate new windows for each row
                for j = 1:height(csvTable)
                    startTime = csvTable.Start(j);
                    endTime = csvTable.End(j);
                    currentLength = endTime-startTime;
                    resultID = csvTable.ResultID(j);
        
                    step = windowlength * (1 - overlap);
                    t = startTime;
        
                    % ensure window is big enough to be split
                    if currentLength >= windowlength
                        while (t + windowlength) <= endTime
                            % Round start and end to nearest whole number upwards
                            winStart = ceil(t);
                            winEnd = ceil(t + windowlength);
            
                            % Append to new table
                            newEpisodes = [newEpisodes; table(winStart, winEnd, resultID, ...
                                'VariableNames', {'Start', 'End', 'ResultID'})];
                            
                            % Increment to next start
                            t = t + step;
                        end
                    end
                end
        
                % Write the new table to CSV
                writetable(newEpisodes, outputPath);

                % output log
                fprintf("Folder: %s succesfully processed \n", folderName);
            end
        end

        function collect_episode_statistics(obj, mergedEpisodes)
            % This function scans subfolders named 'result_A_B_C_D' inside folderPath.
            % In each subfolder, it looks for files named 'Summary_Episodes_X_Y.csv'.
            % It parses A, B, C, D from folder name and X, Y from file name,
            % counts the number of episodes (rows) in each CSV file,
            % and writes a summary table to 'all_episodes_count.csv' in
            % folderPath. If mergedEpisodes is true, than it will do all
            % this but for merged episode files instead
        
            % variable for folder path
            folderPath = obj.resultsFolderPath;
            
            % log beginning of function
            if mergedEpisodes
                fprintf("Collecting merged episode statistics in folder: %s \n", folderPath);
            else
                fprintf("Collecting episode statistics in folder: %s \n", folderPath);
            end

            % Initialize empty cell array to collect data
            summaryData = {};
        
            % List all folders in the given path
            folderList = dir(folderPath);
        
            for i = 1:length(folderList)
                folderName = folderList(i).name;
        
                % Check if it's a folder and matches the expected pattern
                if ~folderList(i).isdir || ~startsWith(folderName, 'result_')
                    warning("Folder %s does not start with 'result_' and will be ignored ", folderName);
                    continue;
                end
        
                % Extract parameters A, B, C, D from folder name
                tokens = regexp(folderName, 'result_(\d+)_([\d]+)_(\d+)_([\d]+)', 'tokens');
                if isempty(tokens)
                    warning("Could not find parameters in Folder %s. Continuing with next folder ", folderName);
                    continue;
                end
        
                parts = tokens{1};
                A = str2double(parts{1});                      % BIS_thr
                % insert dot after first number
                B = str2double(insertAfter(string(parts{2}), 1, '.')); % MAC_thr
                C = str2double(parts{3});                      % min_length
                D = str2double(parts{4});                      % min_reftime
        
                % change filename and regex template depending on
                % mergedEpisodes flag
                if mergedEpisodes
                    fileName = 'Summary_Merged_Episodes_*.csv';
                    fileNameRegex = 'Summary_Merged_Episodes_(\d+)_([\d]+)\.csv';
                else
                    fileName = 'Summary_Episodes_*.csv';
                    fileNameRegex = 'Summary_Episodes_(\d+)_([\d]+)\.csv';
                end
                
                % Path to current result folder
                subfolderPath = fullfile(folderPath, folderName);
        
                % Get all Summary_Episodes_X_Y.csv files
                csvFiles = dir(fullfile(subfolderPath, fileName));
        
                for j = 1:length(csvFiles)
                    csvName = csvFiles(j).name;
        
                    % Extract X (fixed window length) and Y (overlap fraction)
                    tokensCSV = regexp(csvName, fileNameRegex, 'tokens');
                    if isempty(tokensCSV)
                        warning("Could not find parameters in CSV %s. Continuing with next csv file ", csvName);
                        continue;
                    end
        
                    partsCSV = tokensCSV{1};
                    X = str2double(partsCSV{1});               % fixed_window
                    % insert dot after first number
                    Y = str2double(insertAfter(string(partsCSV{2}), 1, '.')); % overlap_frac
        
                    % Count the number of episodes in the CSV file
                    fullCsvPath = fullfile(subfolderPath, csvName);
                    episodeTable = readtable(fullCsvPath);
                    numEpisodes = height(episodeTable);
        
                    % Append to summary data
                    summaryData(end+1, :) = {A, B, C, D, X, Y, numEpisodes};
                end
            end
        
            % Convert to table with appropriate column names
            summaryTable = cell2table(summaryData, 'VariableNames', ...
                {'BIS_thr', 'MAC_thr', 'min_length', 'min_reftime', ...
                 'fixed_window', 'overlap_frac', 'num_episodes'});
        
            if mergedEpisodes
                % Write to CSV and log
                writetable(summaryTable, fullfile(folderPath, 'all_merged_episodes_count.csv'));
                fprintf("Finished calculating merged episode statistics. Saving to folder: %s \n", folderPath);
            else
                % Write to CSV and log
                writetable(summaryTable, fullfile(folderPath, 'all_episodes_count.csv'));
                fprintf("Finished calculating episode statistics. Saving to folder: %s \n", folderPath);
            end
        end
        
        function generate_overlap_summary_tables(obj, mergedEpisodes)
            % This function reads 'all_episodes_count.csv' in the given folder.
            % It groups the data by BIS_thr, MAC_thr, min_length, min_reftime
            % and creates one summary table per unique combination.
            % Each table has columns: window, overlap_000, overlap_025, overlap_050, etc.
            % Output tables are written to a subfolder called 'window vs overlap'.
        
            % variable for folder path
            folderPath = obj.resultsFolderPath;

            if mergedEpisodes
                inputCSV = 'all_merged_episodes_count.csv';
            else
                inputCSV = 'all_episodes_count.csv';
            end

            % log
            fprintf("Creating pivot tables for %s in folder: %s \n", inputCSV ,folderPath);

            % Path to the input CSV
            inputFile = fullfile(folderPath, inputCSV);
        
            % Read the main summary file
            if ~isfile(inputFile)
                error('File "%s" not found in folder: %s', inputCSV, folderPath);
            end
        
            fileData = readtable(inputFile);
        
            % Create output folder
            outputDir = fullfile(folderPath, 'window vs overlap');
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
        
            % Get all unique combinations of parameters
            [uniqueCombos, ~, idx] = unique(fileData(:, {'BIS_thr', 'MAC_thr', 'min_length', 'min_reftime'}), 'rows');
        
            for i = 1:height(uniqueCombos)
                % Get data for the current parameter combo
                comboData = fileData(idx == i, :);
        
                % Extract A, B, C, D
                A = uniqueCombos.BIS_thr(i);
                floatingMAC = uniqueCombos.MAC_thr(i);
                B_str = sprintf('%03d', round(floatingMAC * 100));  % convert float to 3-digit string
                C = uniqueCombos.min_length(i);
                D = uniqueCombos.min_reftime(i);
        
                % Create pivot-like table: rows = fixed_window, columns = overlap types
                uniqueWindows = unique(comboData.fixed_window);
                uniqueOverlaps = unique(comboData.overlap_frac);
        
                % Prepare table with one row per window size
                resultTable = table(uniqueWindows, 'VariableNames', {'window'});
        
                % Sort overlaps ascending
                uniqueOverlaps = sort(uniqueOverlaps);
        
                for j = 1:length(uniqueOverlaps)
                    ov = uniqueOverlaps(j);
                    % Create column name like 'overlap_025'
                    colName = sprintf('overlap_%03d', round(ov * 100));
        
                    % Preallocate column with NaNs
                    colData = NaN(height(resultTable), 1);
        
                    for k = 1:height(resultTable)
                        w = resultTable.window(k);
                        match = comboData.fixed_window == w & comboData.overlap_frac == ov;
        
                        if any(match)
                            colData(k) = comboData.num_episodes(match);
                        end
                    end
        
                    % Add column to table
                    resultTable.(colName) = colData;
                end
        
                % Create filename and log output
                if mergedEpisodes
                    filename = sprintf('result_merged_%d_%s_%d_%d.csv', A, B_str, C, D);
                else
                    filename = sprintf('result_%d_%s_%d_%d.csv', A, B_str, C, D);
                end
                writetable(resultTable, fullfile(outputDir, filename));
                fprintf("Succesfully created pivot table %s \n", filename);
            end
        end

        function generate_diff_merged_counts(obj)
            % This function compares normal episode counts and merged episode counts
            % and generates a pivot table showing the difference in episode counts
            % for each unique min_reftime.
            % The result is saved as 'diff_merged_counts.csv' in the folder.
        
            % variable for folder path
            folderPath = obj.resultsFolderPath;

            % Load the two source tables
            normalPath = fullfile(folderPath, 'all_episodes_count.csv');
            mergedPath = fullfile(folderPath, 'all_merged_episodes_count.csv');
        
            if ~isfile(normalPath) || ~isfile(mergedPath)
                error('Both all_episodes_count.csv and all_merged_episodes_count.csv must exist.');
            end
        
            normalTable = readtable(normalPath);
            mergedTable = readtable(mergedPath);
        
            % Define join keys (all parameters except num_episodes and min_reftime)
            joinKeys = {'BIS_thr', 'MAC_thr', 'min_length', 'fixed_window', 'overlap_frac'};
        
            % Get unique min_reftime values
            refTimes = unique(normalTable.min_reftime);
        
            % Initialize map for storing diffs
            allDiffs = [];
        
            for i = 1:length(refTimes)
                refTime = refTimes(i);
        
                % Filter for current ref_time in both tables
                normalFiltered = normalTable(normalTable.min_reftime == refTime, :);
                mergedFiltered = mergedTable(mergedTable.min_reftime == refTime, :);
        
                % Rename episode columns to include ref_time
                normalFiltered.Properties.VariableNames{'num_episodes'} = 'normal_count';
                mergedFiltered.Properties.VariableNames{'num_episodes'} = 'merged_count';
        
                % Join the two tables on the shared parameters
                mergedData = outerjoin(normalFiltered, mergedFiltered, ...
                    'Keys', joinKeys, ...
                    'MergeKeys', true, ...
                    'Type', 'full');
        
                % Fill missing counts with 0
                if ~ismember('normal_count', mergedData.Properties.VariableNames)
                    mergedData.normal_count = zeros(height(mergedData),1);
                else
                    mergedData.normal_count(isnan(mergedData.normal_count)) = 0;
                end
        
                if ~ismember('merged_count', mergedData.Properties.VariableNames)
                    mergedData.merged_count = zeros(height(mergedData),1);
                else
                    mergedData.merged_count(isnan(mergedData.merged_count)) = 0;
                end
        
                % Compute the difference
                diffCol = mergedData.normal_count - mergedData.merged_count;
                mergedData = mergedData(:, joinKeys);  % remove other columns
                mergedData.(sprintf('ref_%d', refTime)) = diffCol;
        
                % Merge into global diff table
                if isempty(allDiffs)
                    allDiffs = mergedData;
                else
                    allDiffs = outerjoin(allDiffs, mergedData, ...
                        'Keys', joinKeys, ...
                        'MergeKeys', true, ...
                        'Type', 'full');
                end
            end
        
            % Replace missing with NaN or 0 as desired (here we use 0)
            allDiffs = fillmissing(allDiffs, 'constant', 0);
        
            % Write to output file
            writetable(allDiffs, fullfile(folderPath, 'diff_merged_counts.csv'));
        end

        % plots one or two files from resultsfolder and saves the plots in
        % plotsFolderPath as matlab plots. Plots are histogramm and
        % cumulative distribution of episodes based on their lengths
        function plotEpisodeDurations(obj, csvFilename1, csvFilename2)
            % create variables
            inputFolder = obj.resultsFolderPath;
            outputFolder = obj.plotsFolderPath;

            % Search for all subfolders in inputFolder
            subFolders = dir(inputFolder);
            subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));
            
            for i = 1:length(subFolders)
                currentFolderPath = fullfile(inputFolder, subFolders(i).name);
                
                % process given CSV file(s)
                for csvFilename = {csvFilename1, csvFilename2}
                    if isempty(csvFilename{1})
                        continue;
                    end
                    
                    csvPath = fullfile(currentFolderPath, csvFilename{1});
                    if exist(csvPath, 'file')
                        % load CSV as table
                        csvData = readtable(csvPath);
                        
                        % compute episode length
                        durations = csvData.End - csvData.Start;
                        
                        % create histogramm with logarithmic scale
                        figure;
                        histogram(durations, 'BinMethod', 'fd'); % fine bars
                        set(gca, 'YScale', 'log'); % logarithmic y-scale
                        title(['Histogram of ', csvFilename{1}, ' in ', subFolders(i).name], 'Interpreter', 'none');
                        xlabel('Length of episode (s)', 'Interpreter', 'none');
                        ylabel('Frequency of occurence (log)', 'Interpreter', 'none');
                        
                        % save plot
                        plotName = fullfile(outputFolder, [subFolders(i).name, '_', csvFilename{1}, '.png']);
                        saveas(gcf, plotName);
                        close;
                        
                        % cumulative distribution plot
                        figure;
                        cdfplot(durations);
                        title(['Cumulative Distribution of ', csvFilename{1}, ' in ', subFolders(i).name], 'Interpreter', 'none');
                        xlabel('Length of episode (s)', 'Interpreter', 'none');
                        ylabel('Cumulative frequency', 'Interpreter', 'none');
                        
                        % Save plot
                        plotName = fullfile(outputFolder, [subFolders(i).name, '_', csvFilename{1}, '_CDF.png']);
                        saveas(gcf, plotName);
                        close;
                    end
                end
            end
        end

        % plots one or two files from resultsfolder and saves the plots in
        % plotsFolderPath as matlab plots. Plots are bar plots of number of episodes
        function plotEpisodeCounts(obj, csvFilename1, csvFilename2, plotName)
            % create variables
            inputFolder = obj.resultsFolderPath;
            outputFolder = obj.plotsFolderPath;

            % Search for all subfolders in inputFolder
            subFolders = dir(inputFolder);
            subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));
            
            ids = {};
            counts = [];
            
            for i = 1:length(subFolders)
                currentFolderPath = fullfile(inputFolder, subFolders(i).name);
                
                % process given CSV file(s)
                for csvFilename = {csvFilename1, csvFilename2}
                    if isempty(csvFilename{1})
                        continue;
                    end
                    
                    csvPath = fullfile(currentFolderPath, csvFilename{1});
                    if exist(csvPath, 'file')
                        % load CSV as table
                        csvData = readtable(csvPath);
                        
                        % assess number of rows
                        rowCount = height(csvData);
                        
                        % create ID and save values
                        ids{end+1} = [subFolders(i).name, '_', csvFilename{1}];
                        counts(end+1) = rowCount;
                    end
                end
            end
            
            % create bar plot
            figure;
            bar(categorical(ids), counts, 'Interpreter', 'none');
            title('Number of Episodes per File', 'Interpreter', 'none');
            xlabel('File IDs', 'Interpreter', 'none');
            ylabel('Number of Episodes', 'Interpreter', 'none');
            
            % save plot
            saveas(gcf, fullfile(outputFolder, plotName));
            close;
        end

        % Save table from containerField as CSV file in given Folder.
        % tableName is optional and omitting it will save all tables from
        % given containerField
        function saveTableAsCSV(obj, containerField, saveFolder, tableName) 
            % Ensure that saveFolder exists by creating it if not
            if ~exist(saveFolder, 'dir')
                mkdir(saveFolder);
            end

            % When tableName given, set referential path to it
           if nargin > 3
                tableRef = obj.data.(containerField).(tableName);
                % check path of saveFolder
                if istable(tableRef)
                    savePath = fullfile(saveFolder, [tableName, '.csv']);
                    writetable(tableRef, savePath);
                else
                    error('No table for path %s ', tableRef);
                end
            % When no tableName given, iterate through all tables in
            % containerField
            else
                tables = fieldnames(obj.data.(containerField));
                for i = 1:length(tables)
                    currentTable = obj.data.(containerField).(tables{i});
                    if istable (currentTable)
                        savePath = fullfile(saveFolder, [currentTable, '.csv']);
                        writetable(currentTable, savePath);
                    else
                        warning('No table for path %s ', currentTable);
                    end
                end
           end
        end

        % Will Save all Summaries
        function saveAllSummaryTablesAsCSV(obj)
            fields = fieldnames(obj.data);
            for i = 1:length(fields)
                currentField = fields{i};
                if startsWith(currentField, 'result_') % Only evaluate fields starting with 'result_'
                    saveFolder = strcat(obj.resultsFolderPath, currentField);
                    % log savings
                    fprintf("Saving Summaries from %s in path %s\n", currentField, saveFolder);
                    saveTableAsCSV(obj, currentField, saveFolder, 'Summary_Episodes');
                    saveTableAsCSV(obj, currentField, saveFolder, 'Summary_GlobalTimes');
                end
            end
        end


        % Reads and saves metadata of the VitalDB into data with paths from
        % properties
        function obj = readMetadata(obj)
            % Assemble full filepath
            fullMetadaPath = fullfile(obj.metaDataFolderPath, 'metadata_vitaldb');
            % Read file and save in field 'metadata' in data
            readSingleFile(obj, fullMetadaPath, obj.metadataField); 
        end


        % Reads all CSV-Files from the specified inputFolder in this
        % class. 
        function obj = readAllCsvInFolder(obj)

            % Find all CSV-Files in the inputFolderPath
            csvFiles = dir(fullfile(obj.inputFolderPath, '*.csv'));
            
            % Iteration Variables
            loopStart = 1;
            loopEnd = length(csvFiles);
            
            for i = loopStart:loopEnd
                % Name of current File
                tempFileName = fullfile(obj.inputFolderPath, csvFiles(i).name);

                % read File
                readSingleFile(obj, tempFileName, obj.inputTablesField);
            end
        end

    
        % Reads and saves a single CSV-File
        % @fileName: Expects a full filepath to the file with extension
        % @fieldToSave: Field in Data, where file is saved
        function obj = readSingleFile(obj, fullFilePath, fieldToSave)

            % read Data as table
            tempData = readtable(fullFilePath);
            
            % Define Name of table as filename without extension
            [~, fileName, ~] = fileparts(fullFilePath);

            % Ensure that fieldname starts with a letter
            structFileName = strcat('BIS_ID_', fileName);
                
            % Save table in specified field in data
            obj.data.(fieldToSave).(structFileName) = tempData;

            % Display that function ran succesfully
            disp("Succesfully loaded File: " + fileName + " as: " + structFileName + " in data.");
        end

    end
end