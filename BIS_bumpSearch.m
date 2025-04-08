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
        plotsFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\plots\episode lengths v2\';
        inputTablesField = 'inputTables'; % fieldname in data structure of tables with the BIS values
        episodesTablesField = 'episodeTables'; % fieldname with tables of found episodes
        BIS_col_name = 'BIS_BIS'; % Column name for BIS values in BIS tables
        BIS_SR_col_name = 'BIS_SR'; % Column name for BIS Suppression Rate values 
        MAC_col_name = 'Primus_MAC'; % Column for MAC values
        BIS_threshold = 70; % Threshold minimum BIS
        MAC_threshold = 0.5; % Threshold minimum MAC
        min_BIS_episodeTimeInSeconds = 10; % Threshold minimum for episodes
        refractoryTimeInSeconds = 5; % Threshold minimum for time in between episodes
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

        % Searches for Episodes where BIS values and MAC values
        % simultaniously exceed certain thresholds over a minimum duration
        % and flagging episodes if duration in between is below a
        % refractory time. Thresholds and durations are class properties
        % and can be changed through setters
        function obj = detectEpisodes(obj, tableName)
            
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
        function obj = detectEpisodesInAllTables(obj, tablesField)
            
            tables = fieldnames(obj.data.(tablesField));
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


        % Search in given folder for subfolders with Episode_Summary and
        % create new table with name Summary_Merged_Episodes.csv where all
        % flagged episodes were merged together depending on refractory
        % time
        function mergeFlaggedEpisodes(obj, resultsFolder)
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
                    error('%s existiert nicht.',summaryFile);
                end

                % log
                disp(strcat('Geflaggte Episoden in ', summaryFile, ' werden zusammengefasst'));

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
                disp(strcat('Geflaggte Episoden liegen in Tabelle: ', mergedFile));
            end
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
                    error('Der angegebene Pfad ' + tableRef + 'führt nicht zu einer Tabelle.');
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
                        warning('Der angegebene Pfad ' + currentTable + 'führt nicht zu einer Tabelle.');
                    end
                end
           end
        end


        % Reads and saves metadata of the VitalDB into data with paths from
        % properties
        function obj = readMetadata(obj)
            % Assemble full filepath
            fullMetadaPath = fullfile(obj.metaDataFolderPath, 'metadata_vitaldb');
            % Read file and save in field 'metadata' in data
            readSingleFile(obj, fullMetadaPath, 'metadata') 
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
                tempFileName = csvFiles(i).name;

                % read File
                readSingleFile(obj, tempFileName, 'bisMatrices');
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
            disp("Saving File: " + fileName + " as: " + structFileName);
        end

    end
end