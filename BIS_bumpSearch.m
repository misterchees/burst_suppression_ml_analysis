% Suche in BIS Werten nach signifikanten Ausschlägen nach
% oben (bumps). Erweitert darum, zu korrelieren mit MAC Werten,
% um zu ermitteln wie oft und wo der BIS versagt hat (also Wachheit
% angezeigt, aber Anästhetikum Konzentration viel zu hoch dafür...)

classdef BIS_bumpSearch < handle

    properties
        folderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed_BIS_BIS_SR_MAC\';
        data = struct; % Datenstruktur mit allen relevanten Daten
    end

    methods
        % Konstruktor
        function obj = BIS_bumpSearch (inpFolderPath)
            if nargin > 0
                obj.folderPath = inpFolderPath;
            end
        end

        % Pfad ändern
        function obj = setNewPath(obj,inpFolderPath)
            obj.folderPath = inpFolderPath;
        end

        
        % Plottet normale und entsprechende gefilterte Tabelle
        % übereinander.
        % @tablename: Der Name der Tabelle in data.bisMatrices 
        % @filteredTableField: Der Name des Feldes in data aus dem die
        % gefilterte Matrix kommt
        function plotBISandFilteredBIS(obj, tableName, filteredTableField)

            % Überprüfen, ob die Tabelle in data.bismatrices existiert
            if ~isfield(obj.data.bisMatrices, tableName)
                error('Die Tabelle "%s" existiert nicht in data.bismatrices.', tableName);
            end

            % Überprüfen ob das übergebene Feld, in data existiert
            if ~isfield(obj.data, filteredTableField)
                error('Das Feld "%s" existiert nicht in data', filteredTableField);
            end  

            % Haupttabelle als Variable zum besseren Zugriff speichern
            mainTable = obj.data.bisMatrices.(tableName);

            % Name des gefilterten Pondons zur Haupttabelle
            filteredTableName = strcat(tableName, '_filtered');

            % Existenz der gefilterten Tabelle überprüfen
            hasFiltered = isfield(obj.data.(filteredTableField), filteredTableName);
            if hasFiltered
                filteredTable = obj.data.(filteredTableField).(filteredTableName);
            end

            % Daten für das Plotten extrahieren
            time = mainTable.Time;
            bis_bis = mainTable.BIS_BIS;
            bis_sr = mainTable.BIS_SR;
        
            if hasFiltered
                bis_bis_filtered = filteredTable.BIS_BIS;
            else
                bis_bis_filtered = NaN(size(time)); % Falls nicht vorhanden, einfach NaN setzen
            end
            % **Plot erstellen**
                figure; hold on;
            
                % Original BIS_BIS Plot (blau)
                plot(time, bis_bis, 'b-', 'DisplayName', 'BIS_BIS Original', 'LineWidth', 1.5);
            
                % Original BIS_SR Plot (grün)
                plot(time, bis_sr, 'g-', 'DisplayName', 'BIS_SR', 'LineWidth', 1.5);
            
                % Falls gefilterte Daten existieren, plotte sie in rot
                if hasFiltered
                    plot(time, bis_bis_filtered, 'r-', 'DisplayName', 'BIS_BIS Gefiltert', 'LineWidth', 1.5);
                end
            
                % Achsentitel und Legende
                xlabel('Zeit (s)');
                ylabel('Wert');
                title(sprintf('BIS Daten für %s', tableName));
                legend show;
                grid on;
                
                hold off;
        end


        % Filtert alle Zeilen aus der übergebenen Matrix, die in der
        % übergebenen Spalte 'columnName' unter dem threshold liegen.
        % Default wert für threshold falls keiner übergeben wurde
        function obj = searchWithFixedThreshold(obj, name, columnName, threshold)
            matrixToSearch = obj.data.bisMatrices.(name);
             % Überprüfen, ob die angegebene Spalte existiert
            if ~ismember(columnName, matrixToSearch.Properties.VariableNames)
                error('Die angegebene Spalte existiert nicht in der Tabelle.');
            end

            % Falls kein threshold-Wert übergeben wurde, Standardwert setzen
            if nargin < 4  % Falls keine dritte Eingabe als Parameter (threshold)
                threshold = 55;  % Standardwert für threshold
                fprintf('Kein threshold übergeben. Standardwert %d wird verwendet.\n', threshold);
            end
                
            % Logische Maske für Zeilen, bei denen der Wert <= threshold ist
            invalidRows = matrixToSearch.(columnName) <= threshold;
        
            % Name der gefilterten Matrix anzeigen
            filteredTableName = strcat(name, '_filtered');
            % Neue gefilterte Matrix erzeugen
            matrixToSearch{invalidRows, :} = NaN;
            obj.data.filteredWithFixedThreshold.(filteredTableName) = matrixToSearch;
        end

        % Sucht in Tabelle 'tableName' in Spalte 'col1' nach BIS Episoden 
        % mit mindestlänge 'minDuration', bei denen der Grenzwert 'threshold1'
        % überschritten wird und gleichzeitig über die ganze Zeit in Spalte 'col2'
        % der Grenzwert 'threshold2' überschritten wird. Falls zwischen
        % Episoden die 'minRefractoryTime' unterschritten wird, werden die
        % Episoden geflaggt, um zu zeigen, dass da wohl was net stimmt...
        function obj = detectEpisodes(obj, tableName, col1, col2, threshold1, threshold2, minDuration, minRefractoryTime)
            % Zugriff auf die Tabelle
            bisTable = obj.data.bisMatrices.(tableName);
            
            % Spalten extrahieren
            time = bisTable.Time;
            col1Data = bisTable.(col1);
            col2Data = bisTable.(col2);
            
            % Ersten und letzten gültigen Wert bestimmen
            validIdx = find(~isnan(col1Data) & col1Data ~= 0, 1, 'first');
            firstTime = time(validIdx);
            validIdx = find(~isnan(col1Data) & col1Data ~= 0, 1, 'last');
            lastTime = time(validIdx);
            
            % NaN-Werte in col2 durch lineare Interpolation ersetzen
            nanIdx = isnan(col2Data);
            validX = time(~nanIdx);
            validY = col2Data(~nanIdx);
            col2Data(nanIdx) = interp1(validX, validY, time(nanIdx), 'linear', 'extrap');
            
            % Episoden identifizieren
            aboveThreshold1 = col1Data > threshold1; % Erzeuge logischen Vektor ob Grenzwert überschritten
            episodeStart = []; 
            episodeEnd = [];
            flagged = [];
            inEpisode = false;
            startIdx = 0;

            % gehe bis zur Zeile vor dem minimalen Zeitraum
            for i = 1:length(time) - minDuration
                % Suche mit EpisodenMarker und logischem vektor
                % 'aboveThreshold1' nach Beginn einer Episode
                if ~inEpisode && all(aboveThreshold1(i:i+minDuration-1))
                    if all(col2Data(i:i+minDuration-1) > threshold2)
                        inEpisode = true;
                        startIdx = i;
                    end
                % Suche Ende der Episode und speichere Anfang und Ende ab
                elseif inEpisode && ~aboveThreshold1(i)
                    inEpisode = false;
                    episodeStart = [episodeStart; time(startIdx)];
                    episodeEnd = [episodeEnd; time(i-1)];
                end
            end
            
            % Refraktärzeit überprüfen
            flagged = false(size(episodeStart));
            % Flagging von Anfang der einen und Ende der anderen Episode
            % sorgt dafür, dass die ganze Episode im Output markiert wird
            for j = 2:length(episodeStart)
                if (episodeStart(j) - episodeEnd(j-1)) < minRefractoryTime
                    flagged(j-1) = true;
                    flagged(j) = true;
                end
            end
            
            % Fließkommazahl für Feldnamen anpassen
            threshold2_str = strrep(sprintf('%.6f', threshold2), '.', '');
            resultName = sprintf('result_%d_%s_%d_%d', threshold1, threshold2_str, minDuration, minRefractoryTime);
            resultName = matlab.lang.makeValidName(resultName);

            % Ergebnisse speichern
            episodeTable = table(episodeStart, episodeEnd, flagged, 'VariableNames', {'Start', 'End', 'Flagged'});
            obj.data.(resultName).(['result_' extractAfter(tableName, 'BIS_ID_')])... 
                = struct('Episodes', episodeTable, 'FirstValidTime', firstTime, 'LastValidTime', lastTime);
        end

        % Detect Episodes über eine Range an Tabellen
        function obj = detectEpisodesInRange(obj, range, col1, col2, threshold1, threshold2, minDuration, minRefractoryTime)
            for i = range
                fullTableName = strcat("BIS_ID_" + i);
                fullTableNameChar = convertStringsToChars(fullTableName);

                try
                    detectEpisodes(obj, fullTableNameChar, col1, col2, threshold1, threshold2, minDuration, minRefractoryTime)
                catch error
                    errorMessage = strcat('Für die Funktion detectEpisodes ist für Datei ', fullTableNameChar, ...
                    ' folgender Fehler aufgetaucht: ', error.message);
                    warning(errorMessage);
                end
            end
        end

        % Erzeugt aus allen einzelnen Tabellen in den result-Feldern des
        % übergebenen Feldes eine Summary Tabelle. Außerdem noch eine
        % zweite Tabelle mit dem ersten und letzten Ausschlag pro result 
        function obj = generateSummaryTables(obj, resultsField)
            % Summary-Array initialisieren
            episodeList = [];
            timeList = [];
            
            fields = fieldnames(obj.data.(resultsField));
            for i = 1:length(fields)
                if startsWith(fields{i}, 'result_') % Alle result_X Felder auswerten
                    resultNum = str2double(extractAfter(fields{i}, 'result_')); % Patientennummer extrahieren
                    resultStruct = obj.data.(resultsField).(fields{i}); % result Feld abspeichern zum damit weiterarbeiten
                    
                    % Falls Episoden vorhanden, werden diese dem Summary
                    % Array hinzugefügt und auch die Anfangs- und Endzeiten
                    % der Aufzeichnung in timeList abgespeichert
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
            
            % Abspeichern der fertigen Summaries in Summary_Episodes und
            % Summary_GlobalTimes
            obj.data.(resultsField).Summary_Episodes = episodeList;
            obj.data.(resultsField).Summary_GlobalTimes = timeList;
        end

        % Speichert eine Tabelle als CSV ab. Man muss nur den gewüsnchten
        % Ordner und einen Pfad angeben wo sie liegt. Falls kein
        % Tabellenname (tableName) angegeben, werden alle Tabellen aus dem
        % Pfad abgespeichert
        function saveTableAsCSV(obj, containerField, saveFolder, tableName) 
            % Sicherstellen, dass der Ausgabeordner existiert
            if ~exist(saveFolder, 'dir')
                mkdir(saveFolder);
            end

            % Falls Tabellenname übergeben, dann nur diesen abspeichern
           if nargin > 3
                tableRef = obj.data.(containerField).(tableName);
                % Prüfen ob Pfad in der Datenstruktur korrekt ist
                if istable(tableRef)
                    savePath = fullfile(saveFolder, [tableName, '.csv']);
                    writetable(tableRef, savePath);
                else
                    error('Der angegebene Pfad ' + tableRef + 'führt nicht zu einer Tabelle.');
                end
            % Falls kein Tabellenname übergeben wurde, einfach alles im Ordner, was eine Tabelle ist abspeichern    
            else
                tables = fieldnames(obj.data.(containerField));
                for i = 1:length(tables)
                    currentTable = obj.data.(containerField).tables{i};
                    if istable (currentTable)
                        savePath = fullfile(saveFolder, [currentTable, '.csv']);
                        writetable(currentTable, savePath);
                    else
                        warning('Der angegebene Pfad ' + currentTable + 'führt nicht zu einer Tabelle.');
                    end
                end
           end
        end


        % Einlesen der Metadaten zum VitalDB Datensatz
        function obj = readMetadata(obj, metadataFolderPath)
            tempPath = obj.folderPath; % Pfad in temp Variable für später speichern
            obj.folderPath = metadataFolderPath; % Pfad ersetzen mit metadata Ordner
            readSingleFile(obj, 'metadata_vitaldb', 'metadata') % Metadata auslesen und in Feld metadata speichern
            obj.folderPath = tempPath; % Pfad wieder zurücksetzen
        end


        % Liest CSVs ein. Man kann ober und Untergrenze einer Range
        % angeben, aber da die Ordnung der Dateien nicht 1,2,3.. ist
        % sondern 1,10,100... ist das ganze mittelsinnvoll :D
        function obj = readBisCsvInFolder(obj,lowerLimit,upperLimit)
            % Alle CSV-Dateien im Ordner finden
            csvFiles = dir(fullfile(obj.folderPath, '*.csv'));
            
            % Laufvariablen
            loopStart = 1;
            loopEnd = length(csvFiles);
            
            % Setzen der Begrenzungen und abfangen von out-of-bounds
            if nargin > 1
                if lowerLimit < 1
                    lowerLimit = 1;
                end
                if upperLimit > length(csvFiles)
                    upperLimit = length(csvFiles);
                end
                loopStart = lowerLimit;
                loopEnd = upperLimit;
            end


            for i = loopStart:loopEnd
                % Name der aktuellen Datei
                tempFileName = csvFiles(i).name;

                % Nutze Hilfsfunktion
                readSingleFile(obj, tempFileName, 'bisMatrices');
            end
        end

    
        % Einzelnes File einlesen. 
        % @fileName: Erwartet wird Name mitsamt Datentyp.
        % Also example.txt statt nur example
        % @fieldToSave: Name des Feldes in data
        function obj = readSingleFile(obj, fileName, fieldToSave)

            % Dateipfad der CSV-Datei
            fullFilePath = fullfile(obj.folderPath, fileName);

            % Daten aus der CSV-Datei einlesen 
            tempData = readtable(fullFilePath);
            
            % Dateiname ohne Endung als Feldnamen in der Struktur
            [~, fileName, ~] = fileparts(fileName);

            % Felder müssen immer mit Buchstaben beginnen
            structFileName = "BIS_ID_" + string(fileName);

            % Ausgabe
            disp("File: " + fileName + " wird gespeichert als: " + structFileName);
                
            % Speichern der Daten in der Struktur
            obj.data.(fieldToSave).(structFileName) = tempData;
        end

    end
end