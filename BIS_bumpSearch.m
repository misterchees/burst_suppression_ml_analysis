% Suche in BIS Werten nach signifikanten Ausschlägen nach
% oben (bumps)

classdef BIS_bumpSearch < handle

    properties
        folderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed_BIS_BIS_SR\';
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