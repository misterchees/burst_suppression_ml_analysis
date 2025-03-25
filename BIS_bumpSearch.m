% Suche in BIS Werten nach signifikanten Ausschlägen nach
% oben (bumps)

classdef BIS_bumpSearch < handle

    properties
        folderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed\';
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

        % Plottet eine BIS_Matrix. Name ist derjenige in der Struktur
        function plotBISMatrix(obj, name)
            matrixToPlot = obj.data.(name);
            
            % Werte der Spalten (Spaltennamen sollten entsprechend gesetzt sein)
            time = matrixToPlot.Time;
            BIS = matrixToPlot.BIS_BIS;
            BIS_SR = matrixToPlot.BIS_SR;
            
            % Plotting
            figure;  % Neue Figur öffnen
            hold on; % Beide Linien im gleichen Plot darstellen
            
            % Plot für die erste numerische Spalte (Wert 1)
            plot(time, BIS, 'r', 'LineWidth', 2);  % Rot, mit Linienbreite 2
            
            % Plot für die zweite numerische Spalte (Wert 2)
            plot(time, BIS_SR, 'b', 'LineWidth', 2);  % Blau, mit Linienbreite 2
            
            % Achsenbeschriftungen
            xlabel('Zeit (Sekunden)');
            ylabel('BIS Werte');
            
            % Titel (Name der Matrix in der Struktur)
            title(name);
            
            % Legende hinzufügen (mit den Spaltennamen)
            legend('BIS', 'BIS_SR');
            
            % Gitter hinzufügen (optional)
            grid on;
            
            hold off; % Ende der gleichzeitigen Plots

        end

        
        function searchForBumps(obj, fileName)
            % TODO
        end
    
        function obj = readCSVinFolder(obj,lowerLimit,upperLimit)
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
                readSingleFile(obj,tempFileName);
            end
        end
    
        % Einzelnes File einlesen. Erwartet wird Name mitsamt Datentyp.
        % Also example.txt statt nur example
        function obj = readSingleFile(obj, fileName)

            % Dateipfad der CSV-Datei
            fullFilePath = fullfile(obj.folderPath, fileName);

            % Daten aus der CSV-Datei einlesen 
            tempData = readtable(fullFilePath);
            
            % Dateiname ohne Endung als Feldnamen in der Struktur
            [~, fileName, ~] = fileparts(fileName);

            % Felder müssen immer mit Buchstaben beginnen
            fileName = "BIS_ID_" + string(fileName);
                
            % Speichern der Daten in der Struktur
            obj.data.(fileName) = tempData;
        end

    end
end