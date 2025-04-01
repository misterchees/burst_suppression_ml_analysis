% In zweiter Spalte aus CSV aus folder1 wird geschaut ob Werte existieren,
% die nicht 0 oder NaN sind -> Hinweis, ob MAC-Werte vorhanden
% Diese MAC Werte werden an die BIS Werte der entsprechenden CSV
% in folder2 drangehängt und die kombinierte CSV im output folder
% reingepackt. Falls singleCSVName (einfach der name der Datei wie 1.csv)
% übergeben wird, wird das ganze nur für die Datei gemacht, sonst für den
% ganzen Ordner

function csvMerger(folder1, folder2, output_folder, singleCSVName)

    % Fall: einzelne Datei
    if nargin >= 4
        fprintf('Einzelne Datei: %s wird angehängt an entsprechende BIS CSV\n', singleCSVName)
        csvSingleMerger(folder1, folder2, output_folder, singleCSVName);
    % Fall: Alle Dateien im Ordner
    else
        fprintf('Alle Dateien aus %s werden angehängt an entsprechende BIS CSV', folder1)
        % Liste aller CSV-Dateien in Ordner1
        files = dir(fullfile(folder1, '*.csv'));
        
        % Jede einzelne Datei mergen
        for i = 1:length(files)
            csvSingleMerger(folder1, folder2, output_folder, files(i).name);
        end
    end
end

