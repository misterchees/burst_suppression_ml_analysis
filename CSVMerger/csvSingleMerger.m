% Die funktion macht was im Kommentar im csvMerger ist für die Datei
% mit übergebenen Namen im Parameter singleCSVName

function csvSingleMerger(folder1, folder2, output_folder, singleCSVName)
    % Sicherstellen, dass der Ausgabeordner existiert
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    filename = singleCSVName;
    file1_path = fullfile(folder1, filename);
    file2_path = fullfile(folder2, filename);
    output_path = fullfile(output_folder, filename);

    % Datei aus Ordner1 einlesen
    data1 = readtable(file1_path);
    
    % Überprüfen, ob zweite Spalte nicht nur aus 0 oder leeren Werten besteht
    if any(data1(:,2) ~= 0 & ~isnan(data1(:,2)))
        % Prüfen, ob Datei in Ordner2 existiert
        if exist(file2_path, 'file')
            data2 = readtable(file2_path);

            % Prüfen, ob die Spaltenlängen übereinstimmen
            if size(data1,1) ~= size(data2,1)
                warning('Längendifferenz in Datei: %s (Differenz: %d)', filename, abs(size(data1,1) - size(data2,1)));
            end
            
            % Zusammenführen der Daten
            new_data = [data2, data1(:,2)];

            % Neue CSV-Datei speichern
            writetable(new_data, output_path);
        else
            warning('Datei %s existiert nicht in %s', filename, folder2);
        end
    end
end

