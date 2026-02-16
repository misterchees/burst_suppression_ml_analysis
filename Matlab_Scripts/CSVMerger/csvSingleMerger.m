% Die funktion macht was im Kommentar im csvMerger ist für die Datei
% mit übergebenen Namen im Parameter singleCSVName

function csvSingleMerger(folder1, folder2, output_folder, singleCSVName)
    % Sicherstellen, dass der Ausgabeordner existiert
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Variablen setzen
    filename = singleCSVName;
    file1_path = fullfile(folder1, filename);
    file2_path = fullfile(folder2, filename);
    output_path = fullfile(output_folder, filename);

    % Prüfung ob Datei in Ordner1 existiert, sonst Abbruch
    if ~exist(file1_path, 'file')
        warning('Datei %s existiert nicht in %s', filename, folder1);
        return
    end

    % Datei aus Ordner1 einlesen
    opts1 = detectImportOptions(file1_path, 'Delimiter', ','); % Trennzeichen auf komma setzen
    data1 = readtable(file1_path, opts1);
    
    % Vorbereitung auf Überprüfung im folgenden Schritt
    data1_col2 = data1{:, 2};  % Extrahiere zweite Spalte als numerischen Vektor
    if iscell(data1_col2) % Umwandlung in numerisch falls als cell eingelesen
        data1_col2 = str2double(data1_col2);
    end

    % Überprüfen, ob zweite Spalte nicht nur aus 0 oder leeren Werten
    % besteht, sonst Abbruch
    if ~any(data1_col2 ~= 0 & ~isnan(data1_col2))
        warning('Es wurden keine MAC Werte in Datei %s gefunden. Datei wird übersprungen', filename)
        return
    end

    % Prüfen, ob Datei in Ordner2 existiert, sonst Abbruch
    if ~exist(file2_path, 'file')
        warning('Datei %s existiert nicht in %s', filename, folder2);
        return
    end

    opts2 = detectImportOptions(file1_path, 'Delimiter', ','); % Trennzeichen auf komma setzen
    data2 = readtable(file2_path, opts2);

    % Prüfen, ob die Spaltenlängen übereinstimmen
    if size(data1,1) ~= size(data2,1)
        warning('Längendifferenz in Datei: %s (Differenz: %d)', filename, abs(size(data1,1) - size(data2,1)));
    end
    
    % Zusammenführen der Daten
    new_data = [data2, data1(:,2)];

    % Neue CSV-Datei speichern
    writetable(new_data, output_path);
end

