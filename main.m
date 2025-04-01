
% Variablen
mataDataFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\';
range = 1:6338; % Anzahl an MAC-Dateien
MatricesField = 'bisMatrices'; % Feldname BIS-Tabellen
filteredMatricesField = 'filteredWithFixedThreshold'; % Feldname nur nach BIS threshold gefilterte Tabellen
BIS_col_name = 'BIS_BIS'; % Spalte für BIS-Werte
BIS_SR_col_name = 'BIS_SR'; % Spalte für SuppRate
MAC_col_name = 'Primus_MAC'; % Spalte für MAC-Werte
BIS_threshold = 70; % Untergrenze BIS
MAC_threshold = 0.5; % Untergrenze MAC
min_BIS_episodeTimeInSeconds = 5; % Mindestzeit für mutmaßliche BS-Episoden
refractoryTimeInSeconds = 5; % Mindestzeit zw. zwei unterschiedlichen mutmaßlichen BS-Episoden

%% Initialisierung und Metadaten einlesen
bumpSearcherBIS = BIS_bumpSearch();
bumpSearcherBIS = bumpSearcherBIS.readMetadata(mataDataFolderPath);
% Da die Dateien Nummern sind suche ich einfach nach den Nummern in der
% Range definiert und speicher sie intern im Feld "matricesField"
for i = range
     fileAsString = num2str(i); % Da Zahl -> in str konvertieren
     
     % Fehlerbehandlung
     try
        bumpSearcherBIS = bumpSearcherBIS.readSingleFile(fileAsString, MatricesField);
     catch error
         
         disp("Beim Versuch die Datei '" + fileAsString + ...
             "' einzulesen trat folgender Fehler auf: " + error.message);
     end
end

%% Suche nach allen Episoden wo BIS versagt hat
% bumpSearcherBIS.detectEpisodes('BIS_ID_', BIS_col_name, MAC_col_name, BIS_threshold, MAC_threshold, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);
bumpSearcherBIS = bumpSearcherBIS.detectEpisodesInRange(range, BIS_col_name, MAC_col_name, BIS_threshold, MAC_threshold, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);

%% Plotte eine Range an Daten
for i = 1:10
    % Gibt mit Sicherheit eine elegantere Methode als hier, um einen char
    % string mit einer Number als bestandteil zu generieren
    fullName = strcat("BIS_ID_" + i);
    fullNameChar = convertStringsToChars(fullName);
    bumpSearcherBIS.plotBISandFilteredBIS(fullNameChar,filteredMatricesField);
end

%% Suche mit fixem Grenzwert
for i = 1:10
    % Gibt mit Sicherheit eine elegantere Methode als hier, um einen char
    % string mit einer Number als bestandteil zu generieren
    fullName = strcat("BIS_ID_" + i);
    fullNameChar = convertStringsToChars(fullName);
    bumpSearcherBIS = bumpSearcherBIS.searchWithFixedThreshold(fullNameChar, 'BIS_BIS');
end