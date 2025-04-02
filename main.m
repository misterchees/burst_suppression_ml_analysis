
%% Variablen-Deklaration
mataDataFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\';
resultsFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\results\';
plotsFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\plots\episode lengths v2\';
range = 1:6388; % Anzahl an MAC-Dateien
MatricesField = 'bisMatrices'; % Feldname BIS-Tabellen
filteredMatricesField = 'filteredWithFixedThreshold'; % Feldname nur nach BIS threshold gefilterte Tabellen
BIS_col_name = 'BIS_BIS'; % Spalte für BIS-Werte
BIS_SR_col_name = 'BIS_SR'; % Spalte für SuppRate
MAC_col_name = 'Primus_MAC'; % Spalte für MAC-Werte
BIS_threshold = 70; % Untergrenze BIS
MAC_threshold = 0.6; % Untergrenze MAC
min_BIS_episodeTimeInSeconds = 10; % Mindestzeit für mutmaßliche BS-Episoden
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

%% Suche nach allen Episoden wo BIS versagt hat und speichere Ergebnisse
% bumpSearcherBIS.detectEpisodes('BIS_ID_', BIS_col_name, MAC_col_name, BIS_threshold, MAC_threshold, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);
bumpSearcherBIS = bumpSearcherBIS.detectEpisodesInAllTables(MatricesField, BIS_col_name, MAC_col_name, BIS_threshold, MAC_threshold, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);

% Erzeuge Summary Dateien
% Feldnamen so aufbauen wie in detectEpisodes Funktion
MAC_threshold_str = strrep(sprintf('%.6f', MAC_threshold), '.', '');
resultFieldName = sprintf('result_%d_%s_%d_%d', BIS_threshold, MAC_threshold_str, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);
resultFieldName = matlab.lang.makeValidName(resultFieldName);

% Summaries erstellen
bumpSearcherBIS = bumpSearcherBIS.generateSummaryTables(resultFieldName);

% Summaries abspeichern
% Speicherpfad erstellen
resultSavingFolder = strcat(resultsFolderPath, resultFieldName);

% Beide Summaries abspeichern
bumpSearcherBIS.saveTableAsCSV(resultFieldName, resultSavingFolder, 'Summary_Episodes')
bumpSearcherBIS.saveTableAsCSV(resultFieldName, resultSavingFolder, 'Summary_GlobalTimes')

%% Suche über eine Range nach Episoden wo BIS versagt hat
% Range definieren
MAC_threshold_Range = 7:8;
min_BIS_episode_time_Range = 6:10;

for i = MAC_threshold_Range
    MAC_threshold = i*0.1;
    for j = min_BIS_episode_time_Range
        min_BIS_episodeTimeInSeconds = j;

        bumpSearcherBIS = bumpSearcherBIS.detectEpisodesInAllTables(MatricesField, BIS_col_name, MAC_col_name, BIS_threshold, MAC_threshold, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);
        
        % Erzeuge Summary Dateien
        % Feldnamen so aufbauen wie in detectEpisodes Funktion
        MAC_threshold_str = strrep(sprintf('%.6f', MAC_threshold), '.', '');
        resultFieldName = sprintf('result_%d_%s_%d_%d', BIS_threshold, MAC_threshold_str, min_BIS_episodeTimeInSeconds, refractoryTimeInSeconds);
        resultFieldName = matlab.lang.makeValidName(resultFieldName);
        
        % Summaries erstellen
        bumpSearcherBIS = bumpSearcherBIS.generateSummaryTables(resultFieldName);
        
        % Summaries abspeichern
        % Speicherpfad erstellen
        resultSavingFolder = strcat(resultsFolderPath, resultFieldName);
        
        % Beide Summaries abspeichern
        bumpSearcherBIS.saveTableAsCSV(resultFieldName, resultSavingFolder, 'Summary_Episodes')
        bumpSearcherBIS.saveTableAsCSV(resultFieldName, resultSavingFolder, 'Summary_GlobalTimes')
    end
end
        
%% Fasse geflaggte Einträge zusammen
bumpSearcherBIS.mergeFlaggedEpisodes(resultsFolderPath, refractoryTimeInSeconds);

%% Summary Anzahl der Episoden Plots erstellen
bumpSearcherBIS.plotEpisodeCounts(resultsFolderPath, plotsFolderPath, 'Summary_Episodes.csv','', 'Summary_Episodes_Plots')

%% Histogramme der Episodenlänge für jede einzelne Konfiguration der Parameter
bumpSearcherBIS.plotEpisodeDurations(resultsFolderPath, plotsFolderPath, 'Summary_Merged_Episodes.csv','')

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