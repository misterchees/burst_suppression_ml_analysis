
% Variablen
mataDataFolderPath = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\';
range = 1:200;
matricesField = 'bisMatrices';

% Initialisierung und Metadaten einlesen
bumpSearcherBIS = BIS_bumpSearch();
bumpSearcherBIS = bumpSearcherBIS.readMetadata(mataDataFolderPath);
% Da die Dateien Nummern sind suche ich einfach nach den Nummern in der
% Range definiert und speicher sie intern im Feld "matricesField"
for i = range
     fileAsString = num2str(i); % Da Zahl -> in str konvertieren
     
     % Fehlerbehandlung
     try
        bumpSearcherBIS = bumpSearcherBIS.readSingleFile(fileAsString, matricesField);
     catch error
         
         disp("Beim Versuch die Datei '" + fileAsString + ...
             "' einzulesen trat folgender Fehler auf: " + error.message);
     end
end

%% 
for i = 1:10
    % Gibt mit Sicherheit eine elegantere Methode als hier, um einen char
    % string mit einer Number als bestandteil zu generieren
    fullName = strcat("BIS_ID_" + i);
    fullNameChar = convertStringsToChars(fullName);
    bumpSearcherBIS.plotBISMatrix(fullNameChar);
end