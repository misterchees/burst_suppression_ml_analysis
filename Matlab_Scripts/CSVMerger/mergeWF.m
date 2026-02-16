folderMAC = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csv_anesthesia_mac';
folderBIS = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed_BIS_BIS_SR';
output_folder = 'C:\Users\jesus\OneDrive\Dokumente\Jesús\Studium\Fächer - Bioinformatik\Praktische Arbeit und Bachelorarbeit\Material\Daten\vitaldb_csvprocessed_BIS_BIS_SR_MAC';

% Für alle Daten im Ordner einfach csvMerger nutzen
% for i = mergerRange
%     singleCSVString = strcat(i + ".csv");
%     singleCSV = convertStringsToChars(singleCSVString);
%     csvMerger(folderMAC, folderBIS, output_folder, singleCSV);
% end

% Single Merge
csvMerger(folderMAC, folderBIS, output_folder, '4007.csv');