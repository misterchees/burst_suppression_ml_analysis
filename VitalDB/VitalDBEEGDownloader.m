% VitalDB EEG Downloader
% Dieses Skript ermöglicht die Interaktion mit der VitalDB-API zum Herunterladen von EEG-Daten
% Mithilfe von Claude erstellt

% Handle Klassen, arbeiten mit Referenzen, also direkt auf dem Objekt und
% nicht mit Kopien
classdef VitalDBEEGDownloader < handle 
    properties
        apiUrl = 'https://api.vitaldb.net';  % Basis-URL der VitalDB-API
        apiKey = '';  % API-Schlüssel (muss vom Benutzer gesetzt werden)
    end
    
    methods
        function obj = VitalDBEEGDownloader(apiKey)
            % Konstruktor
            if nargin > 0 % Prüfung ob mehr als 0 Argumente übergeben wurden
                obj.apiKey = apiKey;
            end
        end
        
        function setApiKey(obj, apiKey)
            % Methode zum Setzen des API-Schlüssels
            obj.apiKey = apiKey;
        end

        function token = login(obj, username, password)
        % Generiert einen Access Token durch Authentifizierung mit Benutzername und Passwort
        % Passwort wird gemäß RFC1738 URL-kodiert
    
        % URL-Encoding des Passworts
        encodedPassword = urlencode(password);
    
        % Erstellen der Anmeldeparameter
        loginParams = struct('id', username, 'pw', encodedPassword);
    
        % POST-Anfrage an den Authentifizierungsendpoint
        endpoint = '/api/login';
        response = obj.makeRequest('POST', endpoint, loginParams);
    
        % Extrahieren des Tokens aus der Antwort (Annahme: Token ist im 'token' Feld)
        if isfield(response, 'access_token')
            token = response.token;
            % Token als API-Schlüssel setzen
            obj = obj.setApiKey(token);
            fprintf('Erfolgreich angemeldet. Token erhalten und als API-Schlüssel gesetzt.\n');
        else
            token = '';
            warning('Anmeldung fehlgeschlagen. Kein Token erhalten.');
        end
    end
        
        function studies = searchStudies(obj, params)
            % Suche nach Studien basierend auf Parametern
            endpoint = '/studies';
            studies = obj.makeRequest('GET', endpoint, params);
        end
        
        function cases = getCasesByCriteria(obj, criteria)
            % Fälle basierend auf bestimmten Kriterien abrufen
            endpoint = '/cases';
            cases = obj.makeRequest('GET', endpoint, criteria);
        end
        
        function data = getEEGData(obj, caseId, trackId)
            % EEG-Daten für einen bestimmten Fall und Track abrufen
            endpoint = sprintf('/cases/%d/tracks/%d/data', caseId, trackId);
            data = obj.makeRequest('GET', endpoint, []);
        end
        
        function tracks = getTracksForCase(obj, caseId)
            % Verfügbare Tracks für einen Fall abrufen
            endpoint = sprintf('/cases/%d/tracks', caseId);
            tracks = obj.makeRequest('GET', endpoint, []);
        end
        
        function eegTracks = findEEGTracks(obj, caseId)
            % EEG-spezifische Tracks für einen Fall finden
            tracks = obj.getTracksForCase(caseId);
            
            % Annahme: EEG-Tracks haben "EEG" im Namen oder in der Beschreibung
            eegTracks = struct([]);
            trackIdx = 1;
            
            for i = 1:length(tracks)
                track = tracks(i);
                % Prüfen, ob der Track EEG-bezogen ist (anpassen basierend auf VitalDB-Namenskonventionen)
                if contains(lower(track.name), 'eeg') || ...
                   (isfield(track, 'description') && contains(lower(track.description), 'eeg'))
                    eegTracks(trackIdx) = track;
                    trackIdx = trackIdx + 1;
                end
            end
        end
        
        function saveEEGData(obj, data, filePath)
            % EEG-Daten in einer MATLAB-Datei speichern
            eegData = data;  % Möglicherweise muss dies je nach API-Antwortformat angepasst werden
            save(filePath, 'eegData');
            fprintf('EEG-Daten wurden in %s gespeichert\n', filePath);
        end
        
        function downloadEEGForCase(obj, caseId, outputDir)
            % EEG-Daten für einen Fall herunterladen und speichern
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % EEG-Tracks finden
            eegTracks = obj.findEEGTracks(caseId);
            
            if isempty(eegTracks)
                fprintf('Keine EEG-Tracks für Fall %d gefunden\n', caseId);
                return;
            end
            
            % Für jeden EEG-Track Daten herunterladen
            for i = 1:length(eegTracks)
                track = eegTracks(i);
                fprintf('Lade EEG-Daten für Track %s (ID: %d)...\n', track.name, track.id);
                
                data = obj.getEEGData(caseId, track.id);
                
                % Dateiname erstellen
                fileName = sprintf('case_%d_track_%d_%s.mat', caseId, track.id, strrep(track.name, ' ', '_'));
                filePath = fullfile(outputDir, fileName);
                
                % Daten speichern
                obj.saveEEGData(data, filePath);
            end
            
            fprintf('Download abgeschlossen für Fall %d\n', caseId);
        end
        
        function response = makeRequest(obj, method, endpoint, params)
            % HTTP-Anfrage an die VitalDB-API senden
            url = [obj.apiUrl, endpoint];
    
            % Basis-Optionen
            options = weboptions('RequestMethod', method, 'ContentType', 'json');
            
            % API-Schlüssel hinzufügen, falls vorhanden
            if ~isempty(obj.apiKey)
                options.HeaderFields = {'X-API-Key', obj.apiKey};
            end
            
            try
                if strcmpi(method, 'GET')
                    % GET-Anfrage: Parameter zur URL hinzufügen
                    if ~isempty(params)
                        url = [url, '?', obj.buildQueryString(params)];
                    end
                    response = webread(url, options);
                else
                    % POST-Anfrage: Parameter als Body senden
                    response = webwrite(url, params, options);
                end
            catch e
                fprintf('Fehler bei der API-Anfrage: %s\n', e.message);
                response = [];
            end
        end
        
        function queryString = buildQueryString(~, params)
            % Hilfsfunktion zum Erstellen eines Query-Strings aus Parametern
            fields = fieldnames(params);
            queryString = '';
            
            for i = 1:length(fields)
                field = fields{i};
                value = params.(field);
                
                % Wert in String umwandeln, falls nötig
                if isnumeric(value)
                    value = num2str(value);
                elseif islogical(value)
                    if value
                        value = 'true';
                    else
                        value = 'false';
                    end
                end
                
                % Parameter zum Query-String hinzufügen
                if i > 1
                    queryString = [queryString, '&'];
                end
                queryString = [queryString, field, '=', urlencode(value)];
            end
        end
    end
end

