function prepare_and_plot_psd_sets(aw_filepath, faw_filepath, psd_aw_folder, psd_faw_folder, global_outlier_filepath, result_folder_path)
    % Build three PSD sets (faw, wrong_awake, correct_awake) and plot AUC comparisons.
    %
    % :param aw_filepath: CSV with Start, End, ResultID for awake epochs
    % :param faw_filepath: CSV with Start, End, ResultID for false awake epochs
    % :param psd_aw_folder: folder containing PSD CSVs for awake epochs
    % :param psd_faw_folder: folder containing PSD CSVs for false awake epochs
    % :param global_outlier_filepath: CSV with wrongly classified awake epochs
    % :param result_folder_path: folder to save resulting plots
    
    if ~exist(result_folder_path, 'dir')
        mkdir(result_folder_path);
    end
    
    % --- Load tables ---
    aw_tbl = readtable(aw_filepath);
    faw_tbl = readtable(faw_filepath);
    outlier_tbl = readtable(global_outlier_filepath);
    
    % Helper for unique key
    make_key = @(tbl) strcat(string(tbl.Start), "_", string(tbl.End), "_", string(tbl.ResultID));
    
    % Keys
    aw_keys = make_key(aw_tbl);
    faw_keys = make_key(faw_tbl);
    outlier_keys = make_key(outlier_tbl);
    
    % correct_awake = all awake minus outliers
    correct_awake_keys = setdiff(aw_keys, outlier_keys);
    wrong_awake_keys   = intersect(aw_keys, outlier_keys);
    
    % --- Load PSD sets ---
    [Freq, PSD_faw]          = load_psd_set(faw_keys, psd_faw_folder);
    [~,    PSD_wrong_awake]  = load_psd_set(wrong_awake_keys, psd_aw_folder);
    [~,    PSD_correct_awake]= load_psd_set(correct_awake_keys, psd_aw_folder);
    
    % Store sets in struct for easy looping
    sets.faw           = PSD_faw;
    sets.wrong_awake   = PSD_wrong_awake;
    sets.correct_awake = PSD_correct_awake;
    
    % Colors (can be adjusted in main script if needed)
    colors.faw           = {'r', [1 .8 .8]};
    colors.wrong_awake   = {'b', [.8 .8 1]};
    colors.correct_awake = {'g', [.8 1 .8]};
    
    % --- All pairwise combinations ---
    set_names = fieldnames(sets);
    combs = nchoosek(1:numel(set_names), 2);
    
    for i = 1:size(combs,1)
        s1 = set_names{combs(i,1)};
        s2 = set_names{combs(i,2)};
        
        PSD1 = sets.(s1);
        PSD2 = sets.(s2);
        
        c1  = colors.(s1){1};  cs1 = colors.(s1){2};
        c2  = colors.(s2){1};  cs2 = colors.(s2){2};
        
        [AUC, ax, skipped] = plot_PSD_AUC_wrapper(Freq, PSD1, PSD2, c1, c2, cs1, cs2,...
            'subsample', true);

        % Dateinamen für die Plots erzeugen
        out_png = fullfile(result_folder_path, sprintf('psd_auc_plot_%s_vs_%s.png', s1, s2));
        out_svg = fullfile(result_folder_path, sprintf('psd_auc_plot_%s_vs_%s.svg', s1, s2));
        
        % Figure speichern
        saveas(ax(1).Parent, out_png);  % ax(1).Parent ist die übergeordnete Figure
        saveas(ax(1).Parent, out_svg);
        
        % Figure schließen
        close(ax(1).Parent);
        
        fprintf('Saved %s vs %s plots (AUC = %.3f)\n', s1, s2, AUC(1,1));

        
        if ~isempty(skipped)
            fprintf('Skipped rows for %s vs %s:\n', s1, s2);
            disp(skipped);
        end
    
        fprintf('Finished %s vs %s\n', s1, s2);
    end

end


function [Freq, PSD_mat] = load_psd_set(keys, folder_path)
    % Load PSD matrices for given keys
    % :returns: Freq (Hz), PSD_mat (#freqs x #epochs)
    
    PSD_mat = [];
    Freq = [];
    
    for i = 1:numel(keys)
        fname = fullfile(folder_path, "PSD_" + keys(i) + ".csv");
        if ~isfile(fname)
            warning('File not found: %s', fname);
            continue;
        end
        T = readtable(fname);
       
        try
            if isempty(Freq)
                Freq = T.Frequency_Hz;
            end
            PSD_mat(:,end+1) = T.PSD_V2_per_Hz;
        catch ME
            fprintf('Problem in File: %s\n', fname);
            disp('Gefundene Spaltennamen:')
            disp(T.Properties.VariableNames)
            rethrow(ME)
        end
    end
end
