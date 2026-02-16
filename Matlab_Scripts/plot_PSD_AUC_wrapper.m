function [AUC, ax, skipped] = plot_PSD_AUC_wrapper(Freq, PSD1, PSD2, c1, c2, cs1, cs2, varargin)
% Wrapper around plot_PSD_AUC_mad to handle unequal PSD sizes, faulty rows, and provide diagnostics.
% Optionally allows subsampling if PSD1 and PSD2 have different number of columns.

p = inputParser;
addParameter(p,'subsample',true,@islogical); % enable/disable automatic subsampling
parse(p,varargin{:});
do_subsample = p.Results.subsample;

skipped = []; % indices of skipped rows (empty, NaN/Inf, or constant PSDs)

% Remove empty, NaN/Inf, or constant rows in PSD1
bad1 = any(~isfinite(PSD1),2) | all(PSD1==0,2) | std(PSD1,0,2)==0;
if any(bad1)
    warning('Skipping %d problematic rows in PSD1 (empty, NaN/Inf, or constant).', sum(bad1));
    skipped = unique([skipped; find(bad1)]);
    PSD1(bad1,:) = [];
end

% Remove empty, NaN/Inf, or constant rows in PSD2
bad2 = any(~isfinite(PSD2),2) | all(PSD2==0,2) | std(PSD2,0,2)==0;
if any(bad2)
    warning('Skipping %d problematic rows in PSD2 (empty, NaN/Inf, or constant).', sum(bad2));
    skipped = unique([skipped; find(bad2)]);
    PSD2(bad2,:) = [];
end

% Check and equalize number of columns
n_cols1 = size(PSD1,2);
n_cols2 = size(PSD2,2);

if n_cols1 ~= n_cols2
    if do_subsample
        min_cols = min(n_cols1, n_cols2);
        warning('PSD matrices have different number of columns. Subsampling to %d columns.', min_cols);
        idx1 = randsample(n_cols1, min_cols);
        idx2 = randsample(n_cols2, min_cols);
        PSD1 = PSD1(:,idx1);
        PSD2 = PSD2(:,idx2);
    else
        error('PSD1 and PSD2 have different number of columns and subsampling is disabled.');
    end
end

% Warnung, falls nach allem keine Zeilen mehr übrig sind
if isempty(PSD1) || isempty(PSD2)
    error('After removing problematic rows, one of the PSD matrices is empty. Cannot proceed.');
end

% Call original plotting function
[AUC, ax] = plot_PSD_AUC_mad(Freq, PSD1, PSD2, c1, c2, cs1, cs2);
end
