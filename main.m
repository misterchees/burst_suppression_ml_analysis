%% Initialize and read metadata
bumpSearcherBIS = BIS_bumpSearch();
bumpSearcherBIS = bumpSearcherBIS.readMetadata();

%% Read all CSV in Inputfolder
bumpSearcherBIS = bumpSearcherBIS.readAllCsvInFolder();

%% Search for Episodes with single configuration
bumpSearcherBIS = bumpSearcherBIS.setNewThresholds(70, 0.5);
bumpSearcherBIS = bumpSearcherBIS.setNewTimes(5,5);

bumpSearcherBIS = bumpSearcherBIS.detectEpisodesInAllTables();

%% Search for Episodes over range of configurations
rangeMAC = 0.5:0.3:0.8;
rangeEpisodeLength = [5,6,7,8,9,10,15,20];
rangeRefractoryTime = 3:5;

for i = rangeMAC
    bumpSearcherBIS = bumpSearcherBIS.setNewThresholds(70, i);
    for k = rangeEpisodeLength
        for m = rangeRefractoryTime
            bumpSearcherBIS = bumpSearcherBIS.setNewTimes(k,m);
            bumpSearcherBIS = bumpSearcherBIS.detectEpisodesInAllTables();
        end
    end
end

%% Generate Summary Tables for all results
bumpSearcherBIS = bumpSearcherBIS.generateSumaryTablesForAll();

%% Save all Summaries
bumpSearcherBIS.saveAllSummaryTablesAsCSV();

%% Generate Merged Summary Tables directly from Episodefiles in result folder
bumpSearcherBIS.mergeFlaggedEpisodes();
        
%% Fasse geflaggte Einträge zusammen
bumpSearcherBIS.mergeFlaggedEpisodes(resultsFolderPath, refractoryTimeInSeconds);

%% create fixed Window Episodes with overlap
bumpSearcherBIS.generate_windowed_episodes(6,0,0,1);

%% create fixed Window Episodes with overlap (range)
fixedEpisodesLengthRange = [5,6,7,8,9,10,15,20];
overlapRange = 0.0:0.25:0.5;

for currentOverlap = overlapRange
    for currentLength = fixedEpisodesLengthRange
        bumpSearcherBIS.generate_windowed_episodes(currentLength, currentOverlap,0,1);
        bumpSearcherBIS.generate_windowed_episodes(currentLength, currentOverlap,1,1);
    end
end

%% Calculate Global statistic for episode count
% not merged
bumpSearcherBIS.collect_episode_statistics(0);
% merged
bumpSearcherBIS.collect_episode_statistics(1);

%% Generate overlap vs window length tables
% not merged
bumpSearcherBIS.generate_overlap_summary_tables(0);
% merged
bumpSearcherBIS.generate_overlap_summary_tables(1);

%% Generate diff table for merged episodes depending on reftime
bumpSearcherBIS.generate_diff_merged_counts();

%% Get awake time and save as all relevant caseids in csv in metadata folder
bumpSearcherBIS.getAwakeTime();

%% Plot number of episodes
bumpSearcherBIS.plotEpisodeCounts(resultsFolderPath, plotsFolderPath, 'Summary_Episodes.csv','', 'Summary_Episodes_Plots')

%% Histogram of episode length for each configuration of parameter
bumpSearcherBIS.plotEpisodeDurations(resultsFolderPath, plotsFolderPath, 'Summary_Merged_Episodes.csv','')