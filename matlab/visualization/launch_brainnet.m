clear
clc
close all

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

brainFile = fullfile(projectRoot,'results','matlab_multiscale_brain.mat');
brainNetRoot = fullfile(projectRoot,'third_party','BrainNet-Viewer');
outputDir = fullfile(projectRoot,'results','brainnet');

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

addpath(genpath(brainNetRoot));

data = load(brainFile);
brain = data.brain;

P = brain.population.position;
A = brain.connectivity.A;

scale = 25;
P = P * scale;

nodeFile = fullfile(outputDir,'two_factor_populations.node');
edgeFile = fullfile(outputDir,'two_factor_connectivity.edge');

fid = fopen(nodeFile,'w');

if fid < 0
    error('Unable to create node file.');
end

for i = 1:size(P,1)

    if brain.population.factor(i) == "G"
        colorIndex = 1;
    else
        colorIndex = 2;
    end

    nodeSize = 5;
    label = sprintf('%s_%s_%03d', ...
        brain.population.factor(i), ...
        brain.population.hemisphere(i), ...
        brain.population.index(i));

    fprintf(fid,'%.6f %.6f %.6f %d %.6f %s\n', ...
        P(i,1), ...
        P(i,2), ...
        P(i,3), ...
        colorIndex, ...
        nodeSize, ...
        label);
end

fclose(fid);

writematrix(A,edgeFile,'FileType','text','Delimiter','tab');

surfaceFiles = dir(fullfile( ...
    brainNetRoot, ...
    '**', ...
    '*.nv'));

if isempty(surfaceFiles)
    error('No BrainNet .nv surface file was found.');
end

surfaceFile = fullfile( ...
    surfaceFiles(1).folder, ...
    surfaceFiles(1).name);

fprintf('\nBRAINNET ADAPTER COMPLETE\n');
fprintf('Populations: %d\n',size(P,1));
fprintf('G populations: %d\n',sum(brain.population.factor == "G"));
fprintf('S populations: %d\n',sum(brain.population.factor == "S"));
fprintf('Structural edges: %d\n',nnz(A));
fprintf('Surface: %s\n',surfaceFile);
fprintf('Node file: %s\n',nodeFile);
fprintf('Edge file: %s\n',edgeFile);

H = BrainNet_MapCfg( ...
    surfaceFile, ...
    nodeFile, ...
    edgeFile);

if ~isempty(H)
    set(H,'Name','Two-Factor Connectome Brain');
end

drawnow;

