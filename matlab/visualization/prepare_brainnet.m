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

P(:,1) = P(:,1) * 18;
P(:,2) = P(:,2) * 12;
P(:,3) = P(:,3) * 12;

nodeFile = fullfile(outputDir,'two_factor_populations.node');
edgeFile = fullfile(outputDir,'two_factor_connectivity.edge');

fid = fopen(nodeFile,'w');

if fid < 0
    error('Unable to create BrainNet node file.');
end

for i = 1:size(P,1)

    if brain.population.factor(i) == "G"
        colorIndex = 1;
    else
        colorIndex = 2;
    end

    label = sprintf('%s_%s_%03d', ...
        brain.population.factor(i), ...
        brain.population.hemisphere(i), ...
        brain.population.index(i));

    fprintf(fid,'%.6f %.6f %.6f %d %.6f %s\n', ...
        P(i,1), ...
        P(i,2), ...
        P(i,3), ...
        colorIndex, ...
        5, ...
        label);
end

fclose(fid);

writematrix(A,edgeFile,'FileType','text','Delimiter','tab');

surfaceFile = fullfile( ...
    brainNetRoot, ...
    'Data', ...
    'SurfTemplate', ...
    'BrainMesh_ICBM152.nv');

if ~exist(surfaceFile,'file')
    error('ICBM152 surface not found: %s',surfaceFile);
end

fprintf('\nBRAINNET WHOLE-BRAIN ADAPTER READY\n');
fprintf('Surface: %s\n',surfaceFile);
fprintf('Populations: %d\n',size(P,1));
fprintf('G populations: %d\n',sum(brain.population.factor == "G"));
fprintf('S populations: %d\n',sum(brain.population.factor == "S"));
fprintf('Structural edges: %d\n',nnz(A));
fprintf('Node file: %s\n',nodeFile);
fprintf('Edge file: %s\n',edgeFile);
fprintf('\n');
