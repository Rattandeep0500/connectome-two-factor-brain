clear
clc

addpath(genpath(pwd))

cfg = default_config();

brain = build_multiscale_brain(cfg);

output_dir = fullfile(pwd,'..','results');

if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

save(fullfile(output_dir,'matlab_multiscale_brain.mat'),'brain','-v7.3');

fprintf("MULTISCALE BRAIN BUILD COMPLETE\n");
fprintf("Computational populations: %d\n",brain.scale.computational_populations);
fprintf("Equivalent neurons: %d\n",brain.scale.equivalent_neurons);
fprintf("Rendered neurons: %d\n",brain.scale.rendered_neurons);
fprintf("Compression ratio: %.2f\n",brain.scale.compression_ratio);
fprintf("Connectivity edges: %d\n",nnz(brain.connectivity.A));
fprintf("Saved: results/matlab_multiscale_brain.mat\n");
