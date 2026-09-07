function cfg = default_config()

cfg.seed = 20260907;

cfg.brain.num_general = 100;
cfg.brain.num_specific = 100;
cfg.brain.num_populations = 200;
cfg.brain.neurons_per_population_equivalent = 500000000;
cfg.brain.total_equivalent_neurons = 100000000000;

cfg.brain.general_left = 50;
cfg.brain.general_right = 50;
cfg.brain.specific_left = 50;
cfg.brain.specific_right = 50;

cfg.micro.neurons_per_population = 48;
cfg.micro.dendrite_branches = 12;
cfg.micro.dendrite_segments = 7;
cfg.micro.axon_branches = 4;
cfg.micro.axon_segments = 12;

cfg.micro.soma_radius_min = 0.035;
cfg.micro.soma_radius_max = 0.075;

cfg.micro.dendrite_radius_start = 0.018;
cfg.micro.dendrite_radius_end = 0.003;

cfg.micro.axon_radius_start = 0.012;
cfg.micro.axon_radius_end = 0.002;

cfg.geometry.brain_length = 10.0;
cfg.geometry.brain_width = 6.8;
cfg.geometry.brain_height = 5.4;

cfg.geometry.population_noise = 0.22;
cfg.geometry.micro_scale = 0.32;

cfg.connectivity.p_gg = 0.04;
cfg.connectivity.p_ss = 0.04;
cfg.connectivity.p_gs = 0.015;
cfg.connectivity.p_sg = 0.015;
cfg.connectivity.p_inter_general = 0.025;
cfg.connectivity.p_inter_specific = 0.025;
cfg.connectivity.p_cross_inter = 0.008;

cfg.connectivity.weight_mean = 0.5;
cfg.connectivity.weight_std = 0.15;

cfg.render.background = [0.01 0.01 0.015];
cfg.render.edge_alpha = 0.12;
cfg.render.neuron_alpha = 0.8;
cfg.render.soma_alpha = 0.95;

cfg.output.path = fullfile('..','results','matlab_multiscale_brain.mat');

end
