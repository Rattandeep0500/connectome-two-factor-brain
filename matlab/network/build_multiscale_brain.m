function brain = build_multiscale_brain(cfg)

rng(cfg.seed);

N = cfg.brain.num_populations;

population_id = (1:N)';
factor = strings(N,1);
hemisphere = strings(N,1);
population_index = zeros(N,1);

factor(1:100) = "G";
factor(101:200) = "S";

hemisphere(1:50) = "L";
hemisphere(51:100) = "R";
hemisphere(101:150) = "L";
hemisphere(151:200) = "R";

population_index(1:50) = (1:50)';
population_index(51:100) = (1:50)';
population_index(101:150) = (1:50)';
population_index(151:200) = (1:50)';

population_position = zeros(N,3);

for i = 1:N

    h = hemisphere(i);
    f = factor(i);

    if h == "L"
        x = -cfg.geometry.brain_length * 0.22;
    else
        x = cfg.geometry.brain_length * 0.22;
    end

    y = population_index(i);

    y = -cfg.geometry.brain_width * 0.38 + ...
        (y-1) / 49 * cfg.geometry.brain_width * 0.76;

    z = cfg.geometry.brain_height * ...
        0.38 * sin((y + 2.0) * 1.35);

    if f == "G"
        z = z + 0.65;
    else
        z = z - 0.65;
    end

    x = x + randn * cfg.geometry.population_noise;
    y = y + randn * cfg.geometry.population_noise;
    z = z + randn * cfg.geometry.population_noise;

    population_position(i,:) = [x y z];

end

population_type = zeros(N,1);
population_type(factor == "G") = 1;
population_type(factor == "S") = 2;

hemisphere_type = zeros(N,1);
hemisphere_type(hemisphere == "L") = 1;
hemisphere_type(hemisphere == "R") = 2;

A_GG = zeros(N,N);
A_SS = zeros(N,N);
A_GS = zeros(N,N);
A_SG = zeros(N,N);
A_G_inter = zeros(N,N);
A_S_inter = zeros(N,N);
A_cross_inter = zeros(N,N);

for i = 1:N
    for j = 1:N

        if i == j
            continue
        end

        if factor(i) == "G" && factor(j) == "G"

            if hemisphere(i) == hemisphere(j)
                if rand < cfg.connectivity.p_gg
                    A_GG(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            else
                if rand < cfg.connectivity.p_inter_general
                    A_G_inter(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            end

        elseif factor(i) == "S" && factor(j) == "S"

            if hemisphere(i) == hemisphere(j)
                if rand < cfg.connectivity.p_ss
                    A_SS(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            else
                if rand < cfg.connectivity.p_inter_specific
                    A_S_inter(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            end

        elseif factor(i) == "G" && factor(j) == "S"

            if hemisphere(i) == hemisphere(j)
                if rand < cfg.connectivity.p_gs
                    A_GS(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            else
                if rand < cfg.connectivity.p_cross_inter
                    A_cross_inter(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            end

        elseif factor(i) == "S" && factor(j) == "G"

            if hemisphere(i) == hemisphere(j)
                if rand < cfg.connectivity.p_sg
                    A_SG(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            else
                if rand < cfg.connectivity.p_cross_inter
                    A_cross_inter(i,j) = max(0, ...
                        cfg.connectivity.weight_mean + ...
                        cfg.connectivity.weight_std * randn);
                end
            end

        end

    end
end

A = A_GG + A_SS + A_GS + A_SG + ...
    A_G_inter + A_S_inter + A_cross_inter;

neuron_count = N * cfg.micro.neurons_per_population;

neuron_population = zeros(neuron_count,1);
neuron_factor = strings(neuron_count,1);
neuron_hemisphere = strings(neuron_count,1);
neuron_position = zeros(neuron_count,3);
soma_radius = zeros(neuron_count,1);

dendrite_xyz = cell(neuron_count,1);
axon_xyz = cell(neuron_count,1);

k = 0;

for p = 1:N

    center = population_position(p,:);

    if factor(p) == "G"
        factor_scale = 1.15;
    else
        factor_scale = 0.95;
    end

    for n = 1:cfg.micro.neurons_per_population

        k = k + 1;

        angle = 2*pi*rand;
        radius = cfg.geometry.micro_scale * sqrt(rand);

        local = [
            radius*cos(angle)
            radius*sin(angle)
            0.28*radius*randn
        ]';

        pos = center + local;

        neuron_population(k) = p;
        neuron_factor(k) = factor(p);
        neuron_hemisphere(k) = hemisphere(p);
        neuron_position(k,:) = pos;

        soma_radius(k) = ...
            cfg.micro.soma_radius_min + ...
            rand * ...
            (cfg.micro.soma_radius_max - ...
            cfg.micro.soma_radius_min);

        dendrite_xyz{k} = generate_dendrites( ...
            pos, ...
            factor_scale, ...
            cfg);

        axon_xyz{k} = generate_axon( ...
            pos, ...
            factor_scale, ...
            cfg);

    end

end

brain.config = cfg;

brain.population.id = population_id;
brain.population.factor = factor;
brain.population.hemisphere = hemisphere;
brain.population.index = population_index;
brain.population.position = population_position;
brain.population.type = population_type;
brain.population.hemisphere_type = hemisphere_type;

brain.connectivity.A = A;
brain.connectivity.GG = A_GG;
brain.connectivity.SS = A_SS;
brain.connectivity.GS = A_GS;
brain.connectivity.SG = A_SG;
brain.connectivity.G_inter = A_G_inter;
brain.connectivity.S_inter = A_S_inter;
brain.connectivity.cross_inter = A_cross_inter;

brain.neurons.id = (1:neuron_count)';
brain.neurons.population = neuron_population;
brain.neurons.factor = neuron_factor;
brain.neurons.hemisphere = neuron_hemisphere;
brain.neurons.position = neuron_position;
brain.neurons.soma_radius = soma_radius;
brain.neurons.dendrite_xyz = dendrite_xyz;
brain.neurons.axon_xyz = axon_xyz;

brain.scale.equivalent_neurons = ...
    cfg.brain.total_equivalent_neurons;

brain.scale.computational_populations = N;

brain.scale.rendered_neurons = neuron_count;

brain.scale.equivalent_neurons_per_population = ...
    cfg.brain.neurons_per_population_equivalent;

brain.scale.compression_ratio = ...
    cfg.brain.neurons_per_population_equivalent / ...
    cfg.micro.neurons_per_population;

end


function branches = generate_dendrites(pos, factor_scale, cfg)

branches = cell(cfg.micro.dendrite_branches,1);

for b = 1:cfg.micro.dendrite_branches

    theta = 2*pi*rand;
    elevation = -0.9 + 1.8*rand;

    direction = [
        cos(theta)*cos(elevation)
        sin(theta)*cos(elevation)
        sin(elevation)
    ]';

    direction = direction / norm(direction);

    points = zeros(cfg.micro.dendrite_segments,3);

    points(1,:) = pos;

    for s = 2:cfg.micro.dendrite_segments

        step = 0.08 * factor_scale * ...
            (1 - 0.045*s) * ...
            (0.7 + 0.6*rand);

        perturbation = 0.32 * randn(1,3);

        direction = direction + perturbation;
        direction = direction / norm(direction);

        points(s,:) = points(s-1,:) + ...
            direction * step;

    end

    branches{b} = points;

end

end


function branches = generate_axon(pos, factor_scale, cfg)

branches = cell(cfg.micro.axon_branches,1);

for b = 1:cfg.micro.axon_branches

    theta = 2*pi*rand;

    direction = [
        0.5*cos(theta)
        0.5*sin(theta)
        -0.5 + rand
    ]';

    direction = direction / norm(direction);

    points = zeros(cfg.micro.axon_segments,3);

    points(1,:) = pos;

    for s = 2:cfg.micro.axon_segments

        step = 0.12 * factor_scale * ...
            (1 - 0.025*s) * ...
            (0.8 + 0.4*rand);

        perturbation = 0.18 * randn(1,3);

        direction = direction + perturbation;
        direction = direction / norm(direction);

        points(s,:) = points(s-1,:) + ...
            direction * step;

    end

    branches{b} = points;

end

end

