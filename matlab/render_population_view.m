clear
clc
close all

data = load('../results/matlab_multiscale_brain.mat');
brain = data.brain;

P = brain.population.position;

G = brain.population.factor == "G";
S = brain.population.factor == "S";

figure('Color',[0.005 0.005 0.008], ...
    'Name','Multiscale Brain - Population View', ...
    'NumberTitle','off');

ax = axes;
hold(ax,'on');
axis(ax,'equal');
axis(ax,'vis3d');
axis(ax,'off');

scatter3(ax,P(G,1),P(G,2),P(G,3),70,[0.05 0.75 1.0],'filled');
scatter3(ax,P(S,1),P(S,2),P(S,3),70,[1.0 0.35 0.08],'filled');

A = brain.connectivity.A;

for i = 1:size(A,1)
    for j = i+1:size(A,2)

        if A(i,j) > 0
            line(ax, ...
                [P(i,1) P(j,1)], ...
                [P(i,2) P(j,2)], ...
                [P(i,3) P(j,3)], ...
                'Color',[0.30 0.30 0.38], ...
                'LineWidth',0.25);
        end

    end
end

title(ax, ...
    sprintf('100B-NEURON-EQUIVALENT COGNITIVE ARCHITECTURE\n%d populations | %d structural edges', ...
    brain.scale.computational_populations, ...
    nnz(A)), ...
    'Color','w', ...
    'FontSize',15, ...
    'FontWeight','bold');

view(ax,3);
camproj(ax,'perspective');

rotate3d on;

fprintf('\nPOPULATION VIEW COMPLETE\n');
fprintf('Populations: %d\n',brain.scale.computational_populations);
fprintf('Equivalent neurons: %d\n',brain.scale.equivalent_neurons);
fprintf('Structural edges: %d\n',nnz(A));
fprintf('\n');
