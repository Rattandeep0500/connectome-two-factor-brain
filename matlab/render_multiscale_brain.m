clear;
clc;
close all;

data = load('../results/matlab_multiscale_brain.mat');
brain = data.brain;

P = brain.population.position;
A = brain.connectivity.A;

NX = brain.neurons.position;
NF = brain.neurons.factor;

dX = brain.neurons.dendriteX;
dY = brain.neurons.dendriteY;
dZ = brain.neurons.dendriteZ;

aX = brain.neurons.axonX;
aY = brain.neurons.axonY;
aZ = brain.neurons.axonZ;

fig = figure('Color',[0.002 0.002 0.004],'Name','Multiscale Connectome Brain','NumberTitle','off','Renderer','opengl');
set(fig,'Position',[80 50 1500 900]);

ax = axes('Parent',fig);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'vis3d');
axis(ax,'off');
set(ax,'Color',[0.002 0.002 0.004]);
set(ax,'Projection','perspective');

G = NF == "G";
S = NF == "S";

if isfield(brain.neurons,'hemisphere')
    NH = brain.neurons.hemisphere;
else
    NH = strings(size(NF));
    NH(1:floor(numel(NF)/2)) = "L";
    NH(floor(numel(NF)/2)+1:end) = "R";
end

dendriteX = [];
dendriteY = [];
dendriteZ = [];

for k = 1:numel(dX)
    qx = dX{k};
    qy = dY{k};
    qz = dZ{k};
    if isempty(qx)
        continue
    end
    dendriteX = [dendriteX; qx(:); NaN];
    dendriteY = [dendriteY; qy(:); NaN];
    dendriteZ = [dendriteZ; qz(:); NaN];
end

axonX = [];
axonY = [];
axonZ = [];

for k = 1:numel(aX)
    qx = aX{k};
    qy = aY{k};
    qz = aZ{k};
    if isempty(qx)
        continue
    end
    axonX = [axonX; qx(:); NaN];
    axonY = [axonY; qy(:); NaN];
    axonZ = [axonZ; qz(:); NaN];
end

if ~isempty(dendriteX)
    line(ax,dendriteX,dendriteY,dendriteZ,'Color',[0.08 0.48 0.72],'LineWidth',0.12);
end

if ~isempty(axonX)
    line(ax,axonX,axonY,axonZ,'Color',[0.82 0.28 0.08],'LineWidth',0.16);
end

sameHemisphereX = [];
sameHemisphereY = [];
sameHemisphereZ = [];

crossHemisphereX = [];
crossHemisphereY = [];
crossHemisphereZ = [];

for i = 1:size(A,1)
    for j = 1:size(A,2)
        if A(i,j) <= 0
            continue
        end

        p1 = P(i,:);
        p2 = P(j,:);

        if i <= numel(P(:,1)) && j <= numel(P(:,1))
            if i <= floor(size(P,1)/2) == j <= floor(size(P,1)/2)
                sameHemisphereX = [sameHemisphereX; p1(1); p2(1); NaN];
                sameHemisphereY = [sameHemisphereY; p1(2); p2(2); NaN];
                sameHemisphereZ = [sameHemisphereZ; p1(3); p2(3); NaN];
            else
                crossHemisphereX = [crossHemisphereX; p1(1); p2(1); NaN];
                crossHemisphereY = [crossHemisphereY; p1(2); p2(2); NaN];
                crossHemisphereZ = [crossHemisphereZ; p1(3); p2(3); NaN];
            end
        end
    end
end

if ~isempty(sameHemisphereX)
    line(ax,sameHemisphereX,sameHemisphereY,sameHemisphereZ,'Color',[0.18 0.18 0.25],'LineWidth',0.35);
end

if ~isempty(crossHemisphereX)
    line(ax,crossHemisphereX,crossHemisphereY,crossHemisphereZ,'Color',[0.75 0.25 0.65],'LineWidth',0.55);
end

scatter3(ax,NX(G,1),NX(G,2),NX(G,3),14,[0.05 0.72 1.0],'filled','MarkerFaceAlpha',0.82);
scatter3(ax,NX(S,1),NX(S,2),NX(S,3),14,[1.0 0.34 0.08],'filled','MarkerFaceAlpha',0.82);

scatter3(ax,P(:,1),P(:,2),P(:,3),90,[0.92 0.92 0.96],'filled','MarkerFaceAlpha',0.15);

xlabel(ax,'X');
ylabel(ax,'Y');
zlabel(ax,'Z');

title(ax, ...
    sprintf('MULTISCALE CONNECTOME-CONSTRAINED COGNITIVE BRAIN\n%d populations  |  %dB equivalent neurons  |  %d rendered neurons', ...
    brain.scale.computational_populations, ...
    round(brain.scale.equivalent_neurons/1e9), ...
    brain.scale.rendered_neurons), ...
    'Color','w','FontSize',16,'FontWeight','bold');

view(ax,3);
camproj(ax,'perspective');
camlight(ax,'headlight');
camlight(ax,'right');
lighting(ax,'gouraud');

rotate3d(fig,'on');

annotation(fig,'textbox',[0.015 0.02 0.27 0.14], ...
    'String',sprintf(['MULTISCALE BRAIN\n' ...
    'Computational populations: %d\n' ...
    'Equivalent neurons: %dB\n' ...
    'Rendered neurons: %d\n' ...
    'Dendritic coordinates: %d\n' ...
    'Axonal coordinates: %d\n' ...
    'Structural edges: %d'], ...
    brain.scale.computational_populations, ...
    round(brain.scale.equivalent_neurons/1e9), ...
    brain.scale.rendered_neurons, ...
    numel(dendriteX), ...
    numel(axonX), ...
    nnz(A)), ...
    'Color','w','FontSize',10, ...
    'BackgroundColor',[0.015 0.015 0.025], ...
    'EdgeColor',[0.3 0.3 0.4], ...
    'FitBoxToText','on');

legend(ax,{'Dendritic arbor','Axonal arbor','Population connectivity','Interhemispheric connectivity','General system','Specific system','Population centers'}, ...
    'TextColor','w','Color',[0.01 0.01 0.015], ...
    'EdgeColor',[0.25 0.25 0.3], ...
    'Location','northeast');

drawnow;

fprintf('\n');
fprintf('HIGH-DETAIL MULTISCALE RENDER COMPLETE\n');
fprintf('Computational populations: %d\n',brain.scale.computational_populations);
fprintf('Equivalent neurons: %d\n',brain.scale.equivalent_neurons);
fprintf('Rendered neurons: %d\n',brain.scale.rendered_neurons);
fprintf('Dendritic coordinates: %d\n',numel(dendriteX));
fprintf('Axonal coordinates: %d\n',numel(axonX));
fprintf('Structural edges: %d\n',nnz(A));
fprintf('\n');
