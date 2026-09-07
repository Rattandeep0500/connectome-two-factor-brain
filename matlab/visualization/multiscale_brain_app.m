function multiscale_brain_app

clear
clc
close all

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

brainFile = fullfile(projectRoot,'results','matlab_multiscale_brain.mat');
surfaceFile = fullfile(projectRoot,'third_party','BrainNet-Viewer','Data','SurfTemplate','BrainMesh_ICBM152.nv');

data = load(brainFile);
brain = data.brain;

[vertices,faces] = read_nv_surface(surfaceFile);

oldPopulationPosition = brain.population.position;
newPopulationPosition = map_population_coordinates(oldPopulationPosition,vertices);

A = brain.connectivity.A;
factor = brain.population.factor;
hemisphere = brain.population.hemisphere;

fig = figure( ...
    'Color',[0.005 0.007 0.012], ...
    'Name','Multiscale Cognitive Brain', ...
    'NumberTitle','off', ...
    'Renderer','opengl', ...
    'Position',[80 50 1600 920]);

ax = axes( ...
    'Parent',fig, ...
    'Position',[0.04 0.05 0.72 0.90]);

hold(ax,'on');
axis(ax,'equal');
axis(ax,'vis3d');
axis(ax,'off');
view(ax,3);

surfaceObject = patch( ...
    'Parent',ax, ...
    'Vertices',vertices, ...
    'Faces',faces, ...
    'FaceColor',[0.55 0.62 0.72], ...
    'FaceAlpha',0.16, ...
    'EdgeColor','none', ...
    'AmbientStrength',0.35, ...
    'DiffuseStrength',0.70, ...
    'SpecularStrength',0.15);

G = factor == "G";
S = factor == "S";

gObject = scatter3( ...
    ax, ...
    newPopulationPosition(G,1), ...
    newPopulationPosition(G,2), ...
    newPopulationPosition(G,3), ...
    45, ...
    [0.05 0.75 1.00], ...
    'filled', ...
    'MarkerEdgeColor',[0.85 0.95 1.00], ...
    'LineWidth',0.35);

sObject = scatter3( ...
    ax, ...
    newPopulationPosition(S,1), ...
    newPopulationPosition(S,2), ...
    newPopulationPosition(S,3), ...
    45, ...
    [1.00 0.34 0.10], ...
    'filled', ...
    'MarkerEdgeColor',[1.00 0.85 0.70], ...
    'LineWidth',0.35);

gObject.UserData = find(G);
sObject.UserData = find(S);

gObject.ButtonDownFcn = @population_click;
sObject.ButtonDownFcn = @population_click;

gObject.MarkerFaceAlpha = 0.92;
sObject.MarkerFaceAlpha = 0.92;

sameX = [];
sameY = [];
sameZ = [];

crossX = [];
crossY = [];
crossZ = [];

for i = 1:size(A,1)
    for j = i+1:size(A,2)

        if A(i,j) <= 0
            continue
        end

        p1 = newPopulationPosition(i,:);
        p2 = newPopulationPosition(j,:);

        if hemisphere(i) == hemisphere(j)
            sameX = [sameX;p1(1);p2(1);NaN];
            sameY = [sameY;p1(2);p2(2);NaN];
            sameZ = [sameZ;p1(3);p2(3);NaN];
        else
            crossX = [crossX;p1(1);p2(1);NaN];
            crossY = [crossY;p1(2);p2(2);NaN];
            crossZ = [crossZ;p1(3);p2(3);NaN];
        end

    end
end

sameConnections = plot3( ...
    ax, ...
    sameX, ...
    sameY, ...
    sameZ, ...
    'Color',[0.25 0.32 0.45], ...
    'LineWidth',0.45);

crossConnections = plot3( ...
    ax, ...
    crossX, ...
    crossY, ...
    crossZ, ...
    'Color',[0.85 0.20 0.68], ...
    'LineWidth',0.8);

sameConnections.Color(4) = 0.28;
crossConnections.Color(4) = 0.55;

selectedObject = scatter3( ...
    ax, ...
    NaN, ...
    NaN, ...
    NaN, ...
    180, ...
    [1.00 0.95 0.20], ...
    'o', ...
    'LineWidth',2);

info = annotation( ...
    fig, ...
    'textbox', ...
    [0.78 0.62 0.20 0.30], ...
    'String',initial_info(brain), ...
    'Color','w', ...
    'FontSize',11, ...
    'BackgroundColor',[0.015 0.02 0.032], ...
    'EdgeColor',[0.25 0.35 0.48], ...
    'FitBoxToText','on');

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','Whole Brain', ...
    'Units','normalized', ...
    'Position',[0.78 0.54 0.09 0.045], ...
    'Callback',@whole_brain);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','Population Detail', ...
    'Units','normalized', ...
    'Position',[0.88 0.54 0.10 0.045], ...
    'Callback',@population_detail);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','G', ...
    'Units','normalized', ...
    'Position',[0.78 0.48 0.06 0.045], ...
    'Callback',@toggle_g);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','S', ...
    'Units','normalized', ...
    'Position',[0.85 0.48 0.06 0.045], ...
    'Callback',@toggle_s);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','Connections', ...
    'Units','normalized', ...
    'Position',[0.78 0.42 0.13 0.045], ...
    'Callback',@toggle_connections);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','Surface', ...
    'Units','normalized', ...
    'Position',[0.92 0.42 0.06 0.045], ...
    'Callback',@toggle_surface);

uicontrol( ...
    fig, ...
    'Style','pushbutton', ...
    'String','Reset', ...
    'Units','normalized', ...
    'Position',[0.78 0.36 0.20 0.045], ...
    'Callback',@reset_view);

rotate3d(fig,'on');
zoom(fig,'on');
pan(fig,'on');

camlight(ax,'headlight');
camlight(ax,'right');
lighting(ax,'gouraud');
material(ax,'dull');
camproj(ax,'perspective');

title( ...
    ax, ...
    {'MULTISCALE CONNECTOME-CONSTRAINED COGNITIVE BRAIN', ...
     '200 computational populations | G + S | bilateral architecture'}, ...
    'Color','w', ...
    'FontSize',15, ...
    'FontWeight','bold');

state.selectedPopulation = 1;
state.mode = "brain";
state.neuronObjects = gobjects(0);
state.neuronLines = gobjects(0);
state.selectedNeuron = 1;

guidata(fig,state);

fprintf('\n');
fprintf('MULTISCALE BRAIN APP READY\n');
fprintf('Surface vertices: %d\n',size(vertices,1));
fprintf('Surface triangles: %d\n',size(faces,1));
fprintf('Populations: %d\n',size(newPopulationPosition,1));
fprintf('G populations: %d\n',sum(G));
fprintf('S populations: %d\n',sum(S));
fprintf('Structural edges: %d\n',nnz(A));
fprintf('Equivalent neurons: %dB\n',round(brain.scale.equivalent_neurons/1e9));
fprintf('\n');


    function population_click(src,~)

        clicked = get(src,'CurrentPoint');
        indexList = src.UserData;

        x = get(src,'XData');
        y = get(src,'YData');
        z = get(src,'ZData');

        point = clicked(1,1:3);

        distances = ...
            (x-point(1)).^2 + ...
            (y-point(2)).^2 + ...
            (z-point(3)).^2;

        [~,localIndex] = min(distances);

        populationId = indexList(localIndex);

        state = guidata(fig);
        state.selectedPopulation = populationId;
        guidata(fig,state);

        selectedObject.XData = newPopulationPosition(populationId,1);
        selectedObject.YData = newPopulationPosition(populationId,2);
        selectedObject.ZData = newPopulationPosition(populationId,3);

        info.String = population_info(brain,populationId);

    end


    function whole_brain(~,~)

        state = guidata(fig);

        delete_valid(state.neuronObjects);
        delete_valid(state.neuronLines);

        state.neuronObjects = gobjects(0);
        state.neuronLines = gobjects(0);
        state.mode = "brain";

        guidata(fig,state);

        surfaceObject.Visible = 'on';
        gObject.Visible = 'on';
        sObject.Visible = 'on';
        sameConnections.Visible = 'on';
        crossConnections.Visible = 'on';

        xlim(ax,[min(vertices(:,1)) max(vertices(:,1))]);
        ylim(ax,[min(vertices(:,2)) max(vertices(:,2))]);
        zlim(ax,[min(vertices(:,3)) max(vertices(:,3))]);

        view(ax,3);
        axis(ax,'equal');
        axis(ax,'vis3d');

        title( ...
            ax, ...
            {'MULTISCALE CONNECTOME-CONSTRAINED COGNITIVE BRAIN', ...
             '200 computational populations | G + S | bilateral architecture'}, ...
            'Color','w', ...
            'FontSize',15, ...
            'FontWeight','bold');

    end


    function population_detail(~,~)

        state = guidata(fig);
        populationId = state.selectedPopulation;

        delete_valid(state.neuronObjects);
        delete_valid(state.neuronLines);

        surfaceObject.Visible = 'off';
        gObject.Visible = 'off';
        sObject.Visible = 'off';
        sameConnections.Visible = 'off';
        crossConnections.Visible = 'off';

        centerOld = oldPopulationPosition(populationId,:);
        centerNew = newPopulationPosition(populationId,:);

        neuronPositions = brain.neurons.position;
        populationMask = brain.neurons.population == populationId;
        neuronIds = find(populationMask);

        localPositions = neuronPositions(populationMask,:) - centerOld;

        scaleFactor = 8.0;

        localPositions = localPositions * scaleFactor;

        detailPositions = localPositions + centerNew;

        neuronG = brain.neurons.factor(neuronIds) == "G";
        neuronS = brain.neurons.factor(neuronIds) == "S";

        state.neuronObjects = gobjects(2,1);

        state.neuronObjects(1) = scatter3( ...
            ax, ...
            detailPositions(neuronG,1), ...
            detailPositions(neuronG,2), ...
            detailPositions(neuronG,3), ...
            28, ...
            [0.05 0.75 1.00], ...
            'filled', ...
            'MarkerEdgeColor',[0.85 0.95 1.00], ...
            'LineWidth',0.25);

        state.neuronObjects(2) = scatter3( ...
            ax, ...
            detailPositions(neuronS,1), ...
            detailPositions(neuronS,2), ...
            detailPositions(neuronS,3), ...
            28, ...
            [1.00 0.34 0.10], ...
            'filled', ...
            'MarkerEdgeColor',[1.00 0.85 0.70], ...
            'LineWidth',0.25);

        state.neuronObjects(1).UserData = neuronIds(neuronG);
        state.neuronObjects(2).UserData = neuronIds(neuronS);

        state.neuronObjects(1).ButtonDownFcn = @neuron_click;
        state.neuronObjects(2).ButtonDownFcn = @neuron_click;

        dendriteX = [];
        dendriteY = [];
        dendriteZ = [];

        axonX = [];
        axonY = [];
        axonZ = [];

        for q = 1:numel(neuronIds)

            neuronId = neuronIds(q);

            dendrites = brain.neurons.dendrite_xyz{neuronId};

            for b = 1:numel(dendrites)
                branch = dendrites{b};
                branch = (branch-centerOld) * scaleFactor + centerNew;

                dendriteX = [dendriteX;branch(:,1);NaN];
                dendriteY = [dendriteY;branch(:,2);NaN];
                dendriteZ = [dendriteZ;branch(:,3);NaN];
            end

            axons = brain.neurons.axon_xyz{neuronId};

            for b = 1:numel(axons)
                branch = axons{b};
                branch = (branch-centerOld) * scaleFactor + centerNew;

                axonX = [axonX;branch(:,1);NaN];
                axonY = [axonY;branch(:,2);NaN];
                axonZ = [axonZ;branch(:,3);NaN];
            end

        end

        state.neuronLines = gobjects(2,1);

        state.neuronLines(1) = plot3( ...
            ax, ...
            dendriteX, ...
            dendriteY, ...
            dendriteZ, ...
            'Color',[0.10 0.65 0.90], ...
            'LineWidth',0.35);

        state.neuronLines(2) = plot3( ...
            ax, ...
            axonX, ...
            axonY, ...
            axonZ, ...
            'Color',[0.95 0.38 0.10], ...
            'LineWidth',0.45);

        guidata(fig,state);

        minPoint = min(detailPositions,[],1);
        maxPoint = max(detailPositions,[],1);

        margin = max(maxPoint-minPoint) * 0.25;

        xlim(ax,[minPoint(1)-margin maxPoint(1)+margin]);
        ylim(ax,[minPoint(2)-margin maxPoint(2)+margin]);
        zlim(ax,[minPoint(3)-margin maxPoint(3)+margin]);

        axis(ax,'equal');
        axis(ax,'vis3d');

        title( ...
            ax, ...
            sprintf('POPULATION %d | %s | %s | %d RENDERED NEURONS', ...
            populationId, ...
            brain.population.factor(populationId), ...
            brain.population.hemisphere(populationId), ...
            numel(neuronIds)), ...
            'Color','w', ...
            'FontSize',15, ...
            'FontWeight','bold');

        info.String = population_info(brain,populationId);

    end


    function neuron_click(src,~)

        clicked = get(src,'CurrentPoint');
        ids = src.UserData;

        x = src.XData;
        y = src.YData;
        z = src.ZData;

        point = clicked(1,1:3);

        distances = ...
            (x-point(1)).^2 + ...
            (y-point(2)).^2 + ...
            (z-point(3)).^2;

        [~,localIndex] = min(distances);

        neuronId = ids(localIndex);

        state = guidata(fig);
        state.selectedNeuron = neuronId;
        guidata(fig,state);

        info.String = neuron_info(brain,neuronId);

    end


    function toggle_g(~,~)

        if strcmp(gObject.Visible,'on')
            gObject.Visible = 'off';
        else
            gObject.Visible = 'on';
        end

    end


    function toggle_s(~,~)

        if strcmp(sObject.Visible,'on')
            sObject.Visible = 'off';
        else
            sObject.Visible = 'on';
        end

    end


    function toggle_connections(~,~)

        if strcmp(sameConnections.Visible,'on')
            sameConnections.Visible = 'off';
            crossConnections.Visible = 'off';
        else
            sameConnections.Visible = 'on';
            crossConnections.Visible = 'on';
        end

    end


    function toggle_surface(~,~)

        if strcmp(surfaceObject.Visible,'on')
            surfaceObject.Visible = 'off';
        else
            surfaceObject.Visible = 'on';
        end

    end


    function reset_view(~,~)

        resetCamera;

    end


    function resetCamera

        xlim(ax,[min(vertices(:,1)) max(vertices(:,1))]);
        ylim(ax,[min(vertices(:,2)) max(vertices(:,2))]);
        zlim(ax,[min(vertices(:,3)) max(vertices(:,3))]);

        axis(ax,'equal');
        axis(ax,'vis3d');
        view(ax,3);

    end

end


function [vertices,faces] = read_nv_surface(filename)

fid = fopen(filename,'r');

if fid < 0
    error('Unable to open surface file.');
end

raw = textscan(fid,'%f','CommentStyle','#');

fclose(fid);

values = raw{1};

vertexCount = values(1);

vertexStart = 2;
vertexEnd = vertexStart + vertexCount*3 - 1;

vertices = reshape(values(vertexStart:vertexEnd),3,vertexCount)';

triangleIndex = vertexEnd + 1;
triangleCount = values(triangleIndex);

faceStart = triangleIndex + 1;
faceEnd = faceStart + triangleCount*3 - 1;

faces = reshape(values(faceStart:faceEnd),3,triangleCount)';

vertices = double(vertices);
faces = double(faces);

end


function mapped = map_population_coordinates(points,vertices)

vmin = min(vertices,[],1);
vmax = max(vertices,[],1);

pmin = min(points,[],1);
pmax = max(points,[],1);

normalized = (points-pmin)./max(pmax-pmin,eps);

mapped = vmin + normalized.*(vmax-vmin);

mapped(:,1) = mapped(:,1)*0.82;
mapped(:,2) = mapped(:,2)*0.72;
mapped(:,3) = mapped(:,3)*0.72;

end


function text = initial_info(brain)

text = sprintf( ...
    ['MULTISCALE BRAIN\n\n' ...
     'Equivalent neurons: %dB\n' ...
     'Computational populations: %d\n' ...
     'General system: %d\n' ...
     'Specific system: %d\n' ...
     'Rendered neurons: %d\n' ...
     'Structural edges: %d\n\n' ...
     'Click a population\n' ...
     'then select Population Detail'], ...
    round(brain.scale.equivalent_neurons/1e9), ...
    brain.scale.computational_populations, ...
    sum(brain.population.factor == "G"), ...
    sum(brain.population.factor == "S"), ...
    brain.scale.rendered_neurons, ...
    nnz(brain.connectivity.A));

end


function text = population_info(brain,index)

neuronCount = sum(brain.neurons.population == index);

text = sprintf( ...
    ['POPULATION %03d\n\n' ...
     'Factor: %s\n' ...
     'Hemisphere: %s\n' ...
     'Population index: %d\n' ...
     'Rendered neurons: %d\n' ...
     'Equivalent neurons: 500M\n\n' ...
     'Click Population Detail\n' ...
     'Rotate / Zoom / Pan'], ...
    index, ...
    brain.population.factor(index), ...
    brain.population.hemisphere(index), ...
    brain.population.index(index), ...
    neuronCount);

end


function text = neuron_info(brain,index)

text = sprintf( ...
    ['NEURON %d\n\n' ...
     'Population: %d\n' ...
     'Factor: %s\n' ...
     'Hemisphere: %s\n' ...
     'Soma radius: %.4f'], ...
    index, ...
    brain.neurons.population(index), ...
    brain.neurons.factor(index), ...
    brain.neurons.hemisphere(index), ...
    brain.neurons.soma_radius(index));

end


function delete_valid(objects)

if isempty(objects)
    return
end

for k = 1:numel(objects)

    if isgraphics(objects(k))
        delete(objects(k));
    end

end

end
