function [vertices,faces] = read_nv_surface(filename)

fid = fopen(filename,'r');

if fid < 0
    error('Unable to open surface file: %s',filename);
end

data = textscan(fid,'%f','CommentStyle','#');

fclose(fid);

values = data{1};

if isempty(values)
    error('Surface file is empty.');
end

vertexCount = values(1);

vertexStart = 2;
vertexEnd = vertexStart + vertexCount * 3 - 1;

vertices = reshape( ...
    values(vertexStart:vertexEnd), ...
    [3,vertexCount])';

triangleCountIndex = vertexEnd + 1;
triangleCount = values(triangleCountIndex);

faceStart = triangleCountIndex + 1;
faceEnd = faceStart + triangleCount * 3 - 1;

faces = reshape( ...
    values(faceStart:faceEnd), ...
    [3,triangleCount])';

vertices = double(vertices);
faces = double(faces);

end
