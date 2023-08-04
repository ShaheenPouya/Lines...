% geopolygon picker and interpolator
% Bryce Karlins: 2022-10-30
% Edited by Ali Yildirim, 2023-07-21
% Creates a blank geoplot at some bounds and allows user to create and
% tweak a closed polygon. From there it performs a smoothing step with
% cubic spline interpolation with end conditions (curve fitting toolbox
% required) and then upsamples the spline path to a final point count. Also
% includes some code to plot other relevant TMS lines. 
%% Blank Geoplot
clear
clf
figure(1)
geoaxes
geobasemap satellite
% geolimits([33.032,33.0412],[-97.285,-97.278])  %TMS
% geolimits([36.268,36.2765],[-115.015,-115.0065]) %LVMS
% geolimits([33.5277,33.5367],[-86.6260,-86.6127]) %Barber
geolimits([39.5819, 39.5871],[-86.7505, -86.7393]) % Putnam


% Old Bounds, black
% oldBounds();

% Polimove QJ bounds, green
% 2021 LiDAR bounds, red
% pitLine(26,926);
% RaceyBounds();
% runData();
% nodes();
%geobasemap satellite

%% Draw/Load Polygon
% Ask user if they want to load polygon data from a file
answer = questdlg('Do you want to load polygon data from a file?', ...
    'Load Polygon', ...
    'Yes','No','Yes');

switch answer
    % If user answers Yes, load from file.
    % Else, makes user draw the polygons.
    % Lets user draw the polygons for inner and outer rois from scratch.
    % Saves them to .mat files.
    case 'Yes'
        [file,path] = uigetfile("*.mat");
        if file == 0  % user pressed cancel or closed the window
            errordlg('No file selected. Exiting.', 'Error');
            return;
        end
        
        loadedData = load([path,file]);
        
        if isfield(loadedData, 'roi_Inner')
            roi_Inner = drawpolygon('FaceAlpha', 0, 'FaceSelectable', false, 'Color', "red", 'Position', loadedData.roi_Inner.Position);
        else
            errordlg('Selected file does not contain "roi_Inner". Exiting.', 'Error');
            return;
        end
        [file,path] = uigetfile("*.mat");
        if file == 0  % user pressed cancel or closed the window
            errordlg('No file selected. Exiting.', 'Error');
            return;
        end
        loadedData = load([path,file]);
        if isfield(loadedData, 'roi_Outer')
            roi_Outer = drawpolygon('FaceAlpha', 0, 'FaceSelectable', false, 'Color', "blue", 'Position', loadedData.roi_Outer.Position);
        else
            errordlg('Selected file does not contain "roi_Outer". Exiting.', 'Error');
            return;
        end
        
    case 'No'
        msgbox('Please draw the INNER polygon.', 'Attention', 'modal');
        waitforbuttonpress;
        roi_Inner = drawpolygon('FaceAlpha', 0, 'FaceSelectable', false, 'Color', "red");
        
        % Ensure the drawn INNER polygon is a closed loop
        roi_Inner.Position = [roi_Inner.Position; roi_Inner.Position(1, :)];
    
        msgbox('Please draw the Outer polygon.', 'Attention', 'modal');
        waitforbuttonpress;
        roi_Outer = drawpolygon('FaceAlpha', 0, 'FaceSelectable', false, 'Color', "blue");
    
        % Ensure the drawn OUTER polygon is a closed loop
        roi_Outer.Position = [roi_Outer.Position; roi_Outer.Position(1, :)];
    
        save('roi_Inner.mat', 'roi_Inner');
        save('roi_Outer.mat', 'roi_Outer');
        disp('Done drawing, polygons saved');
end
%% Save ROI if needed
save('roi_Inner.mat', 'roi_Inner');
save('roi_Outer.mat', 'roi_Outer');

%% Draw Smooth lines here
% Smoothened lines
roi_Inner_interp = interpolate_roi(roi_Inner, 1800);
roi_Outer_interp = interpolate_roi(roi_Outer, 1800);

%% Clear the plot and draw smooth lines instead
delete(roi_Inner);
delete(roi_Outer);

% Draw interpolated ROIs
line(roi_Inner_interp(:, 1), roi_Inner_interp(:, 2), 'Color', 'red', 'LineWidth', 2);
line(roi_Outer_interp(:, 1), roi_Outer_interp(:, 2), 'Color', 'blue', 'LineWidth', 2);

%% Save lines to csv
% Inner smooth line
writematrix(roi_Inner_interp, 'inner_600_1800.csv');

% Outer smooth line
writematrix(roi_Outer_interp, 'outer_600_1800.csv');
%% Relax lines
% Set the tolerance and alpha
tolerance = 0.001; % adjust this value as needed
alpha = 0.1; % adjust this value as needed

% Call the function for the two polygons
[simplifiedRoi1, relaxedRoi1] = simplify_and_relax(roi_Inner_interp, tolerance, alpha);
[simplifiedRoi2, relaxedRoi2] = simplify_and_relax(roi_Outer_interp, tolerance, alpha);
%% Draw relaxed lines
delete(findobj(gca, 'Color', 'red'));
delete(findobj(gca, 'Color', 'blue'));

% Draw interpolated ROIs
line(simplifiedRoi1(:, 1), simplifiedRoi1(:, 2), 'Color', 'red', 'LineWidth', 2);
line(simplifiedRoi2(:, 1), simplifiedRoi2(:, 2), 'Color', 'blue', 'LineWidth', 2);
%% Save relaxed to csv

% Relaxed inner line
writematrix(simplifiedRoi1, 'relaxed_inner.csv');

% Relaxed outer line
writematrix(simplifiedRoi2, 'relaxed_outer.csv');

%% Functions
% Line smoothener
function [smoothLine] = interpolate_roi(raw_roi, nPoints)
    rawLine = raw_roi.Position; % get array of points from roi
    %rawLine(end+1,1:2) = rawLine(1,1:2); % copy first point to end to create closed shape

    intermedLine = interparc(600, rawLine(:,1), rawLine(:,2), 'CSAPE'); % do a coarse spline interp to smooth handdrawn path
    smoothLine = interparc(nPoints, intermedLine(:,1), intermedLine(:,2), 'CSAPE'); % upsample to final point count
end

% Line relaxer
% Function to simplify and relax polygons
function [simplifiedRoi, relaxedRoi] = simplify_and_relax(roi, tolerance, alpha)
    % Simplify the polygon using the Douglas-Peucker algorithm
    simplifiedRoi = reducepoly(roi, tolerance);

    % Relax the simplified polygon
    relaxedRoi = relax_roi(simplifiedRoi, alpha);
end

% Relaxation function
function points_relaxed = relax_roi(points, alpha)
    % Make sure the points are looping by appending the start to the end
    points = [points; points(1, :)];

    dx = gradient(points(:, 1));
    dy = gradient(points(:, 2));
    
    % Relaxation
    new_x = points(:, 1) + alpha * dx;
    new_y = points(:, 2) + alpha * dy;
    
    % Ensure that the relaxed line is also looping
    new_x(end) = new_x(1);
    new_y(end) = new_y(1);
    
    points_relaxed = [new_x, new_y];
end

