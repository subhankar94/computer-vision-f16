% read in data
image_points = importdata('assignment1/sfm_points.mat');

% calculate centroids and center points
centroids = sum(image_points, 2)/600; 
for k = 1:size(image_points,3)
    for l = 1:size(image_points,2)
        image_points(:,l,k) = image_points(:,l,k) - centroids(:,1,k);
    end
end

% construct W
W = [image_points(:,:, 1);
     image_points(:,:, 2);
     image_points(:,:, 3);
     image_points(:,:, 4);
     image_points(:,:, 5);
     image_points(:,:, 6);
     image_points(:,:, 7);
     image_points(:,:, 8);
     image_points(:,:, 9);
     image_points(:,:, 10);];
 
% calculate svd of W
[u, d, v] = svd(W);
 
% calculate camera locations
camera_locs = u(:, 1:3)*d(1:3,1:3);

% calculate 3-D world points
point_locs = v(:,1:3);
point_locs = point_locs';

figure(6);
plot3(point_locs(1,:), point_locs(2,:), point_locs(3,:))