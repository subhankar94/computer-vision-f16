% run RANSAC
n = 100;
p = 3;
max_inliers_count = 0;
for k = 1:n
    % pick p matches at random
    samples_idx = randsample(size(matches, 2), p);
    samples = matches(:, samples_idx);
    
    % set up matrix A and vector b
    A = zeros(2*p,6);
    b = zeros(6,1);
    row_idx = 1;
    for l = 1:p;
        A(row_idx, 1:2) = f1(1:2, samples(1, l))';
        A(row_idx, 5) = 1;
        b(row_idx, 1) = f2(1, samples(2, l));
        row_idx = row_idx + 1;
        A(row_idx, 3:4) = f1(1:2, samples(1, l))';
        A(row_idx, 6) = 1;
        b(row_idx, 1) = f2(2, samples(2, l));
        row_idx = row_idx + 1;
    end
    
    % solve system
    q = A\b;
    
    % transform all points
    m = [q(1,1) q(2,1); q(3,1) q(4,1)];
    t = [q(5,1); q(6,1)];
    inliers_count = 0;
    inliers_idx = [];
    for l = 1:size(x1,2)
        d = m*[x1(l); y1(l)] + t - [x2(l); y2(l)];
        if sqrt(d'*d) < 10
            inliers_count = inliers_count + 1;
            inliers_idx = [inliers_idx l];
        end
    end
    if inliers_count > max_inliers_count
        max_inliers_count = inliers_count;
        max_inliers_idx = inliers_idx;
        best_q = q;
    end
end

% refit based on all inliers
% set up matrix A and vector b
samples = matches(:, max_inliers_idx);
p = max_inliers_count;
A = zeros(2*p,6);
b = zeros(2*p,1);
row_idx = 1;
for l = 1:p;
    A(row_idx, 1:2) = f1(1:2, samples(1, l))';
    A(row_idx, 5) = 1;
    b(row_idx, 1) = f2(1, samples(2, l));
    row_idx = row_idx + 1;
    A(row_idx, 3:4) = f1(1:2, samples(1, l))';
    A(row_idx, 6) = 1;
    b(row_idx, 1) = f2(2, samples(2, l));
    row_idx = row_idx + 1;
end
q = A\b;

H = [ q(1) q(2) q(5) ; q(3) q(4) q(6) ; 0 0 1 ];
transformed_image = imtransform(img1,maketform('affine', H'));
figure(5);
vp = (size(transformed_image,1)-size(img2,1))/2;
hp = (size(transformed_image,2)-size(img2,2))/2;
canvas = zeros(size(transformed_image));
canvas(1+vp:size(img2,1)+vp, 1+hp:size(img2,2)+hp) = img2;
imshow(horzcat(transformed_image, canvas));
%imshow(transformed_image);