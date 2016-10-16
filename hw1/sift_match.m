% run('/usr/local/bin/vlfeat-0.9.20/toolbox/vl_setup')

% read image 1 & 2 and resize for convenience
img1 = imread('assignment1/scene.pgm');
img2 = imread('assignment1/book.pgm');
I1 = single(img1);
I2 = single(img2);

% calculate SIFT descriptors of image 1
[f1, d1] = vl_sift(I1);
% calculate SIFT descriptors of image 2
[f2, d2] = vl_sift(I2);

% overlay descriptors on image 1
figure(2);
imshow(img1);
h11 = vl_plotframe(f1(:,:));
h21 = vl_plotframe(f1(:,:));
set(h11,'color','k','linewidth',1.5);
set(h21,'color','y','linewidth',0.75);

% overlay descriptors on image 2
figure(3);
imshow(img2);
h12 = vl_plotframe(f2(:,:));
h22 = vl_plotframe(f2(:,:));
set(h12,'color','k','linewidth',1.5);
set(h22,'color','y','linewidth',0.75);

[m, s] = nn_desc(d1, d2);

[drop, perm] = sort(s, 'ascend');
m = m(:, perm);
s = s(perm);
matches = m;
scores = s;

% open a new figure
figure(4); clf;
% pad the smaller image with 0s and concatenate them
padding = size(img1)-size(img2);
imshow(cat(2, img1, padarray(img2, padding, 'post')));
% start points from 1st image
x1 = f1(1, matches(1,:));
y1 = f1(2, matches(1,:));
% end points in 2nd image, x values shifted by length
% of 1st image
x2 = f2(1, matches(2,:));
y2 = f2(2, matches(2,:));
x2p = x2 + size(img1,2);

hold on;
% print lines
h = line([x1; x2p], [y1; y2]);
set(h,'linewidth', 1);
% shift 2nd image descriptors by length of 1st image
f2p = f2; f2p(1,:) = f2p(1,:) + size(img1, 2);
des1 = vl_plotframe(f1(:,matches(1,:)));
des2 = vl_plotframe(f2p(:,matches(2,:)));
set(des1,'color','y','linewidth',2);
set(des2,'color','y','linewidth',2);
axis image off;
