% 3(a)
% read in data
image = importdata('assignment1/image.txt',' ',0);
world = importdata('assignment1/world.txt',' ',0);

% concatenate 1's to make homogenous points
image = cat(1, image, ones(1, size(image,2)));
world = cat(1, world, ones(1, size(world,2)));

% construct matrix A
A = zeros(20,12);
for k = 1:size(image,2)
    A((2*k-1):(2*k), :) = [zeros(1,4) -world(:,k)' image(2,k)*world(:,k)';
                          world(:,k)'  zeros(1,4) -image(1,k)*world(:,k)'];
end
[u, s, v] = svd(A);

% pick eigenvector corresponding to smallest eigen value
smallest_eval_idx = 0;
smallest_eval = realmax;
for k = size(s,2):-1:1
    if s(k,k) > 0
        if s(k,k) < smallest_eval
            smallest_eval = s(k,k);
            smallest_eval_idx = k;
        end
    end
end
evec = v(:,smallest_eval_idx);

% construct P from smallest eigenvector
P = [evec(1:4)'; evec(5:8)'; evec(9:12)'];

% calculate projection
projection = P*world;
% rescale projection
for k = 1:size(projection, 2)
    projection(:,k) = projection(:,k)/projection(3,k);
end

% verify projection
sum(image-projection)';

% 3(b)
% first method to find camera co-ordinates
[u, s, v] = svd(P);
%C = cat(1, v(:,4), [1]);
C = v(:,4)/v(4,4);
P*C;

% second method to find camera co-ordinates
M = P(:,1:3);
[q, r] = qr(flipud(M)');
K = rot90(r',2); % faster fliplr(flipud(r'))
R = flipud(q');
t = K\P(:,4);
Ctilde = -R\t;
