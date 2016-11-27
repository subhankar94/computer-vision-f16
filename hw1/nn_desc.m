function [matches, scores] = nn_desc(d1, d2)
    % create an array to store the index and distance
    % of the nearest neighbour
    matches = zeros(2, size(d1, 2));
    scores = zeros(1, size(d1, 2));
    
    % for each descriptor in d1, calculate the nearest
    % neighbor
    for k = 1:size(d1,2)
        min_dist = realmax;
        min_2_dist = realmax;
        min_idx = 0;
        min_2_idx = 0;
        for j = 1:size(d2, 2)
            diff_vec = double(d1(:,k)-d2(:, j));
            dist = sqrt(diff_vec'*diff_vec);
            if dist < min_dist
                min_2_dist = min_dist;
                min_2_idx = min_idx;
                min_dist = dist;
                min_idx = j;
            elseif dist < min_2_dist
                min_2_dist = dist;
                min_2_idx = j;
            end
        end
        % only save non spurious matches
        if min_dist/min_2_dist < 0.9
            matches(:, k) = [k; min_idx];
            scores(k) = min_dist;
        end
    end
    % only return non spurious matches
    matches = matches(:, any(matches,1));
    scores = scores(scores ~= 0);
end