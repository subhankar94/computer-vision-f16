function y = approxGauss(img, kernel_width)
% make sure kernel width is odd
if ~mod(kernel_width, 2)
    error('kernel width must be odd')
end

% the approximate Gaussian filter can be
% constructed by taking advantge of the
% linearity and associativity of convolutions;
   
% given kernel to be transformed into 
baseKer = 0.25*[1 2 1];
    
% create an 'identity' filter, i.e., a zero
% filter with one in the center
e_filter = zeros(kernel_width, kernel_width);
e_filter(int32(kernel_width/2), int32(kernel_width/2)) = 1;
    
% we want ker * img
% naturally, (ker * e_filter) sums to 1
% repeatedly convolve e_filter with baseKer'*baseKer
% to acheive an approximate Gaussian filter
ker = e_filter;
while ker(1,1)==0
    ker = conv2(ker, baseKer'*baseKer, 'same');
end
y = uint8(conv2(double(img), double(ker), 'valid'));
end
    