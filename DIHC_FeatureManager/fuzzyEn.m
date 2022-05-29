
function [entr] = fuzzyEn(s, m, tau, r)
%FUZZYEN A function for fuzzy entropy
% 
% MATLAB IMPLEMENTATION FOR ENTROPY MEASURES ----- A 2015 NEW-RELEASE!
% 
% $Author:  LI, Peng
%           Shandong University, PR China
%           Email: pli@sdu.edu.cn
% $Date:    Mar. 23, 2015
%

s = cell2mat(s);
% parse inputs
narginchk(4, 4);

r = r * std(s);
% normalization
% s    = zscore(s(:));

% reconstruction
N    = length(s);
indm = hankel(1:N-m*tau, N-m*tau:N-tau);    % indexing elements for dim-m
inda = hankel(1:N-m*tau, N-m*tau:N);        % for dim-m+1
indm = indm(:, 1:tau:end);
inda = inda(:, 1:tau:end);
ym   = s(indm);
ya   = s(inda);

% using pdist for saving time, but a large RAM is required
cheb = pdist(ym, 'chebychev'); % inf-norm
cm   = sum(exp(-log(2).*(cheb./r).^2))*2 / (size(ym, 1)*(size(ym, 1)-1));

cheb = pdist(ya, 'chebychev');
ca   = sum(exp(-log(2).*(cheb./r).^2))*2 / (size(ya, 1)*(size(ya, 1)-1));

entr = -log(sum(ca) / sum(cm));

