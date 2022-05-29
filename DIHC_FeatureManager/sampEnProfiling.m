% Function to generate the sample entropy profile of a given time-series

%function [SE]= sampEnProfiling(ts,m)
function [SE]= sampEnProfiling(ts,m)
% SEprofile - sample entropy (SampEn)profile w.r.t parameter r
% ts is the input time-series
% m is the embedding dimension (normally take values 2 or 3)

ts = cell2mat(ts);
%disp(ts)

[b,a,range] = CHM(ts,m); % Cumuative histogram method to find r-profile

%disp('------- 1')
SEprofile=[range log(b./a)] ; % profile: column 1 is the r-value and column 2 is the corresponding SampEn value.

%disp('------- 2')
%disp(SEprofile)
%Correct for Nan and infinite values in the profile
SE=SEprofile(:,2);
SE=SE(isinf(SE)==0);
SE=SE(isnan(SE)==0);

%disp('------- 3')
% Compute TotalSampEn and AvgSampEn
%TotalSampEn=sum(SE);
%AvgSampEn=mean(SE);

%----------------------------------------------------------------------------
% coded by Radhagayathri Udhayakumar,
% radhagayathri.udhayakumar@deakin.edu.au
% Updated version: 19th September 2020
% 
% To cite:
% 1. Radhagayathri K. Udhayakumar, Chandan Karmakar, and Marimuthu
% Palaniswami, Approximate entropy profile: a novel approach to comprehend 
% irregularity of short-term HRV signal, Nonlinear Dynamics, 2016.
% 2. Radhagayathri K Udhayakumar, Chandan Karmakar, and Marimuthu 
% Palaniswami, Understanding irregularity characteristics of short-term HRV 
% signals using sample entropy profile, IEEE Transactions on Bio-Medical
% Engineering, 2018.
%----------------------------------------------------------------------------