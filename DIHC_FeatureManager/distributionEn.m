
function [DistEn]= distributionEn(data,m,M)
% The function finds the value of Distribution entropy, using Peng Li's
% method. 
% Output argument is DistEn - Distribution entropy value
% Input arguments:
% data - the input signal
% m - the embedding dimension, could use m=2
% The bin number - use Peng Li's fixed bin number M=500
data = cell2mat(data);
[row,col]=size(data);
if row==1
    data=data';
    N=col;
else
    N=row;
end

ts=data;
%Formation of template matrix for embedding dimension m
tmpltMatM=[];
for i=1:m
    tmpltMatM=[tmpltMatM ts(i:N-m+i)];
end

matLenM=size(tmpltMatM,1);
%Distance matrix for embedding dimension m
allDistM=[];
for i=1:matLenM
    tmpltVec=tmpltMatM(i,:);
    matchMat=tmpltMatM([1:i-1 i+1:matLenM],:);
    tmpltMat=repmat(tmpltVec,matLenM-1,1);
    d= max(abs(tmpltMat-matchMat),[],2);   
    allDistM=[allDistM d];
end

%Peng Li method: Random fixed bin number
freqCountBin = hist(allDistM,M);
ProbfreqCountBin = freqCountBin./length(allDistM);
%freqCountBin = hist(dist,M);
%ProbfreqCountBin = freqCountBin./length(dist);
prob=ProbfreqCountBin(ProbfreqCountBin~=0);
y=prob.*log2(prob);
DistEn=-(1.0/(log2(double(M))))*sum(double(y));
%DistEn=-(1/(log2(M)))*sum(y);
