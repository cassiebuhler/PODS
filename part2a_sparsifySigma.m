%% Brute Force Search
clear all; close all; clc

sigma = readmatrix("data/full/sigmaTest.csv");
correlation = readmatrix("data/full/thetaTest.csv");

sparsifySigs(sigma,correlation,0)

sigma_neg = readmatrix("data/full/sigma_neg.csv");
theta_neg = readmatrix("data/full/theta_neg.csv");

sparsifySigs(sigma_neg,theta_neg,1)

function sparsifySigs(sigma,correlation,negative)
theta = abs(round(correlation,3));
tau = sort(unique(theta)); %list of possible tau values.
n = length(theta);
sparseSig = sigma;
completeSig = sigma;


for i = 1:length(tau)
    mask = diag(ones(1,n)) == 0;
    dropInd = abs(theta) <= tau(i); % mask to sparsify the matrix
    sparseSig(dropInd) = 0; %sparsify the covariance
    offDiag = reshape(sparseSig(mask),[n-1,n]); %matrix without diagnoals
    assetsIn = find(sum(offDiag)>0); %finding assets with off-diagonal values
    mask(assetsIn,assetsIn) = 0;
    completeSig(mask) = 0;
    sparselevels(i) = 1-(nnz(completeSig)/numel(completeSig));
    if sparselevels(i) >= 0.5 && sparselevels(i) <= 0.51
        sparse50 = completeSig;
    elseif sparselevels(i)  >= 0.6 && sparselevels(i) <= 0.62
        sparse60 = completeSig;
    elseif sparselevels(i)  >= 0.7&& sparselevels(i) <= 0.71
        sparse70 = completeSig;
    elseif sparselevels(i)  >= 0.8 && sparselevels(i) <= 0.81
        sparse80 = completeSig;
    elseif sparselevels(i)  >= 0.9 && sparselevels(i) <= 0.91
        sparse90 = completeSig;
    elseif sparselevels(i) >= 0.98 && sparselevels(i) <= 0.99
        sparse99 = completeSig;
    end

end

if negative == 0
    writematrix(sparse50,"data/sparse/sigma50.csv")

    writematrix(sparse60,"data/sparse/sigma60.csv")

    writematrix(sparse70,"data/sparse/sigma70.csv")

    writematrix(sparse80,"data/sparse/sigma80.csv")

    writematrix(sparse90,"data/sparse/sigma90.csv")

    writematrix(sparse99,"data/sparse/sigma99.csv")
elseif negative == 1

    writematrix(sparse50,"data/sparse/sigma50_neg.csv")

    writematrix(sparse60,"data/sparse/sigma60_neg.csv")

    writematrix(sparse70,"data/sparse/sigma70_neg.csv")

    writematrix(sparse80,"data/sparse/sigma80_neg.csv")

    writematrix(sparse90,"data/sparse/sigma90_neg.csv")

    writematrix(sparse99,"data/sparse/sigma99_neg.csv")
end
end


