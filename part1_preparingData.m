%% Preparing Data
% In this script we split our data into testing and training sets. 
% The target vectors are obtained by solving the Markowitz problem 

clear all; close all; clc;

%% Data
returnData = importdata("data/yfinance/returns8year.csv",",")';
returns = returnData.data;

[t,n] = size(returns); %t = num days, n = num stocks 

%75:25 split
trainInd = 1:1:round(0.75*t); 
testInd = (round(0.75*t)+1):1:1999;
trainRet = returns(trainInd,:);
testRet = returns(testInd,:);

%LSTM network gets standardized data 
[trainReturns, testReturns] = standardize(trainRet,testRet);

%training data
mTrain = mean(trainRet)';
sigmaTrain = cov(trainRet);
thetaTrain = corr(trainRet);

writematrix(trainRet,"data/full/returnsTrain_unstandardized.csv")
writematrix(trainReturns,"data/full/returnsTrain_standardized.csv")
writematrix(mTrain,"data/full/muTrain.csv")
writematrix(sigmaTrain,"data/full/sigmaTrain.csv")
writematrix(thetaTrain,"data/full/thetaTrain.csv")

%testing data
mTest = mean(testRet)';
sigmaTest = cov(testRet);
thetaTest = corr(testRet);

writematrix(testRet,"data/full/returnsTest_unstandardized.csv")
writematrix(testReturns,"data/full/returnsTest_standardized.csv")
writematrix(mTest,"data/full/muTest.csv")
writematrix(sigmaTest,"data/full/sigmaTest.csv")
writematrix(thetaTest,"data/full/thetaTest.csv")

% data that yields negative covariances 
returnData_negativeCor = importdata("data/yfinance/returns2year_negativeCorrelations.csv",",")';
returns_neg = returnData_negativeCor.data;
mu_neg = mean(returns_neg)';
sigma_neg = cov(returns_neg);
theta_neg = corr(returns_neg);

writematrix(mu_neg,"data/full/mu_neg.csv")
writematrix(sigma_neg,"data/full/sigma_neg.csv")
writematrix(theta_neg,"data/full/theta_neg.csv")

%% Getting target data 
% Finding maximum lambda for each set 
% find lambda that acheves the minimum risk of the portfolio. Use this
% lambda as the largest lambda for solving markowitz model
minRiskTrain = MinRisk(mTrain,sigmaTrain);
maxLambdaTrain = findLambda(minRiskTrain,mTrain,sigmaTrain);
YTrain =  markowitzTarget(mTrain, sigmaTrain,maxLambdaTrain);
writematrix(YTrain,"data/full/ytrain.csv")

minRiskTest = MinRisk(mTest,sigmaTest);
maxLambdaTest = findLambda(minRiskTest,mTest,sigmaTest);
YTest =  markowitzTarget(mTest, sigmaTest,maxLambdaTest);
writematrix(YTest,"data/full/ytest.csv")


%% Functions 
function [trainz, testz] = standardize(traindata,testdata)
mu = mean(traindata,1);
sigma = std(traindata,0,1);
trainz = (traindata-mu)./sigma;
testz = (testdata-mu)./sigma;
end

function optRisk = MinRisk(m,sigma)
model.Q = sparse(sigma);
model.A = sparse(ones(length(sigma),1)');
model.rhs = 1;
model.sense = '=';
model.modelsense = 'min';
model.lb  = zeros(length(sigma),1);
model.ub  = ones(length(sigma),1);
params.method = 0; % Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
% gurobi_write(model, 'markowitz_minRisk.lp');
results = gurobi(model,params);
optObj = results.objval;
optRisk =  (results.x)'*sigma*(results.x);
optReturn = m'*results.x;
end

function maxLambda = findLambda(minRisk,mu,sigma)
lambda = linspace(500,5000,109);

for i = 1:length(lambda)
    model.Q = -lambda(i)*sparse(sigma);
    model.obj = mu';
    model.A = sparse(ones(length(sigma),1)');
    model.rhs = 1;
    model.sense = '=';
    model.modelsense = 'max';
    model.lb  = zeros(length(sigma),1);
    model.ub  = ones(length(sigma),1);
    params.method = 0; % Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
    if i>1
        model.vbasis = results.vbasis;
        model.cbasis = results.cbasis;
    end
%     gurobi_write(model, 'markowitz.lp');
    results = gurobi(model,params);
    optRisk(i,:) =  (results.x)'*sigma*(results.x);
    optReturn(i,:) = mu'*results.x;
    w(i,:) = results.x';
    numAssets(i,:) = nnz(results.x');
    %once the risk is in our threshold, save the lambda value
    if abs(optRisk(i,:)  - minRisk) <= 1e-8
        maxLambda = lambda(i);
        break;
    end
end
end



function target =  markowitzTarget(m, sigma,maxLambda)
%Lambda values are closely distributed and become more spread out as it
%increases.
lambda = unique([linspace(0,1,121),linspace(1,2,61),linspace(2,4,51), linspace(4,7,61),linspace(7,20,61),linspace(21,60,81),linspace(60,100,51), linspace(100,200,71) linspace(200,maxLambda,81)]);

for i = 1:length(lambda)
    model.Q = -lambda(i)*sparse(sigma);
    model.obj = m';
    model.A = sparse(ones(length(sigma),1)');
    model.rhs = 1;
    model.sense = '=';
    model.modelsense = 'max';
    model.lb  = zeros(length(sigma),1);
    model.ub  = ones(length(sigma),1);
    params.method = 0; % Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
    if i>1
        model.vbasis = results.vbasis;
        model.cbasis = results.cbasis;
    end
%     gurobi_write(model, 'markowitz.lp');
    results = gurobi(model,params);
    w(i,:) = results.x';
    portfolios(i,1:length(m)) = (results.x)';
end
target_ = any(portfolios,1);
target = categorical(double(target_))';
end


