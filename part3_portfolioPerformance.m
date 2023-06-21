% clear all;close all;clc
%% Full Data
% read in full data
mu= readmatrix("data/full/muTest.csv");
sigma = readmatrix("data/full/sigmaTest.csv");
ytest = readmatrix("data/full/ytest.csv");

lastDay = readmatrix("data/yfinance/lastDayReturns8year.csv");
lastDay = lastDay(2:end);

%getting max lambda
minRisk = MinRisk(mu,sigma);
maxLambda = findLambda(minRisk,mu,sigma);

%% Sparse Data
sigma50= readmatrix("data/sparse/sigma50.csv");
sigma60= readmatrix("data/sparse/sigma60.csv");
sigma70= readmatrix("data/sparse/sigma70.csv");
sigma80= readmatrix("data/sparse/sigma80.csv");
sigma90= readmatrix("data/sparse/sigma90.csv");
sigma99= readmatrix("data/sparse/sigma99.csv");

%getting max lambda
sparseSigs = {sigma50,sigma60,sigma70,sigma80,sigma90,sigma99};
minRiskSparse=[];
maxLambdaSparse = [];

for i = 1:length(sparseSigs)
    minRiskSparse(i) = MinRisk(mu,sparseSigs{i});
    maxLambdaSparse(i) = findLambda(minRiskSparse(i),mu,sparseSigs{i});
end

%% Sparse Data - neg correlations 
mu_neg= readmatrix("data/full/mu_neg.csv");
sigma_neg= readmatrix("data/full/sigma_neg.csv");
lastDay_neg= readmatrix("data/yfinance/lastDayReturns2year_negativeCorrelations.csv");
lastDay_neg = lastDay_neg(2:end);

sigma50_neg= readmatrix("data/sparse/sigma50_neg.csv");
sigma60_neg= readmatrix("data/sparse/sigma60_neg.csv");
sigma70_neg= readmatrix("data/sparse/sigma70_neg.csv");
sigma80_neg= readmatrix("data/sparse/sigma80_neg.csv");
sigma90_neg= readmatrix("data/sparse/sigma90_neg.csv");
sigma99_neg= readmatrix("data/sparse/sigma99_neg.csv");

%getting max lambda
sparseSigs_neg = {sigma50_neg,sigma60_neg,sigma70_neg,sigma80_neg,sigma90_neg,sigma99_neg};
minRiskSparse_neg=[];
maxLambdaSparse_neg = [];

% full negative
minRisk_neg = MinRisk(mu_neg,sigma_neg);
maxLambda_neg = findLambda(minRisk_neg,mu_neg,sigma_neg);

% sparse negative
for i = 1:length(sparseSigs_neg)
    minRiskSparse_neg(i) = MinRisk(mu_neg,sparseSigs_neg{i});
    maxLambdaSparse_neg(i) = findLambda(minRiskSparse_neg(i),mu_neg,sparseSigs_neg{i});
end




%% Reduced LSTM Data
ypredLSTM= readmatrix("data/reduced_LSTM/ypred.csv");
muLSTM = readmatrix("data/reduced_LSTM/muTest.csv");
sigmaLSTM= readmatrix("data/reduced_LSTM/sigmaTest.csv");
lastDayLSTM = lastDay(ypredLSTM==1);

%getting max lambda for LP
minRiskLSTM = MinRisk(muLSTM,sigmaLSTM);
maxLambdaLSTM = findLambda(minRiskLSTM,muLSTM,sigmaLSTM);


%% Reduced LP Data
ypredLP= readmatrix("data/reduced_LP/ypred.csv");
muLP= readmatrix("data/reduced_LP/muTest.csv");
sigmaLP= readmatrix("data/reduced_LP/sigmaTest.csv");
lastDayLP = lastDay(ypredLP==1);

%getting max lambda for LP
minRiskLP = MinRisk(muLP,sigmaLP);
maxLambdaLP = findLambda(minRiskLP,muLP,sigmaLP);


%% Compare portfolios
%look at predictions for reduced data 
getConfusionMat(ytest,ypredLSTM,ypredLP) 

%get porfolio results
[resultsFull,metricsFull] = markowitz(mu,sigma,lastDay,sigma,maxLambda);

[resultsLSTM,metricsLSTM] = markowitz(muLSTM,sigmaLSTM,lastDayLSTM,sigmaLSTM,maxLambdaLSTM);

[resultsLP,metricsLP] = markowitz(muLP,sigmaLP,lastDayLP,sigmaLP,maxLambdaLP);


[results50,metrics50] = markowitz(mu,sigma50,lastDay,sigma,maxLambdaSparse(1));
[results60,metrics60] = markowitz(mu,sigma60,lastDay,sigma,maxLambdaSparse(2));
[results70,metrics70] = markowitz(mu,sigma70,lastDay,sigma,maxLambdaSparse(3));
[results80,metrics80] = markowitz(mu,sigma80,lastDay,sigma,maxLambdaSparse(4));
[results90,metrics90] = markowitz(mu,sigma90,lastDay,sigma,maxLambdaSparse(5));
[results99,metrics99] = markowitz(mu,sigma99,lastDay,sigma,maxLambdaSparse(6));

[resultsFull_neg,metricsFull_neg] = markowitz(mu_neg,sigma_neg,lastDay_neg,sigma_neg,maxLambda_neg);

[results50_neg,metrics50_neg] = markowitz(mu_neg,sigma50_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(1));
[results60_neg,metrics60_neg] = markowitz(mu_neg,sigma60_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(2));
[results70_neg,metrics70_neg] = markowitz(mu_neg,sigma70_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(3));
[results80_neg,metrics80_neg] = markowitz(mu_neg,sigma80_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(4));
[results90_neg,metrics90_neg] = markowitz(mu_neg,sigma90_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(5));
[results99_neg,metrics99_neg] = markowitz(mu_neg,sigma99_neg,lastDay_neg,sigma_neg,maxLambdaSparse_neg(6));




%% Save Portfolio Output
% 
% writetable(resultsFull,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/resultsFull.csv")
% writetable(resultsLSTM,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/resultsLSTM.csv")
% writetable(resultsLP,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/resultsLP.csv")
% 
% writetable(results50,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results50.csv")
% writetable(results60,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results60.csv")
% writetable(results70,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results70.csv")
% writetable(results80,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results80.csv")
% writetable(results90,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results90.csv")
% writetable(results99,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results99.csv")
% 
% writetable(resultsFull_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/resultsFull_neg.csv")
% writetable(results50_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results50_neg.csv")
% writetable(results60_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results60_neg.csv")
% writetable(results70_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results70_neg.csv")
% writetable(results80_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results80_neg.csv")
% writetable(results90_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results90_neg.csv")
% writetable(results99_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/results99_neg.csv")
% % 
% % 
% % 
% writetable(metricsFull,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metricsFull.csv")
% writetable(metricsLSTM,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metricsLSTM.csv")
% writetable(metricsLP,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metricsLP.csv")
% 
% writetable(metrics50,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics50.csv")
% writetable(metrics60,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics60.csv")
% writetable(metrics70,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics70.csv")
% writetable(metrics80,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics80.csv")
% writetable(metrics90,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics90.csv")
% writetable(metrics99,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics99.csv")
% 
% writetable(metricsFull_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metricsFull_neg.csv")
% writetable(metrics50_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics50_neg.csv")
% writetable(metrics60_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics60_neg.csv")
% writetable(metrics70_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics70_neg.csv")
% writetable(metrics80_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics80_neg.csv")
% writetable(metrics90_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics90_neg.csv")
% writetable(metrics99_neg,"/Users/cassiebuhler/Desktop/MarkowitzCode/results/portfolios/metrics99_neg.csv")


%% Get optimizer results
[AllRunTimes0,AllIterations0] = timingMarkowitz(mu,sigma,'Full',maxLambda);
stat0 = getOptimizerStats(AllRunTimes0,AllIterations0);
% writematrix(AllRunTimes0,"results/optimizer/AllRunTimesFull.csv")
% writematrix(AllIterations0,"results/optimizer/AllIterationsFull.csv")

[AllRunTimesLP,AllIterationsLP] = timingMarkowitz(muLP,sigmaLP,'LP',maxLambdaLP);
statLP = getOptimizerStats(AllRunTimesLP,AllIterationsLP);
% writematrix(AllRunTimesLP,"results/optimizer/AllRunTimesLP.csv")
% writematrix(AllIterationsLP,"results/optimizer/AllIterationsLP.csv")

[AllRunTimesLSTM,AllIterationsLSTM] = timingMarkowitz(muLSTM,sigmaLSTM,'LSTM',maxLambdaLSTM);
statLSTM = getOptimizerStats(AllRunTimesLSTM,AllIterationsLSTM);
% writematrix(AllRunTimesLSTM,"results/optimizer/AllRunTimesLSTM.csv")
% writematrix(AllIterationsLSTM,"results/optimizer/AllIterationsLSTM.csv")

[AllRunTimes99,AllIterations99] =  timingMarkowitz(mu,sigma99,'Sparse99',maxLambdaSparse(6));
stat99 = getOptimizerStats(AllRunTimes99,AllIterations99);
% writematrix(AllRunTimes99,"results/optimizer/AllRunTimes99.csv")
% writematrix(AllIterations99,"results/optimizer/AllIterations99.csv")

[AllRunTimes90,AllIterations90] = timingMarkowitz(mu,sigma90,'Sparse90',maxLambdaSparse(5));
stat90 = getOptimizerStats(AllRunTimes90,AllIterations90);
% writematrix(AllRunTimes90,"results/optimizer/AllRunTimes90.csv")
% writematrix(AllIterations90,"results/optimizer/AllIterations90.csv")

[AllRunTimes80,AllIterations80] = timingMarkowitz(mu,sigma80,'Sparse80',maxLambdaSparse(4));
stat80 = getOptimizerStats(AllRunTimes80,AllIterations80);
% writematrix(AllRunTimes80,"results/optimizer/AllRunTimes80.csv")
% writematrix(AllIterations80,"results/optimizer/AllIterations80.csv")

[AllRunTimes70,AllIterations70] = timingMarkowitz(mu,sigma70,'Sparse70',maxLambdaSparse(3));
stat70 = getOptimizerStats(AllRunTimes70,AllIterations70);
% writematrix(AllRunTimes70,"results/optimizer/AllRunTimes70.csv")
% writematrix(AllIterations70,"results/optimizer/AllIterations70.csv")

[AllRunTimes60,AllIterations60] = timingMarkowitz(mu,sigma60,'Sparse60',maxLambdaSparse(2));
stat60 = getOptimizerStats(AllRunTimes60,AllIterations60);
% writematrix(AllRunTimes60,"results/optimizer/AllRunTimes60.csv")
% writematrix(AllIterations60,"results/optimizer/AllIterations60.csv")

[AllRunTimes50,AllIterations50] = timingMarkowitz(mu,sigma50,'Sparse50',maxLambdaSparse(1));
stat50 = getOptimizerStats(AllRunTimes50,AllIterations50);
% writematrix(AllRunTimes50,"results/optimizer/AllRunTimes50.csv")
% writematrix(AllIterations50,"results/optimizer/AllIterations50.csv")




[AllRunTimesFull_neg,AllIterationsFull_neg] =  timingMarkowitz(mu_neg,sigma_neg,'Full_neg',maxLambda_neg);
statFull_neg = getOptimizerStats(AllRunTimesFull_neg,AllIterationsFull_neg);
% writematrix(AllRunTimesFull_neg,"results/optimizer/AllRunTimesFull_neg.csv")
% writematrix(AllIterationsFull_neg,"results/optimizer/AllIterationsFull_neg.csv")

[AllRunTimes99_neg,AllIterations99_neg] =  timingMarkowitz(mu_neg,sigma99_neg,'Sparse99_neg',maxLambdaSparse_neg(6));
stat99_neg = getOptimizerStats(AllRunTimes99_neg,AllIterations99_neg);
% writematrix(AllRunTimes99_neg,"results/optimizer/AllRunTimes99_neg.csv")
% writematrix(AllIterations99_neg,"results/optimizer/AllIterations99_neg.csv")

[AllRunTimes90_neg,AllIterations90_neg] = timingMarkowitz(mu_neg,sigma90_neg,'Sparse90_neg',maxLambdaSparse_neg(5));
stat90_neg = getOptimizerStats(AllRunTimes90_neg,AllIterations90_neg);
% writematrix(AllRunTimes90,"results/optimizer/AllRunTimes90.csv")
% writematrix(AllIterations90,"results/optimizer/AllIterations90.csv")

[AllRunTimes80_neg,AllIterations80_neg] = timingMarkowitz(mu_neg,sigma80_neg,'Sparse80_neg',maxLambdaSparse_neg(4));
stat80_neg = getOptimizerStats(AllRunTimes80_neg,AllIterations80_neg);
% writematrix(AllRunTimes80_neg,"results/optimizer/AllRunTimes80_neg.csv")
% writematrix(AllIterations80_neg,"results/optimizer/AllIterations80_neg.csv")

[AllRunTimes70_neg,AllIterations70_neg] = timingMarkowitz(mu_neg,sigma70_neg,'Sparse70_neg',maxLambdaSparse_neg(3));
stat70_neg = getOptimizerStats(AllRunTimes70_neg,AllIterations70_neg);
% writematrix(AllRunTimes70_neg,"results/optimizer/AllRunTimes70_neg.csv")
% writematrix(AllIterations70_neg,"results/optimizer/AllIterations70_neg.csv")

[AllRunTimes60_neg,AllIterations60_neg] = timingMarkowitz(mu_neg,sigma60_neg,'Sparse60_neg',maxLambdaSparse_neg(2));
stat60_neg = getOptimizerStats(AllRunTimes60_neg,AllIterations60_neg);
% writematrix(AllRunTimes60_neg,"results/optimizer/AllRunTimes60_neg.csv")
% writematrix(AllIterations60_neg,"results/optimizer/AllIterations60_neg.csv")


[AllRunTimes50_neg,AllIterations50_neg] = timingMarkowitz(mu_neg,sigma50_neg,'Sparse50_neg',maxLambdaSparse_neg(1));
stat50_neg = getOptimizerStats(AllRunTimes50_neg,AllIterations50_neg);
% writematrix(AllRunTimes50_neg,"results/optimizer/AllRunTimes50_neg.csv")
% writematrix(AllIterations50_neg,"results/optimizer/AllIterations50_neg.csv")


[stat0,stat50,stat60,stat70,stat80,stat90,stat99] %print out sparsity results 

[stat0, statLP, statLSTM] %print out sparsity results 

[statFull_neg,stat50_neg,stat60_neg,stat70_neg,stat80_neg,stat90_neg,stat99_neg] %print out sparsity results 

%% Functions
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


function [results,metrics] = markowitz(m,sigma,lastDay, sigmaFull,maxLambda)

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
    % gurobi_write(model, 'markowitz5.lp');
    results = gurobi(model,params);
    optRisk(i,:) =  (results.x)'*sigmaFull*(results.x);
    optReturn(i,:) = m'*results.x;
    optObj(i,:) =  optReturn(i,:) - lambda(i)*optRisk(i,:);
    actualReturn(i,:) = lastDay*results.x;
    w(i,:) = results.x';
    numAssets(i,:) = nnz(results.x');
    if results.status ~= 'OPTIMAL'
        results.status
        break
    end
end
totalNumAssets = nnz(sum(w)) ;
metricsData = [totalNumAssets,...
    mean(optObj),median(optObj),...
    mean(optRisk), median(optRisk), mean(optReturn), median(optReturn), mean(actualReturn), median(actualReturn)];

metricsNames = {'TotalNumAssets','ObjMean','ObjMedian','RiskMean','RiskMedian','ReturnMean','ReturnMedian','ActualReturnMean','ActualReturnMedian'};

metrics = array2table(metricsData,'VariableNames', metricsNames);
results = array2table([optObj, optRisk,optReturn,actualReturn, numAssets],...
    'VariableNames', {'Obj','Risk','Return','ActualReturn','NumAssets'});

%Visualizations of portfolio
%
% xvalues = readcell("data/yfinance/tickers8year.csv");
% yvalues = lambda;
% boolean = 1*(w>0);
% h = heatmap(xvalues,yvalues,boolean,'GridVisible','off','ColorbarVisible','off');
%
% figure
% hold on
% plot(optRisk,optReturn,'o-','color',[0.4660 0.6740 0.1880],'linewidth',2)
% xlabel('Risk','fontsize',16)
% ylabel('Return','fontsize',16)
% title('Efficient Frontier','fontsize',20)

end

function getConfusionMat(ytest,ypredLSTM,ypredLP)
YPredLSTM = categorical(ypredLSTM);
YPredLP = categorical(ypredLP);
YTest = categorical(ytest);

figure
plotconfusion(YTest,YPredLSTM)
xlabel("Actual Class",'fontsize',16)
ylabel("Predicted Class",'fontsize',16)
title("LSTM Confusion Matrix",'fontsize',20)

figure
plotconfusion(YTest,YPredLP)
xlabel("Actual Class",'fontsize',16)
ylabel("Predicted Class",'fontsize',16)
title("LP Confusion Matrix",'fontsize',20)
end


function [AllRunTimes,AllIterations] = timingMarkowitz(m,sigma,sparseLevel,maxLambda)
lambda = unique([linspace(0,1,121),linspace(1,2,61),linspace(2,4,51), linspace(4,7,61),linspace(7,20,61),linspace(21,60,81),linspace(60,100,51), linspace(100,200,71) linspace(200,maxLambda,81)]);

portfolioLength = length(lambda);
totalRuns = 20;
AllRunTimes = zeros(totalRuns,portfolioLength);
AllIterations = zeros(totalRuns,portfolioLength);
params = struct;
for j = 1:totalRuns
    for i = 1:length(lambda)
        model.Q = -lambda(i)*sparse(sigma);
        model.obj = m';
        model.A = sparse(ones(length(sigma),1)');
        model.rhs = 1;
        model.sense = '=';
        model.modelsense = 'max';
        model.lb  = zeros(length(sigma),1);
        model.ub  = ones(length(sigma),1);
        formatSpec = 'results/optimizer/logs/%s_%d.txt';
        str = sprintf(formatSpec,sparseLevel,j);
        params.LogFile = str;
        if i>1
            model.vbasis = results.vbasis;
            model.cbasis = results.cbasis;
        end
        params.method = 0; % Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
        %         gurobi_write(model, 'markowitz.lp');
        results = gurobi(model,params);
        runtime(i,:) = results.runtime;
        itercount(i,:) = results.itercount;
    end
    AllRunTimes(j,:) = runtime;
    AllIterations(j,:) = itercount;
end
end

function out = getOptimizerStats(Runtime,Iterations)
totalIter = sum(Iterations(1,:),2);
avgIter = mean(Iterations,'all');
avgRun = mean(Runtime,'all');
RPI = sum(Runtime,2)./sum(Iterations,2);
avgRPI = mean(RPI);
out = [totalIter;
    avgIter;
    avgRun;
    avgRPI];
end

