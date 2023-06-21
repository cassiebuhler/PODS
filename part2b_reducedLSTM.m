clear all; close all; clc;
trainReturns = readmatrix("data/full/returnsTrain_standardized.csv");
testReturns = readmatrix("data/full/returnsTest_standardized.csv");

Ytrain = readmatrix("data/full/ytrain.csv");
Ytest = readmatrix("data/full/ytest.csv");
YTrain = categorical(double(Ytrain))';
YTest = categorical(double(Ytest))';

n = length(YTrain);


% need to get returns in this format for LSTM network 
for i = 1:n
    XTrain{i,:} = [trainReturns(:,i)'];
    XTest{i,:} = [testReturns(:,i)';];
end


prior0 = sum(double(YTrain)-1 ==0)/numel(double(YTrain)); % num class 0 
prior1 = sum(double(YTrain)-1 ==1)/numel(double(YTrain)); % num class 1

% just checking the distribution of test data , not using in training 
prior0test = sum(double(YTest)-1 ==0)/numel(double(YTest)); % num class 0 
prior1test = sum(double(YTest)-1 ==1)/numel(double(YTest)); % num class 1


%% Parameters
rng(0)

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',60, ...
    'MiniBatchSize',n/2,...
    'InitialLearnRate',0.00025, ...
    'L2regularization',0.00001,... 
    'Verbose',0, ...
    'Plots','training-progress');


numClasses = 2;
numFeatures = 1;

classWeights = [prior1 prior0];
 
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(150,'OutputMode','last') %150
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% network
[net,info] = trainNetwork(XTrain,YTrain,layers,options);

%% prediction
YPred = classify(net,XTest);
% sum(double(ypred)-1 ==1)
% sum(double(ypred)-1 ==1)/numel(double(ypred))

figure
plotconfusion(YTest,YPred') %confusion matrix

ypred = double(YPred)-1; %converting back to numeric 

writematrix(ypred,"data/reduced_LSTM/ypred.csv")

%% Reducing matrix with prediction

%read in full data
mu= readmatrix("data/full/muTest.csv");
sigma = readmatrix("data/full/sigmaTest.csv");
lastDay = readmatrix("data/yfinance/lastDayReturns8year.csv");
lastDay = lastDay(2:end);
tickers = readmatrix("data/yfinance/tickers8year.csv");

%reduce 
muLSTM = mu(ypred == 1);
sigmaLSTM = sigma(ypred == 1,ypred == 1);
lastDayLSTM = lastDay(ypred == 1);
tickersLSTM = tickers(ypred == 1);

writematrix(muLSTM,"data/reduced_LSTM/muTest.csv");
writematrix(sigmaLSTM,"data/reduced_LSTM/sigmaTest.csv");
writematrix(lastDayLSTM,"data/reduced_LSTM/lastDayReturnsLSTM.csv");
writematrix(tickersLSTM,"data/reduced_LSTM/tickersLSTM.csv");



