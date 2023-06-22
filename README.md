# <ins>P</ins>ortfolio <ins>O</ins>ptimization via <ins>D</ins>imension Reduction and <ins>S</ins>parsification (PODS)

Abstract:

*The Markowitz mean-variance portfolio optimization model aims to balance expected return and risk when investing. However, there is a significant limitation when solving large portfolio optimization problems efficiently: the large and dense covariance matrix. Since portfolio performance can be potentially improved by considering a wider range of investments, it is imperative to be able to solve large portfolio optimization problems efficiently, typically in microseconds. We propose dimension reduction and increased sparsity as remedies for the covariance matrix. The size reduction is based on predictions from machine learning techniques and the solution to a linear programming problem. We find that using the efficient frontier from the linear formulation is much better at predicting the assets on the Markowitz efficient frontier, compared to the predictions from neural networks. Reducing the covariance matrix based on these predictions decreases both runtime and total iterations. We also present a technique to sparsify the covariance matrix such that it preserves positive semi-definiteness, which improves runtime per iteration. The methods we discuss all achieved similar portfolio expected risk and return as we would obtain from  a full dense covariance matrix but with improved optimizer performance.*

---

### Contents: 
#### Code
- **part0_dataCollection.ipynb:** Pulling Yahoo! Finance stock data for desired dates.
- **part1_preparingData.m:** Split data into testing and training sets. The target vectors are obtained by solving the Markowitz model.
- **part2a_sparsifySigma.m:** Sparsification by correlation 
- **part2b_reducedLSTM:** Getting predictions from the neural network. Reducing test data based on predictions. 
- **weightedClassificationLayer:** Layer in LSTM network from Part2a
- **part2c_reducedLP.py:** Solving the LP formulation of the Markowitz model with the simplex method. Reducing test data based on predictions. 
- **part3_portfolioPerformance:**  Numerical testing based on portfolios and optimizer performance. 

#### Data
- **data/yfinance:** Data returned from Part0
- **data/full:** Data returned from Part1
- **data/sparse:** Data returned from Part2a
- **data/reduced_LSTM:** Data returned from Part2b
- **data/reduced_LP:** Data returned from Part2c

#### Results
- **results/optimizer:** Runtime and iterations for each problem on all 20 instances
- **results/optimizer/logs:** Logs from optimizer results
- **results/portfolios:** Runtime and iterations for each problem on all 20 instances**

---
