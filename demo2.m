% -------------------------------------------------------------------------
% Copyright (C) 2020 by D. di Serafino, G. Toraldo, M. Viola.
%
%                           COPYRIGHT NOTIFICATION
%
% This file is part of SDG.
%
% SDG is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% SDG is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with SDG. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------
% 
% Authors:
%  Daniela di Serafino (daniela [dot] diserafino [at] unicampania [dot] it)
%  Gerardo Toraldo     (toraldo                  [at] unina       [dot] it)
%  Marco Viola         (marco [dot] viola        [at] unicampania [dot] it)
%
% Version: 1.0
% Last Update: 28 August 2020
%
%==========================================================================
% 
% DESCRIPTION:
% This is an example of use of the SDG method [1] for the solution of
% unconstrained optimization problems coming from the training of linear
% classifiers via a regularized logistic regression model.
% We consider the "cod-rna" dataset [Uzilov et al., BMC Bioinformatics,
% 7(173), 2006], available from the LIBSVM website at
% 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html'.
% The training set was downloaded and converted to the MATLAB format thanks
% to the MATLAB interface included in the LIBSVM package (available from
% 'https://www.csie.ntu.edu.tw/~cjlin/libsvm/' under Modified BSD License).
% The script performs a 10-fold cross validation with a single value of the
% regularization parameter.
% The solution of the training model is computed with SDG (see [1])
% equipped with the BFGS method.
% 
% NOTE: We precomputed the cross-validation folders with the 'crossvalind'
% function included in the Bioinformatics Toolbox, so that the script is
% independent from specific toolboxes.
% 
%==========================================================================
%
% REFERENCES:
% [1] D. di Serafino, G. Toraldo and M. Viola,
%     "Using gradient directions to get global convergence of Newton-type 
%      methods", Applied Mathematics and Computation, article 125612, 2020.
%      DOI: 10.1016/j.amc.2020.125612
%
% Preprint available from ArXiv
%     https://arxiv.org/abs/2004.00968
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2020/04/7717.html
% 
%==========================================================================

addpath('./Demo_files/');
fprintf ('10-fold cross validation test on the cod-rna dataset\n');

fprintf ('=====================================================================\n');

%% Loading data and cross-validation indices
load('cod-rna.mat');
acc_vect = zeros(10,1);

% Fixing the regularization parameter to be about 1/#samples for each
% of the 10 folders

mu = 10/length(train_l);  

for foldind = 1:10

    fprintf ('\nTaking out folder %2d\n',foldind);

    % Extracting current fold
    X = train(cvindices~=foldind,:);
    y = train_l(cvindices~=foldind);
    Xtest = train(cvindices==foldind,:);
    ytest = train_l(cvindices==foldind);
    
    % Z-standardization of data
    [X,Xmean,Xvar] = zscore(X,0,1);
    
    featurestokeep = Xvar>eps;
    X = X(:,featurestokeep);
    Xmean = Xmean(featurestokeep);
    Xvar = Xvar(featurestokeep);
    Xtest = Xtest(:,featurestokeep);
    for k = 1:length(Xvar)
        Xtest(:,k) = (Xtest(:,k)-Xmean(k))/Xvar(k);
    end
    
    % Adding extra feature for biased linear classifier
    [m,n] = size(X);    
    X = [X, ones(m,1)];
    Xtest = [Xtest, ones(size(ytest))];
    n = n+1;

    % Setting options struct
    options = struct('DirType','BFGS',...        % using globalized BFGS method
        'LSType','Armijo','LSInterp','Cubic',... % using Armijo line search with quadratic/cubic interpolation
        'AbsTol',0,'RelTol',1e-5,...             % setting relative tolerance on gradient norm
        'Verbose',1);                            % printing final information for each call of SDG

    % Solving training model via SDG
    [x, f, g, flag, otherinfo] = SDG(@(w)logreg(w,X,y,m,mu),zeros(n,1),options);
    
    % Computing test accuracy
    ntest = length(ytest);
    tmp = Xtest*x;
    tmp = (ytest.*tmp)<=0;
    testerr = nnz(tmp)/(ntest);
    acc_vect(foldind) = 1-testerr;
    
    if foldind<10
        fprintf ('---------------------------------------------------------------------\n');
    end
end

fprintf ('\n=====================================================================\n');

fprintf ('\nTraining completed.\n');
fprintf ('Accuracy mean %5.2f\n',mean(acc_vect)*100);
fprintf ('Accuracy std  %5.2f\n',std(acc_vect)*100);
