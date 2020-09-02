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
% unconstrained optimization problems.
% We consider the Brown badly-scaled function introduced in [Moré, Garbow,
% Hillstrom, ACM Trans. Math. Softw., 7(1), 1981] and minimize it by
% using globalized versions of the Newton, BFGS, and LBFGS methods.
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

n     = 2;               % problem size
x_sol = [1.e6; 2.e-6];   % problem solution
x0    = ones(n,1);       % starting point

%% TEST 1 - Newton's method

fprintf("\nTEST 1 - Globalized Newton's method\n");

% Setting options struct
options = struct('HessianFcn',true,'DirType','Newton',... % using globalized Newton's method
    'LSType','Armijo','LSInterp','Cubic',...              % using Armijo line search with quadratic/cubic interpolation
    'AbsTol',1e-6,'RelTol',0,...                          % setting tolerances for gradient norm
    'Verbose',2);                                         % printing information at each step

% Running SDG
[x_n, f_n, g_n, flag_n, otherinfo_n] = SDG(@(w)brown(w),x0,options);
fprintf('Rel. solution error: %.2e\n',norm(x_n-x_sol)/norm(x_sol));

fprintf ('\n-----------------------------------------------------------------------');

%% TEST 2 - BFGS method
fprintf("\n\nTEST 2 - Globalized BFGS method\n");

% Setting options struct
options = struct('DirType','BFGS',...        % using globalized BFGS method
    'LSType','Armijo','LSInterp','Cubic',... % using Armijo line search with quadratic/cubic interpolation
    'AbsTol',1e-6,'RelTol',0,...             % setting tolerances for gradient norm
    'Verbose',2);                            % printing information at each step

% Running SDG
[x_b, f_b, g_b, flag_b, otherinfo_b] = SDG(@(w)brown(w),x0,options);
fprintf('Rel. solution error: %.2e\n',norm(x_b-x_sol)/norm(x_sol));

fprintf ('\n-----------------------------------------------------------------------');

%% TEST 3 - LBFGS method
fprintf("\n\nTEST 3 - Globalized L-BFGS method\n");

% Setting options struct
options = struct('DirType','LBFGS',...       % using globalized L-BFGS method
    'LSType','Armijo','LSInterp','Cubic',... % using Armijo line search with quadratic/cubic interpolation
    'AbsTol',1e-6,'RelTol',0,...             % setting tolerances for gradient norm
    'Verbose',2);                            % printing information at each step

% Running SDG
[x_lb, f_lb, g_lb, flag_lb, otherinfo_lb] = SDG(@(w)brown(w),x0,options);
fprintf('Rel. solution error: %.2e\n',norm(x_lb-x_sol)/norm(x_sol));
