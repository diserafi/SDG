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

function [x_new, num_f, alpha] = ...
    linesearch_minunc(Fhandle, x, f, g, d, MaxLSIt, alpha0, LSType, LSInterp)

%==========================================================================
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
% This function implements backtracking line search for unconstrained
% optimization algorithms. The user can choose between Armijo and Wolfe
% line searches and between different step length updating strategies.
% See the INPUT ARGUMENTS for further details.
% 
%==========================================================================
%
% INPUT ARGUMENTS
% 
% Fhandle = function handle, function to be minimized; Fhandle should return
%           at least 2 arguments, namely the function value and the gradient
%           value;
% x       = double array, current iterate;
% f       = double, current function value;
% g       = double array, current gradient;
% d       = double array, search direction;
% MaxLSIt = integer, maximum number of line-search iterations;
% alpha0  = double, starting step length;
% LSType  = string, line search rule to be used:
%            'Armijo' : Armijo line search rule,
%            'Wolfe' :  Wolfe line search rule;
% LSInterp = string, interpolation rule for the line-search, for details
%            see Section 3.5 in [Nocedal and Wright, 2006]: 
%            'Quadratic' : a quadratic interpolation model is used,
%            'Cubic' :     a quadratic model is used in the first LS step,
%                          then the algorithm switches to a cubic
%                          interpolation model,
%            'None'  :     no interpolation is used, the step length is
%                          reduced by a factor of 0.5.
% 
% OUTPUT ARGUMENTS
% 
% x_new   = double array, new iterate;
% num_f   = integer, number of calls to Fhandle;
% alpha   = double, final step length.
%
%==========================================================================
% 
% NOTE: This code is a modification of the M-file 'linesearch3' included in
% the 'mchol' library by Haw-ren Fang and Dianne O'Leary, available from
% 'https://github.com/hrfang/mchol' under GNU Lesser GPL v2.1 License.
% 
%==========================================================================


if ~exist('LSType','var') || isempty(LSType)
    LSType = 'Armijo';      % 'Wolfe'
end
if ~exist('LSInterp','var') || isempty(LSInterp)
    LSInterp = 'Quadratic'; % 'None';% 'Cubic';
end

assert(MaxLSIt>=1);

% line search parameters
sigma1 = 1e-4;
if strcmp(LSType,'Wolfe')
    sigma2 = 0.90;
end
tau_low  = 0.1;
tau_high = 0.5;

% first step
if ~exist('lam0','var') || isempty(alpha0)
    alpha = 1;  % initial full step length is typical for Newton-type methods (Newton, quasi-Newton, modified Newton)
else
    alpha = alpha0;
end
% set up parameters for the first (quadratic) model/approximation
q0   = f;
qp0  = g'*d;   % must be negative (i.e. d must be a descent direction), otherwise it doesn't work
if (qp0 >= 0)
    x_new = x;
    num_f = 0;
    alpha = 0;
    return     % make sure d is a descent direction
end

x_new   = x + alpha*d;
switch LSType
    case 'Armijo'
        f_new   = Fhandle(x_new);
    case 'Wolfe'
        [f_new,g_new]   = Fhandle(x_new);
        slopet = g_new'*d;
end
num_f = 1;  % number of function calls

qc   = f_new;

fgoal = f + sigma1*alpha*qp0;
if strcmp(LSType,'Wolfe')
    ggoal = sigma2*alpha*qp0;
end

switch LSType
    case 'Armijo'
        LSCheck = (f_new > fgoal);
    case 'Wolfe'
        LSCheck = (f_new > fgoal || slopet<ggoal);
end

while LSCheck && (num_f <= MaxLSIt)
    % ft <= fgoal is the sufficient decrease condition, known as Armijo rule
    if strcmp(LSInterp,'None')
        alpha_new = alpha*tau_high;
    elseif strcmp(LSInterp,'Cubic') && num_f > 1 % 1st iteration, quadratic
        alpha_new = cubic_step(q0, qp0, alpha, qc, alpha_old, qm);
    else
        alpha_new = quadratic_step(q0, qp0, alpha, qc);
    end
    
    % project alpha_new to the interval alpha*[tau_low,tau_high];
    alpha_new = min(alpha*tau_high,max(alpha_new,alpha*tau_low));
    
    % lam now is the newly estimated step length
    qm = qc;
    alpha_old = alpha;
    alpha = alpha_new;
    x_new = x + alpha*d;
    switch LSType
        case 'Armijo'
            f_new   = Fhandle(x_new);
        case 'Wolfe'
            [f_new,g_new]   = Fhandle(x_new);
            slopet = g_new'*d;
    end
    num_f = num_f+1;
    qc = f_new;
    fgoal = f + sigma1*alpha*qp0;
    if strcmp(LSType,'Wolfe')
        ggoal = sigma2*alpha*qp0;
    end
    switch LSType
        case 'Armijo'
            LSCheck = (f_new > fgoal);
        case 'Wolfe'
            LSCheck = (f_new > fgoal || slopet<ggoal);
    end    
end
end
% ======================= END OF THE MAIN FUNCTION ========================


% ================= AUXILIARY FUNCTIONS FOR INTERPOLATION =================
% 
function lambda = quadratic_step(q0, qp0, lamc, qc)
%
%   Quadratic polynomial line-search step.
%
%   Usage: lambda = quadratic_step(q0, qp0, qc, tau_low, tau_high)
%
%   This function finds the minimizer of the quadratic polynomial q(lam)
%   that satisfies
%
%   q(0)=q0, q'(0)=qp0, q(lamc)=qc
%
%   Note that this search step is performed because of the the failure of
%   the sufficient decrease condition, which guarantees that the quadratic
%   polynomial is convex provided that qp0 < 0 (i.e. the search direction
%   is a descent direction) and lamc > 0.

    den = 2*(qc-q0-qp0*lamc);
    lambda = -qp0*lamc*lamc / den;
end

function lambda = cubic_step(q0, qp0, lamc, qc, lamm, qm)
%
%   Cubic polynomial line-search step.
%
%   Usage: lambda = cubic_step(q0, qp0, qc, lamm, qm)
%
%   This routine finds the minimizer of the cubic polynomial q(lam) for
%   lam > 0, with q satisfying
%
%   q(0)=q0, q'(0)=qp0, q(lamc)=qc, q(lamm)=qm

    A = [ lamc^2, lamc^3; lamm^2, lamm^3 ];
    b = [ qc-q0-qp0*lamc; qm-q0-qp0*lamm ];
    c = A\b;
    lambda = (-c(1) + sqrt(c(1)*c(1)-3*c(2)*qp0)) / (3*c(2));
end