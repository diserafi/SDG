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

function [x, f, g, flag, outargs] = SDG(Fhandle, x0, options)

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
% This function implements the "Steepest Descent Globalized (SDG) method",
% for the solution of unconstrained optimization problems of the form
%
% (1)                           min  f(x)
%
% with f being at least continuously differentiable. The main idea of
% SDG is to combine Newton-type directions with scaled steepest-descent
% steps, to obtain at each iteration a descent direction d satisfying 
% 
% (2)              -d^T g >= Epsilon * norm(d) * norm(g),
%
% where g is the gradient of f at the current iterate and Epsilon is a
% given threshold. The descent direction has the form
% 
% (3)                   d = beta*d_N - (1-beta)*xi*g,
% 
% where d_N may be a Newton, BFGS or LBFGS direction, xi is a step length
% for the gradient direction (e.g. a Barzilai-Borwein-type step length),
% and beta is a scalar value in [0,1]. See [1] for further details.
% The algorithm generally stops when
%
% (4)             ||g|| <= max ( AbsTol, RelTol*||g_0|| ),
%
% where AbsTol and RelTol are given absolute and relative tolerances,
% respectively, and g_0 is the gradient at the starting point. See the
% output parameter flag for other situations where SDG may stop. 
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
%
% INPUT ARGUMENTS
% 
% Fhandle = function handle, function to be minimized; Fhandle should return
%           at least 2 arguments, namely the function value and the gradient
%           of f(x); Fhandle can optionally return the Hessian as a third
%           output argument in case the Newton direction is selected, if so
%           the option 'HessianFcn' must be set to 'true' (see 'options');
% x0      = double array, starting point;
% options = [OPTIONAL] struct array with the following (possible) entries,
%           to be specified as pairs ('propertyname', propertyvalue);
%           the string 'propertyname' can be:
%           AbsTol  = double, absolute tolerance on the gradient norm
%                     [DEFAULT 1e-5]; 
%           RelTol  = double, relative tolerance on the gradient norm
%                     [DEFAULT 1e-5]; 
%           MaxIt   = integer, maximum number of iterations [DEFAULT 2000];
%           MaxTime = double, maximum time in seconds [DEFAULT 360];
%           Epsilon = double, lower bound on the angle between the descent
%                     direction and the gradient [DEFAULT 0.5];
%           VarEpsilon = double, coefficient in (0,1) regulating the
%                        decrease of Epsilon; in detail, Epsilon is
%                        multiplied by VarEpsilon every time the
%                        Newton-Type iteration is rejected; if set to 0
%                        then Epsilon is kept constant [DEFAULT 0.95];
%           DirType = string, Newton-Type direction to be used:
%                      'Newton': the Newton method is used; the user
%                         needs to provide the Hessian matrix as the third
%                         output argument of Fhandle and set HessianFcn to
%                         true,
%                      'BFGS' : the BFGS method is used; the Hessian
%                         matrix is initialized by following the heuristic
%                         proposed in [Nocedal and Wright, 2006, eq. (6.20)]
%                         [DEFAULT],
%                      'LBFGS' : the Limited-memory BFGS method is used;
%                         the memory can be specified via the LBFGSMem
%                         option;
%           HessianFcn = logical, if true Fhandle returns the Hessian
%                        matrix as the third output variable [DEFAULT false];
%           LBFGSMem = integer, memory of the Limited-memory BFGS method
%                      [DEFAULT 10];
%           SDType  = string, step length to be used for the gradient
%                     direction:
%                      'BB1' :    the BB1 step length is used,
%                      'BB2' :    the BB2 step length is used [DEFAULT],
%                      'ABBmin' : the ABBmin step length with adaptive
%                                 switching rule proposed in
%                                 [Bonettini et al., Inv. Prob. 25(1), 2009],
%                      'Unit' :   the gradient step length is set to 1;
%           LSType  = string, line search rule to be used:
%                      'Armijo' : Armijo line search rule [DEFAULT],
%                      'Wolfe' :  Wolfe line search rule;
%           LSInterp = string, interpolation rule for line-search step length,
%                      for details see Section 3.5 in [Nocedal and Wright, 2006]:
%                      'Quadratic' : a quadratic interpolation models is used,
%                      'Cubic' :  a quadratic model is used in the first LS
%                                 step, then the algorithm switches to the
%                                 use of a cubic interpolation model [DEFAULT],
%                      'None'  :  no interpolation is used, the step length
%                                 is reduced by a factor of 0.5;
%           MaxLSIt = integer, maximum number of line searches per
%                     iteration [DEFAULT 10];
%           Verbose = integer, level of verbosity:
%                      0: no prints [DEFAULT],
%                      1: print information at final step,
%                      2: print information at each step.
% 
% OUTPUT ARGUMENTS
% 
% x       = double array, computed solution;
% f       = double, objective function value at the computed solution;
% g       = double array, gradient of f(x) at the computed solution;
% flag    = integer, information on the execution:
%           0 : SDG found a point satisfying the stopping criterion (4),
%           1 : the stopping criterion (4) was not satisfied, SDG stopped
%               because the number of iterations excedeed MaxIt,
%           2 : the stopping criterion (4) was not satisfied, SDG stopped
%               because the execution time excedeed MaxTime,
%           3 : the stopping criterion (4) was not satisfied, SDG stopped
%               because the relative variation of the objective function
%               was below the given tolerance,
%          -1 : the stopping criterion (4) was not satisfied, SDG stopped
%               because the Hessian matrix was NaN or Inf;
% outargs = struct array, containing further information on the execution,
%           it contains the following entries:
%           it         = integer, number of iterations performed;
%           f_vect     = double array, history of the o.f. value;
%           num_f      = integer, total number of function evaluations;
%           beta_vect  = double array, history of the values of beta used
%                        in (3); 
%           alpha_vect = double array, history of the step lengths obtained
%                        with the selected line search; 
%           tot_time   = elapsed time.
%
%==========================================================================

if nargin < 2 || isempty(Fhandle) 
    error('Please specify at least the objective function handle and the starting point.');
end

%% Initializing parameters
AbsTol = 1e-5;
RelTol = 1e-5;
FunTol = 10*eps;
MaxIt = 2000;
MaxTime = 3600;
Epsilon = 0.5;
AdaptEpsilon = true;
VarEpsilon = 0.95;
DirType = 'BFGS';    % 'Newton'; % 'LBFGS';
HessianFcn = false;
LBFGSMem = 10;       % only useful for LBFGS
SDType = 'BB2';      % 'Unit'; % 'ABBmin'; % 'BB1'; %  
LSType = 'Armijo';   % 'Wolfe'; %  
LSInterp = 'Cubic';  % 'None'; % 'Quadratic';
MaxLSIt = 10;
Verbose = 0;

%% Grabbing personalized settings from options
optionnames = fieldnames(options);
for iter=1:numel(optionnames)
    switch upper(optionnames{iter})
        case 'ABSTOL'
            AbsTol = options.(optionnames{iter});
        case 'RELTOL'
            RelTol = options.(optionnames{iter});
        case 'MAXIT'
            MaxIt = options.(optionnames{iter});
        case 'MAXTIME'
            MaxTime = options.(optionnames{iter});
        case 'EPSILON'
            Epsilon = options.(optionnames{iter});
        case 'VAREPSILON'
            VarEpsilon = options.(optionnames{iter});
            if VarEpsilon == 0
                AdaptEpsilon = false;
            end
        case 'DIRTYPE'
            DirType = options.(optionnames{iter});
        case 'HESSIANFCN'
            HessianFcn = options.(optionnames{iter});
        case 'LBFGSMEM'
            LBFGSMem = options.(optionnames{iter});
        case 'SDTYPE'
            SDType = options.(optionnames{iter});
        case 'LSTYPE'
            LSType = options.(optionnames{iter});
        case 'LSINTERP'
            LSInterp = options.(optionnames{iter});
        case 'VERBOSE'
            Verbose = options.(optionnames{iter});
        otherwise
            error(['Unrecognized option: ''' optionnames{iter} '''']);
    end
end

if ~HessianFcn && strcmp(DirType,'Newton')
    warning("Hessian must be provided for Newton's direction to be used. Switching to BFGS.");
    DirType = 'BFGS';
end


if Verbose
    fprintf('\nUsing the %s method globalized by SD with %s step length.',DirType,SDType)    
end
%% Starting computation
x = x0(:);
f_vect = zeros(MaxIt+1,1);
num_f = 0;
beta_vect = zeros(MaxIt,1);
alpha_vect = zeros(MaxIt,1);
st_time = tic;
st_bb = 0;
flag = 0;

if strcmp(SDType,'ABBmin')
    m = 3;
    tau = 0.5;
    bb2vec = 1e5*ones(m,1);
end
switch DirType
    case 'BFGS'
        H = speye(length(x));
    case 'LBFGS'
        H = struct('S',[],'Y',[],'YtS',[]);
        st_bb = 2;
end

xi = 1;
it = 1;

if Verbose>1
    fprintf('\n   it      function     gnorm   epsilon      beta    cosine     alpha ');
    fprintf('\n______________________________________________________________________');
end

while (1)
    tot_time = toc(st_time);
    if strcmp(DirType,'Newton')
        [f,g,H] = Fhandle(x);
        H = (H+H')/2; % ensuring symmetry of the Hessian matrix
    else
        [f,g] = Fhandle(x);
    end
    num_f = num_f + 1;
    normg = norm(g);
    f_vect(it) = f;
    if Verbose>1
       fprintf('\n%5d  %.6e  %.2e  ',it-1,f,normg); 
    end
    if it == 1
        GradTol = max(AbsTol,RelTol*normg); % computing stopping tolerance
    end
    if (it>MaxIt) 
        flag = 1;
        if Verbose>1
            fprintf('---------------FINISHED---------------\n');
        end
        if Verbose
            fprintf('\nSDG stopped because the maximum number of iteration was reached.');
        end
    elseif (tot_time>MaxTime) 
        flag = 2;        
        if Verbose>1
            fprintf('---------------FINISHED---------------\n');
        end
        if Verbose
            fprintf('\nSDG stopped because the maximum execution time was reached.');
        end
    elseif ((it>1) && (abs(f_vect(end)-f_vect(end-1))/abs(f_vect(end-1)) <= FunTol ) )
        flag = 3;
        if Verbose>1
            fprintf('---------------FINISHED---------------\n');
        end
        if Verbose
            fprintf('\nSDG stopped because the relative function variation was below the tolerance.');
        end
    elseif ~strcmp(DirType,'LBFGS') && (any(any(isnan(H))) || any(any(isinf(H))))
        flag = -1;
        if Verbose>1
            fprintf('---------------FINISHED---------------\n');
        end
        if Verbose
            fprintf('\nSDG stopped because the Hessian matrix contains a NaN or an Inf entry.');
        end
    end
    
    if (normg <= GradTol) || flag
        if (~flag) && (Verbose>1)
            fprintf('---------------FINISHED---------------\n');
        end
        f_vect = f_vect(1:it);
        beta_vect = beta_vect(1:it-1);
        alpha_vect = alpha_vect(1:it-1);
        outargs = struct('it',it-1,'f_vect',f_vect,'num_f',num_f,...
            'beta_vect',beta_vect,'alpha_vect',alpha_vect,'tot_time',tot_time);
        if Verbose
            if flag==0
                fprintf('\nSDG found a point which satisfies the stopping criterion.');
            end
            fprintf('\nf(x)         = %.15e',f);
            fprintf('\nnorm(g)      = %.2e',normg);
            fprintf('\n#iterations  = %d',outargs.it);
            fprintf('\n#func evals  = %d',outargs.num_f);
            fprintf('\nElapsed time = %d seconds\n',tot_time);
        end
        break;
    end
    
% Heuristic initialization of the inverse Hessian approximation in case 
% BFGS is used [see Nocedal and Wright, 2ed, eq. (6.20)]
    if it == 1 && strcmp(DirType,'BFGS') 
        x_temp = x-g/normg;
        [~,g_temp] = Fhandle(x_temp);
        y = g_temp-g;
        s = x_temp-x;
        
        tmp = (s'*y)/(y'*y);
        if tmp>eps
            H = tmp*eye(length(s));
        end
    end
    
    % Computing iterate and gradient differences
    if it>1
        s = x-x_old;
        y = g-g_old;
    end

    switch DirType
        case 'BFGS'
            % Updating inverse Hessian approximation
            % and evaluating BFGS direction
            if it>1 
                H = updateQN(H,s,y,DirType);
            end
            dx = (H*g);
        case 'LBFGS'
            % Updating inverse Hessian approximation
            % and evaluating LBFGS direction
            if it>1
                H = updateQN(H,s,y,DirType,LBFGSMem);
            end
            if length(H.YtS)>2
                dx = compute_lbfgs_prod(H,g);
            else
                dx = g;
            end
        case 'Newton'
            % Evaluating Newton direction
            dx =  (H\g);
    end

    cosNEW = (dx'*g)/(norm(dx)*normg);
    if any(isnan(dx))
        dx = zeros(size(dx));
        cosNEW = 0;
    end

    if Verbose>1
       fprintf('%.2e  ',Epsilon); 
    end

    if (it>st_bb) && (cosNEW>Epsilon)
        dx=-dx;
        beta_vect(it) = 1;
    else        
        if it>1
            temp = s'*y;
            if temp <= 0 && ~strcmp(SDType,'Unit')
                xi = min(xi*10,1e5);
            else
                if strcmp(SDType,'BB1')
                    xi = (s'*s)/temp;
                elseif strcmp(SDType,'BB2')
                    xi = temp/(y'*y);
                elseif strcmp(SDType,'ABBmin')
                    bb2vec(1:m-1) = bb2vec(2:m);
                    BB1 = (s'*s)/temp;
                    BB2 = temp/(y'*y);
                    
                    bb2vec(m) = BB2;
                    
                    if (it <= m)
                        xi = min(bb2vec);
                    elseif (BB2/BB1 < tau)
                        xi = min(bb2vec);
                        tau = tau*0.9;
                    else
                        xi = BB1;
                        tau = tau*1.1;
                    end
                else
                    xi = 1/norm(g);
                end
            end
        else
            xi = 1/norm(g);
        end        
        
        if it>st_bb && cosNEW>0
            rho_m = xi*(Epsilon - 1);
            pi_m = g'*dx/(normg^2) - Epsilon*(norm(dx)/normg);
            beta = rho_m/(rho_m + pi_m);
            beta_vect(it) = beta;
            dx = -beta*dx-(1-beta)*xi*g;
        else
            dx=-xi*g;
            beta_vect(it) = 0;
        end

        % Sice Newton-type step was rejected we decrease the value of Epsilon
        if AdaptEpsilon
            Epsilon = Epsilon*VarEpsilon;
        end

    end
    
    if Verbose>1
        cosdx = -(dx'*g)/(norm(dx)*normg);
        fprintf('%.2e  %.2e  ',beta_vect(it),cosdx); 
    end

    x_old = x;
    g_old = g;
    
    alpha0 = 1;
    [ x, num_f_part, alpha ] = linesearch_minunc(Fhandle,x,f,g,dx,MaxLSIt,alpha0,LSType,LSInterp);
    num_f = num_f + num_f_part;
    alpha_vect(it) = alpha;

    if Verbose>1
        fprintf('%.2e',alpha); 
    end

    it = it+1;
end

end
% ======================= END OF THE MAIN FUNCTION ========================


% ============= AUXILIARY FUNCTIONS FOR QUASI-NEWTON METHODS ==============
% 
function [H,skip] = updateQN(H,s,y,Type,mem)

sty = s'*y;
go_on = (sty >= sqrt(eps)*max(eps,norm(s)*norm(y)));

if go_on
    switch Type
        case 'BFGS'
            Hy = H*y;
            H = H + (1 + y'*Hy/sty) * ...
                (s*s')/sty - (s*Hy' + ...
                Hy*s')/sty;
        case 'LBFGS'
            H.S = [H.S s];
            H.Y = [H.Y y];
            H.YtS = [H.YtS y'*s];
            if(size(H.S,2)>mem)
                H.S(:,1)   = [];
                H.Y(:,1)   = [];
                H.YtS(:,1) = [];
            end
    end
    skip = 0;
else
    skip = 1;
end
end

function Hg = compute_lbfgs_prod(H,g)

mem = size(H.S,2);
p = g;
r = zeros(mem,1);

for i = mem:-1:1
    r(i) = 1/(H.YtS(i));
    a(i) = r(i)*H.S(:,i)'*p;
    p = p - a(i)*H.Y(:,i);
end

gamma_0 = (H.YtS(end))/norm(H.Y(:,end))^2;
q = gamma_0*p;    

for j = 1:mem
    b = r(j)*H.Y(:,j)'*q;
    q = q + H.S(:,j)*(a(j) - b);
end

Hg = q;

end
