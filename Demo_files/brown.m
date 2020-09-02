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

function [f,g,H] = brown(x)

%==========================================================================
%
% Authors:
%  Daniela di Serafino (daniela [dot] diserafino [at] unicampania [dot] it)
%  Gerardo Toraldo     (toraldo                  [at] unina [dot] it      )
%  Marco Viola         (marco [dot] viola        [at] unicampania [dot] it)
%
% Version: 1.0
% Last Update: 28 August 2020
%
%==========================================================================
%
% This function implements the Brown badly-scaled function introduced in
% [Moré, Garbow, Hillstrom, ACM Trans. Math. Softw., 7(1), 1981].
% 
%==========================================================================
%
% INPUT ARGUMENTS
% 
% x       = double array, current point;
% 
% OUTPUT ARGUMENTS
% 
% f       = double, function value;
% g       = double array, gradient;
% H       = double matrix, Hessian matrix;
%
%==========================================================================
% 
% NOTE: This code is a modification of the M-files related to Problem 10 of
% the problem collection by John Burkardt, available at
% 'https://people.sc.fsu.edu/~jburkardt/m_src/test_opt/test_opt.html'
% under GNU Lesser GPL v3 License.
% 
%==========================================================================

    f = ( x(1) - 1.e+6 )^2 + ( x(2) - 2.e-6 )^2 + ( x(1) * x(2) - 2.0 )^2;
    
    if nargout > 1
        g = zeros ( 2, 1 );
        g(1) = 2.0 * x(1) - 2.e+6 + 2.0 * x(1) * x(2) * x(2) - 4.0 * x(2);
        g(2) = 2.0 * x(2) - 4.e-6 + 2.0 * x(1) * x(1) * x(2) - 4.0 * x(1);
    end
    if nargout > 2
        H = zeros(2,2);
        H(1,1) = 2.0 + 2.0 * x(2) * x(2);
        H(1,2) = 4.0 * x(1) * x(2) - 4.0;
        H(2,1) = 4.0 * x(1) * x(2) - 4.0;
        H(2,2) = 2.0 + 2.0 * x(1) * x(1);
    end
    
end

