function [f,g] = LogReg(x,A,b,N,mu)
    tmp = A*x;
    expterm = exp(-tmp.*b);
    denterm = 1+expterm;
    f = sum(log(denterm))/N + 0.5*mu*norm(x)^2;
    if nargout>1
        tmp = (expterm.*(-b))./denterm;
        g = transpose(tmp)*A;
        g = transpose(g)/N;
        g = g + mu*x;
    end
end    