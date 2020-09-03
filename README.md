# SDG

## A MATLAB package implementing the Steepest-Descent Globalized (SDG) method for the solution of unconstrained optimization problems

## Authors
Daniela di Serafino, University of Campania "Luigi Vanvitelli", Caserta, Italy, daniela [dot] diserafino [at] unicampania [dot] it    
Gerardo Toraldo, University of Naples Federico II, Napoli, Italy, toraldo [at] unina [dot] it     
Marco Viola, University of Campania "Luigi Vanvitelli", Caserta, Italy, marco [dot] viola [at] unicampania [dot] it     

## Last Update
Version 1.0 - August 28, 2020

## Description
SDG is a MATLAB implementation of the Steepest-Descent Globalized (SDG)
method for the solution of unconstrained optimization problems of the form

              min  f(x)

with `f` being at least continuously differentiable. The main idea of SDG is
to combine Newton-type directions with scaled steepest-descent steps, to
obtain at each iteration a descent direction `d` satisfying

              -d^T g >= Epsilon * norm(d) * norm(g),

where `g` is the gradient of `f` at the current iterate and Epsilon is a given
threshold. The descent direction has the form

              d = beta*d_N - (1-beta)*xi*g,

where `d_N` may be a Newton, BFGS or LBFGS direction, `xi` is a step length for
the gradient direction (e.g. a Barzilai-Borwein-type step length), and `beta`
is a scalar value in [0,1]. See [1] for further details.

### References
[1] D. di Serafino, G. Toraldo and M. Viola,
*Using gradient directions to get global convergence of Newton-type methods*,
Applied Mathematics and Computation, article 125612, 2020, DOI: 10.1016/j.amc.2020.125612.
Preprint available from [ArXiv](https://arxiv.org/abs/2004.00968) and [Optimization Online](http://www.optimization-online.org/DB_HTML/2020/04/7717.html).

## Software requirements
SDG runs under MATLAB. It has been tested under MATLAB 2018b.

## Contents of the package
Here's the list of SDG files.

MAIN FILES:

- `sdg.m`               : main function;
- `linesearch_minunc.m` : function implementing unconstrained line search.

See the documentation inside each file for further details.

## Example of use
- `demo1.m`     : example of use of SDG on the Brown badly-scaled function;
- `demo2.m`     : example of use of SDG for training a linear classifier;

Subfolder `Demo_files`
- `brown.m`     : Brown badly-scaled function;
- `logreg.m`    : regularized logistic regression function;
- `cod-rna.mat` : training data for the cod-rna dataset.

## License
[![GNU GPL v3.0](http://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl.html)
