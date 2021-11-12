function [intrans,rho,epsilon,r_trans] = Estimate_rho_Gaussian_traits(g,H,Sigma,V,E,delta)
%% Estimates the correlation coefficient rho, relative intransitivity, and epsilon
% based on a quadratic model with gradient g and Hessian H for normally
% distributed traits

%% Inputs:
% 1. g: the gradient in performance at the expected traits (column vector)
% 2. H: a struct with H.xx and H.xy, H.xx is on-diagonal block of Hessian,
% H.xy is off-diagonal block
% 3. Sigma: the covariance in the trait distribution
% 4. V: the number of competitors (vertices)
% 5. E: the number of edges in the graph
% 6. delta: a tolerance defining an upper bound on epsilon beneath which
% competition is "locally" transitive

%% Outputs
% 1. intrans: the expected relative intransitivity defined as
% sqrt(E[||F_c||^2]/E[||F||^2]])
% 2. rho: the correlation coefficient
% 3. epsilon: the small quantity such that rho = 1/(2(1 + epsilon))
% 4. r_trans: an upper bound on the largest std such that a Gaussian
% distribution with covariance proportional to Sigma would be locally
% transitive (epsilon less than delta) (where std is defined as
% sqrt(trace(Sigma)/T)

%% get dimensions
T = length(g);
L = E - (V - 1);

%% normalize
s_true = sqrt(trace(Sigma)/T);
S = T*Sigma/trace(Sigma);

%% define function to compute epsilon
a = trace(-H.xy*S*H.xy*S);
b = 2*(g'*S*g);
c = trace(H.xx*S*H.xx*S);
epsilons = @(s) (a*s.^2)./(b + c*s.^2);

%% compute epsilon, rho, and intrans
epsilon = epsilons(s_true);
rho = 1./(2*(1 + epsilon));
intrans = sqrt((1 - 2*rho)*(L/E));

%% compute r_trans (where epsilon = delta)
if a > delta*c
    r_trans = sqrt(delta*b/(a - delta*c));
else
    r_trans = inf;
end
