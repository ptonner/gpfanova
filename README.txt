# Gaussian Process Functional ANOVA

Implementation of a functional ANOVA (FANOVA) model, based partly on the model in
 [Bayesian functional ANOVA modeling using Gaussian process prior distributions](http://projecteuclid.org/euclid.ba/1340369795). To implement a
 FANOVA model, an underlying general framework is defined for modeling functional
 observations:

 $$ Y(t) = X \beta(t),$$

 where
 $$ Y(t) = [y_1(t),\dots,y_m(t)]^T, $$
 $$\beta(t) = [\beta_1(t),\dots,\beta_f(t)]^T,$$
 $$ X: m \times f$$
 for a given time $t$. The design matrix $X$ defines the relation between the functions $\beta$ and observations $y$. In general, the rank of $X$ should match the number of functions $f$. The FANOVA model can then be described by a specific form of $X$ such that

 $$ y_{i,j}(t) = \mu(t) + \alpha_i(t) + \beta_j(t) + \alpha\beta_{i,j}(t). $$
