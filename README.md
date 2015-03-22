# MCMC Demo
Code demonstrating Dan Foreman-Mackey's emcee code for MCMC fitting, and comparison with a generalized linear fitting that I wrote.

# Introduction

Dan Foreman-Mackey documented the emcee package for python really well, here I'm presenting my own first use case for this useful tool that allows any model to be fit to any dataset in a very precise way with reliable error bars on the derived parameters. His version of what I'm presenting is [here](http://dan.iel.fm/emcee/current/user/line/).

A more theoretical introduction to the MCMC concept can be found in [Chib and Greenberg, "Understanding the Metropolis-Hastings Algorithm"](http://streaming.stat.iastate.edu/~stat444x_B/Literature/ChibGreenberg.pdf). For generalized linear fitting, I am following Garcia's book called "Numerical Methods for Physics."

This assumes you have ipython, numpy, matplotlib, and the emcee and triangle_plot packages installed.

# Linear Fitting

Linear fitting is the case where the model is linear in the parameters. So, these two models relating y and x are linear in their parameters:

y = mx + b

y = a + b*cos(x) + c*sin(x)

And this model is not linear in the parameters a or b, since they appear inside the trigonometric function:

y = cos(a*x + b)

So, for linear fitting, there exists an extremely fast matrix-based algorithm to estimate both the parameters and the error bars from a dataset, assuming the errors on the data itself are gaussian. This is discussed many places, including Garcia's book. I implemented the equations from that book in python in the module gen_linfit above. Also, even though it can be used for non-linear models, the emcee module can be used for linear models as well, to check the two methods against each other.

The demo code assumes some ground truth for the parameters m and b, and selects some x values. It generates the exact y values, and adds gaussian noise to each of them to generate a fake data set. After starting ipython and typing:

```python
%run emcee_demo_linear.py
```

the two figures below should appear. The exact values may change slightly, since new random numbers are generated at each run, but they should be very similar.

![Best Fit](emcee_demo_bestfit.png?raw=true)
![Best Fit](emcee_demo_triangleplot.png?raw=true)

The figure caption in the top panel shows that the estimated parameters and parameter covariance agree between the two methods. For this case of a tiny data set, both methods ran fairly quickly, but for a large dataset the matrix method would be much faster and therefore preferred if the model is linear.

The emcee method is able to handle non-linear models because it uses several "walkers" to "walk" among different values of the parameters m and b, evaluating the likelihood of the parameters at each step it takes. It chooses its steps randomly, but weights the steps it takes to spend more time at favorable parameters than unfavorable ones. At the end of this day, this property causes the set of parameter values the algorithm stepped through to have the same statistical properties as the posterior distribution of the parameters themselves. This allows best-fit and confidence intervals to be extracted.
