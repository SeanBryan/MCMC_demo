import emcee
import numpy as nm
import pylab
import gen_linfit


# make the fake data, y = mx + b
xdat = nm.array([1.0,2.0,3.0,4.0,5.0,6.0])
actual_m = 2.1
actual_b = 1.3
realparams = nm.array([actual_b,actual_m])
noiselevel = 0.5
ynoise = nm.ones(len(xdat))*noiselevel
Y = [lambda x: 1.0,    lambda x: x]
ydat = gen_linfit.calc_model(xdat,realparams,Y) + nm.random.randn(len(xdat))*noiselevel


##### do linear fitting, the superfast matrix inversion way    #####
linear_params,linear_params_cov = gen_linfit.fit(xdat,ydat,ynoise,Y)
linear_best_fit_y = gen_linfit.calc_model(xdat,linear_params,Y)
##### END do linear fitting, the superfast matrix inversion way #####


##### do the fit the emcee mcmc way     #####
# define the likelihood function
def lnprob(params, xdat, ydat, ynoise, Y):
	# calculate the model here
	model = gen_linfit.calc_model(xdat,params,Y)
	# calculate chi squared
	chi2 = nm.sum(((model - ydat)*(model - ydat)) / (ynoise*ynoise))
	# calculate log likelihood and return
	return -0.5*chi2

# set up the number of walkers and initialze them
nwalkers = 40
ndim = 2 # it's a two-parameter fit
# make the initial distribution a gaussian with width 1/2, and initial mean equalish to the input parameters
p0 = 0.5*nm.random.rand(ndim * nwalkers).reshape((nwalkers, ndim)) + nm.concatenate((2.0*nm.ones([nwalkers,1]),1.0*nm.ones([nwalkers,1])),axis=1)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[xdat, ydat, ynoise, Y])

# run for 100 steps to burn in
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()

# after the burn in, run 1000 steps
sampler.run_mcmc(pos, 1000)

# calculate the mean parameters and their covariance matrix
mcmc_params = nm.mean(sampler.flatchain,axis=0)
mcmc_params_cov = nm.cov(nm.transpose(sampler.flatchain))

# calculate the model
mcmc_best_fit_y = gen_linfit.calc_model(xdat,mcmc_params,Y)
##### END do the fit the emcee mcmc way #####

# plot the results
pylab.figure(1)
pylab.errorbar(xdat,ydat,yerr=noiselevel,fmt='*',label='Data')
pylab.plot(xdat,linear_best_fit_y, \
	       label='Linear Fit:\nm = ' + str(round(linear_params[1],4)) + ' +/- ' + str(round(nm.sqrt(linear_params_cov[1,1]),4)) + \
	       '\nb = ' + str(round(linear_params[0],4)) + ' +/- ' + str(round(nm.sqrt(linear_params_cov[0,0]),4)) + \
	       '\ncovariance = ' + str(round(linear_params_cov[0,1],4)))
pylab.plot(xdat,mcmc_best_fit_y, \
	       label='MCMC Fit:\nm = ' + str(round(mcmc_params[1],4)) + ' +/- ' + str(round(nm.sqrt(mcmc_params_cov[1,1]),4)) + \
	       '\nb = ' + str(round(mcmc_params[0],4)) + ' +/- ' + str(round(nm.sqrt(mcmc_params_cov[0,0]),4)) + \
	       '\ncovariance = ' + str(round(mcmc_params_cov[0,1],4)))
pylab.xlim([0.5,6.5])
pylab.legend(loc='upper left')

## this is already generated in the triangle plot
## but, just to illustrate how to access the raw merged chain
## plot histograms of the two parameters from the MCMC way
#pylab.figure(4)
#pylab.subplot(211)
#pylab.hist(sampler.flatchain[:,0],color='k',histtype='step',bins=50)
#pylab.title('b parameter')
#pylab.subplot(212)
#pylab.hist(sampler.flatchain[:,1],color='k',histtype='step',bins=50)
#pylab.title('m parameter')

# make a triangle plot
import triangle
f1 = triangle.corner(sampler.flatchain,labels=[r'$b$',r'$m$'], \
	quantiles=[(1-0.9973)/2.0,(1-0.9545)/2.0,(1-0.6827)/2.0, 0.6827 + (1-0.6827)/2.0 ,0.9545 + (1-0.9545)/2.0 ,0.9973 + (1-0.9973)/2.0],show_titles=True)

# plot the chains walking around
pylab.figure(3)
for i in range(sampler.chain.shape[0]):
	pylab.plot(sampler.chain[i,:,0],sampler.chain[i,:,1])
pylab.xlabel('b')
pylab.ylabel('m')
pylab.title('Path of the Individual Chains')

pylab.ion()
pylab.show()