import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform 
import scipy.stats as stats
from scipy.special import gamma, factorial

### MCMC ASSIGNMENT 1 ###
#Create an MCMC that draws from Student-t distribution

#The Student-t takes an input of n samples from a normal distribution
#lets create those samples with a mean of 1 and sig of 1
samples = np.random.normal(1, 1, 20)

#Now create our likelihood function
def loglike(x, mu, sig, nu):
    pi = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig)) * (1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
    return(np.sum(np.log(pi)))