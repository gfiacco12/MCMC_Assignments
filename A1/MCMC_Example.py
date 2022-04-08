from platform import java_ver
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform 
import scipy.stats as stats

#Random Gaussian distribution with mu=0 and sigma=1, 100 entries
#we are going to try to simulate this for unknown sigma
N = 1000
data = np.random.normal(0, 6, N)
x = np.linspace(-5, 5, N)
curve = stats.norm(0, 1)
#plt.hist(data, bins=20, density=True)
#plt.plot(x, curve.pdf(x))
#plt.show()


#start by defining all of the functions we will need 
#1. our likelihood for the data will be a Gaussian with a known mean of 0 and unknown sigma
#take the log of the likelihood
def Likelihood(sig, dat):
    z = ( 1 / (np.sqrt(2.0 * np.pi)* sig))* np.exp((-((dat)**2.0)/(2.0*(sig**2.0))))
    #define the prior - make it flat
    #i gave it a sigma range of 0.1 - 10 which seemed like a generous range since I know it needs to 
    #be relatively small. If sigma is outside that range, prior = 0. 
    if sig < 0.1 or sig > 10:
        return(-1e6)
    return(np.sum(np.log(z)))
#print(Likelihood(1, data))
#print(Likelihood(3, data))

### MCMC TIME ###
#inputs are data array, alpha starting value, number of steps for convergence
def MCMC(dat, alp, conv):
    #create proposal - gaussian with new sigma of our choosing
    a_0 = alp #first step
    niter = conv
    naccept = 0 #creating a counter for the points accepted
    accepted_values = []
    new_steps = []
    step_iter = []

    #create jump
    
    for j in range(0, niter):

        new_step = a_0 + stats.norm(0, alp).rvs()
        new_steps.append(new_step)
        #print('current new step', new_step)
        #print('likelihood proposed', Likelihood(new_step, dat))
        #print('likelihood old', Likelihood(a_0, dat))

        #calculate Hastings Ratio - sigma in our likelihood is alpha here
        #ratio of likelihoods = difference of log likelihood
        y = Likelihood(new_step, dat) - Likelihood(a_0, dat)
        #print(y)
       # #now we want the negative number to be chosen, so compare to 0
        ratio =min(0, y)
        #print("the ratio is" , np.exp(ratio))
       
        #generate a random number between 0 and 1
        u = np.random.uniform(0,1)
        #print("the random number is", u)
        #print("the current alpha is", a_0)
        #we need to re-exponentiate the log ratio to compare it to the random number
        if u < np.exp(ratio):
            #accept the point
            naccept = naccept + 1
            a_0 = new_step
            accepted_values.append(new_step)
            step_iter.append(j)
        else:
            step_iter.append(j)
            accepted_values.append(a_0)
        #print("the new alpha will be", a_0)
        #print(' -----------')
        #print(' -----------')
    print(len(accepted_values))
    #print(naccept/niter)
    plt.hist(accepted_values, bins=25, density=True)
    plt.show()

    #Running mean plot
    run_avg1 = np.cumsum(accepted_values)
    final = []
    
    for i in range(0,len(run_avg1)): 
        final.append(run_avg1[i]/(1+i))
   
    print(len(step_iter))
    print(len(final))

    plt.plot(step_iter, final)
    plt.show()
    
    return(a_0, accepted_values)

MCMC(data, 1, 10000)
