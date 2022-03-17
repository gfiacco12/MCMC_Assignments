import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform 
import scipy.stats as stats
from scipy.special import gamma, factorial

### MCMC ASSIGNMENT 1 ###
#Create an MCMC that draws from Student-t distribution

#Since we want to output a student-t distribution, our estimated parameter needs to be the data we want 
#to model, so x_i is going to be our jumps

#Now create our likelihood function
#inputs are x: jump (aka simulated data point), mu,sig,nu: parameters of the distribution
def loglike(x):
    mu = 1
    sig = 1
    nu = 3

    a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
    b = (1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
    pi = a * b
    #now impliment flat priors for our parameter
    if x > sig+4 or x < sig-4:
       return (-1e6)
    return(np.sum(np.log(pi))) #returns sum of the logs

#create the MCMC
#inputs are starting point, jump size, number of points until convergence 
def mcmc(alp, conv):
    x_0 = alp #starting point
    naccept = 0 #counter for acceptance
    accepted_values = [] #array for all accepted values
    step_iter = [] #array that matches all accepted values

    #create the jump
    for i in range(0, conv):

        #proposal distribution is a gaussian, with jump size alpha, mean (???)
        new_step = x_0 + stats.norm(0, alp).rvs()
       # print('current new step', new_step)
        #calculate Hastings ratio - difference of the logs
        y = loglike(new_step) - loglike(x_0)
        #now we want the negative number to be chosen, so compare to 0
        ratio =min(0, y)
        #print("the ratio is" , np.exp(ratio))
        #generate a random number between 0 and 1
        u = np.random.uniform(0,1)
        #print("the random number is", u)
        #print("the current jump is", x_0)
        #we need to re-exponentiate the log ratio to compare it to the random number
        if u < np.exp(ratio):
            #accept the point
            naccept = naccept + 1
            x_0 = new_step
            accepted_values.append(new_step)
            step_iter.append(i)
        else:
            accepted_values.append(x_0)
        #print("the new alpha will be", x_0)
        #print(' -----------')
        #print(' -----------')
    '''''
        for j in accepted_values:
            mu = 1
            sig = 1
            nu = 3

            a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
            b = (1 + (1/nu)*((j - mu)/sig)**2)**(-0.5*(nu+1))
            pi = a * b
            dist_data.append(pi)
    '''
    print(naccept/conv)

    ### Graphing ###
    mu = 1
    sig = 1
    nu = 3
    x = np.linspace(sig-4, sig+4, len(accepted_values) )
    a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
    b = (1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
    pi = a * b
    
    plt.hist(accepted_values, bins=50, density=True, label="MCMC data")
    plt.plot(x, pi, label="Student-t Dist")
    plt.legend()
    plt.title("Posterior Distribution")
    plt.show()

    return(accepted_values, naccept, step_iter)

mcmc(0.1, 50000)

