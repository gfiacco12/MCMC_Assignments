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
#inputs are jump size, number of points until convergence 
def mcmc(alp, conv):
    x_0 = np.random.normal(0,1) #starting point
    naccept = 0 #counter for acceptance
    accepted_values = [] #array for all accepted values
    step_iter = [] #array that matches all accepted values
    mean_values = []

    #create the jump
    for i in range(0, conv):

        #proposal distribution is a gaussian, with jump size alpha, mean (???)
        new_step = x_0 + stats.norm(0, alp).rvs()
       
        #calculate Hastings ratio - difference of the logs
        y = loglike(new_step) - loglike(x_0)

        #now we want the negative number to be chosen, so compare to 0
        ratio =min(0, y)
       
        #generate a random number between 0 and 1
        u = np.random.uniform(0,1)

        #we need to re-exponentiate the log ratio to compare it to the random number
        if u < np.exp(ratio):
            #accept the point
            naccept = naccept + 1
            x_0 = new_step
            accepted_values.append(new_step)
            step_iter.append(i)
        else:
            accepted_values.append(x_0)
            step_iter.append(i)

    print(naccept/conv)
    print(len(accepted_values), len(step_iter))

    ### Graphing ###
    #NOTE: Fix the curve, it isnt the right height. Histogram is good though

    mu = 1
    sig = 1
    nu = 3
    x = np.linspace(sig-5, sig+5, len(accepted_values) )
    pi = stats.t.pdf(x, nu, mu, sig)
    '''
    plt.hist(accepted_values, bins=50, density=True, label="MCMC data")
    plt.plot(x, pi, label="Student-t Dist")
    plt.legend()
    plt.title("Posterior Distribution")
    plt.show()
    '''
    return(accepted_values, naccept, step_iter)

### Part a - Plotting chains for different jump values 

def chain_plotting():

    #val = accepted value array, n = counter, iter = array of iterations
    val1, n1, iter1 = mcmc(0.1, 10000)
    val2, n2, iter2 = mcmc(0.01, 10000)
    val3, n3, iter3 = mcmc(1, 10000)
    val4, n4, iter4 = mcmc(10, 10000)
 
    #Trace plot
    fig, axs = plt.subplots(2, 2)
    plt.suptitle("Trace plot for N = 10,000")

    axs[0, 1].plot(iter1, val1)
    axs[0, 1].set_title("alpha=0.1")
    axs[0,0].plot(iter2, val2)
    axs[0, 0].set_title("alpha=0.01")
    axs[1,0].plot(iter3, val3)
    axs[1,0].set_title("alpha=1.0")
    axs[1,1].plot(iter4, val4)
    axs[1, 1].set_title("alpha=10.0")
    
    plt.tight_layout()
    plt.show()

    return()

chain_plotting()
