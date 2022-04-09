from pickletools import optimize
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import uniform 
import scipy.stats as stats
from scipy.special import gamma, factorial
import statsmodels.api as sm
import scipy.linalg as lin
from scipy import optimize

#create student-t samples
def dat_like(x):
    mu = 1 
    sig = 1
    nu = 3
    pi = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))*(1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
        #now impliment flat priors for our parameter
    if x > sig+50 or x < sig-50:
       return (-1e6)
    return(np.sum(np.log(pi))) #returns sum of the logs

#for these parameters, the fisher matrix gave alpha = 1.33
def data_generator(alp, conv):
    x_0 = np.random.normal(0,1) #starting point
    naccept = 0 #counter for acceptance
    data_points = [] #array for all accepted values
    step_iter = [] #array that matches all accepted values

    #create the jump    
    #down-sample by factor of 10
    for i in range(0, conv):
        #proposal distribution is a gaussian, with jump size alpha, mean (???)
        new_step = x_0 + (stats.norm(0, 1).rvs() / np.sqrt(alp))
        #calculate Hastings ratio - difference of the logs
        y = dat_like(new_step) - dat_like(x_0)
        #now we want the negative number to be chosen, so compare to 0
        ratio =min(0, y)
        #generate a random number between 0 and 1
        u = np.random.uniform(0,1)
        #we need to re-exponentiate the log ratio to compare it to the random number
        if i%10 ==0:
            if u < np.exp(ratio):
                #accept the point
                naccept = naccept + 1
                x_0 = new_step
                data_points.append(new_step)
                step_iter.append(i)
            else:
                data_points.append(x_0)
                step_iter.append(i)
    return(data_points)

##################### MCMC TIME ##################################
def loglike(dat, param):
    mu = param[0]
    nu = param[1]
    sig = param[2]

    a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
    b = (1 + (1/nu)*((dat - mu)/sig)**2)**(-0.5*(nu+1))
    pi = a * b
    #now impliment flat priors for our parameters
    if mu > 5 or mu < -5:
        return (-1e6)
    if nu > 10 or nu < 0.1:
        return (-1e6)
    if sig > 10 or sig < 0.1:
        return (-1e6)
    return(np.sum(np.log(pi))) #returns sum of the logs


#Create the Fisher Matrix for all the parameters
def fisher(dat):
    
    #use finite difference method to calculate derivatives
    params = np.array([1, 3, 1]) #injected paramters for evaulation
    #initializing step size vectors and Hessian matrix
    vector1 = np.zeros(np.size(params))
    vector2 = np.zeros(np.size(params))
    H = np.zeros((np.size(params),np.size(params)))
    h = 0.001
    for i in range(0, len(params)):
        for j in range(0, len(params)):
            if i == j:  #diagonal elements
                vector1[i] = h
                H[i][i] = (loglike(dat, params+vector1) - 2*loglike(dat, params) + loglike(dat, params - vector1))/(h**2)
            if i != j:
                vector1[i] = h 
                vector2[j] = h
                H[i][j] = (loglike(dat, params + vector1+vector2) - loglike(dat, params + vector1-vector2) - loglike(dat, params - vector1+vector2) + loglike(dat, params - vector1-vector2)) / (4*h**2)
    fisher = -H
    #now get the eigenvalues. These will tell us the jump directions
    print(fisher)
    eigval, eigvec = lin.eig(fisher)
    print(eigval)
    print(eigvec)
    return(eigval, eigvec)

#inputs are data array from above, alpha starting value -based on Fisher Matrix
def MCMC(dat, conv):
    #create proposal - gaussian with new sigma of our choosing
    naccept = 0 #creating a counter for the points accepted
    accepted_values_mu = []
    accepted_values_nu = []
    accepted_values_sig = []
    step_iter = []
    eigval, eigvec = fisher(dat)

    #initialize starting points
    mu_0 = np.random.normal(0,1)
    nu_0 = np.random.normal(0,1)
    sig_0 = np.random.normal(0,1)
    old_params = [mu_0, nu_0, sig_0]

    #create jumps for all three parameters - vectorize
    for j in range(0, conv):

        #pick which eigendirection to jump towards
        v = np.random.uniform(0,1)
        if v <= (1/3):
            jump_dir = (1 / np.sqrt(eigval[0])) * eigvec[0] * stats.norm(0,1).rvs()
        if v > (1/3) and v <= (2/3):
            jump_dir = (1 / np.sqrt(eigval[1])) * eigvec[1] * stats.norm(0,1).rvs()
        if v > (2/3):
            jump_dir = (1 / np.sqrt(eigval[2])) * eigvec[2] * stats.norm(0,1).rvs()

        #need to create a jump for each parameter
        parameters = [new_mu, new_nu, new_sig] = old_params + jump_dir
        #print(np.real(parameters))

        #calculate Hastings Ratio - sigma in our likelihood is alpha here
        #ratio of likelihoods = difference of log likelihood
        y = loglike(dat, np.real(parameters) )- loglike(dat, np.real(old_params))

        #now we want the negative number to be chosen, so compare to 0
        ratio =min(0, y)
        #print("the ratio is" , np.exp(ratio))
       
        #generate a random number between 0 and 1
        u = np.random.uniform(0,1)

        #we need to re-exponentiate the log ratio to compare it to the random number
        if u < np.exp(ratio):
            #accept the point
            naccept = naccept + 1
            [mu_0, nu_0, sig_0] = [new_mu, new_nu, new_sig]
            accepted_values_mu.append(new_mu)
            accepted_values_nu.append(new_nu)
            accepted_values_sig.append(new_sig)
            step_iter.append(j)
        else:
            step_iter.append(j)
            accepted_values_mu.append(mu_0)
            accepted_values_nu.append(nu_0)
            accepted_values_sig.append(sig_0)
    
    plt.hist(accepted_values_nu, bins=50, density=True, label="MCMC data")
    plt.show()
    return()
MCMC(data_generator(1.33, 1000), 10000)