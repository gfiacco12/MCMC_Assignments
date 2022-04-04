import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform 
import scipy.stats as stats
from scipy.special import gamma, factorial
import statsmodels.api as sm



### MCMC ASSIGNMENT 1 ###
#Create an MCMC that draws from Student-t distribution

#Since we want to output a student-t distribution, our estimated parameter needs to be the data we want 
#to model, so x_i is going to be our jumps

#Now create our likelihood function
#inputs are x: jump (aka simulated data point), mu,sig,nu: parameters of the distribution
def loglike(x):
    mu = -2
    sig = 7
    nu = 10

    a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
    b = (1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
    pi = a * b
    #now impliment flat priors for our parameter
    if x > sig+20 or x < sig-50:
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
        new_step = x_0 + (stats.norm(0, 1).rvs() / np.sqrt(alp))
       
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

    mu = -2
    sig = 7
    nu = 10
    x = np.linspace(-20,20, conv )
    pi = stats.t.pdf(x, nu, mu, sig)
    
    plt.hist(accepted_values, bins=50, density=True, label="MCMC data")
    plt.plot(x, pi, label="Student-t Dist")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.xlim(-20, 20)
    plt.title("Posterior Distribution using Fisher Jumps")
    plt.show()
    
    return(accepted_values, naccept, step_iter)
mcmc(0.0212, 50000)


### Part a - Plotting chains for different jump values 

def chain_plotting():

    #val = accepted value array, n = counter, iter = array of iterations
    val1, n1, iter1 = mcmc(0.1, 100)
    val2, n2, iter2 = mcmc(0.01, 1000)
    val3, n3, iter3 = mcmc(1, 1000)
    val4, n4, iter4 = mcmc(10, 1000)
 
    #Trace plot
    
    fig, axs = plt.subplots(2, 2)
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    plt.suptitle("Trace plot for N = 10,000")

    axs[0, 1].plot(iter1, val1)
    axs[0, 1].set_title("$\\alpha$=0.1")
    axs[0,0].plot(iter2, val2)
    axs[0, 0].set_title("$\\alpha$=0.01")
    axs[1,0].plot(iter3, val3)
    axs[1,0].set_title("$\\alpha$=1.0")
    axs[1,1].plot(iter4, val4)
    axs[1, 1].set_title("$\\alpha$=10.0")

    plt.xlabel("Iterations")
    plt.ylabel("Jumps")
    
    plt.tight_layout()
    plt.show()
  
    return()

#chain_plotting()
#mcmc(10.0, 10000)
from statsmodels.graphics import tsaplots
def stat_calc():
    data, n1, iter1 = mcmc(0.01, 1000000)
    #val2, n2, iter2 = mcmc(0.01, 1000000)
    #val3, n3, iter3 = mcmc(1, 1000000)
    #val4, n4, iter4 = mcmc(10, 1000000)
    fig1 = tsaplots.plot_acf(data, lags=10000)
    plt.title("Autocorrelation for $\\alpha=10$")
    plt.xlabel("h")
    plt.ylabel("Correlation Coefficient")
    '''
    fig2 = tsaplots.plot_acf(val2, lags=60)
    plt.title("Autocorrelation for $\\alpha=0.01$")
    plt.xlabel("Lag at k")
    plt.ylabel("Correlation Coefficient")
    fig3 = tsaplots.plot_acf(val3, lags=60)
    plt.title("Autocorrelation for $\\alpha=1$")
    plt.xlabel("Lag at k")
    plt.ylabel("Correlation Coefficient")
    fig4 = tsaplots.plot_acf(val4, lags=60)
    plt.title("Autocorrelation for $\\alpha=10$")
    plt.xlabel("Lag at k")
    plt.ylabel("Correlation Coefficient")
    '''
    plt.show()
    return()
    
#Fisher Information matrix
#compute partial derivatives with respect to x - our data
import sympy as sp
from sympy import symbols, diff, solve
def fisher():
    x = symbols('x', real=True)
    mu = -2
    sig = 7
    nu = 10
    a = (gamma((nu+1)/2) / (gamma(nu/2) * np.sqrt(2*np.pi)*sig))
    b = (1 + (1/nu)*((x - mu)/sig)**2)**(-0.5*(nu+1))
    pi = a * b

    #find maxima for pi
    fprime = sp.diff(pi,x)
    max_val = sp.solve(fprime, x)
    print(max_val)
    #want ln(pi)
    f = sp.log(pi)
    #take two partial derivatives wrt x
    diff1 = sp.diff(f,x)
    diff2 = compile(str(sp.diff(diff1,x)), 'test', 'eval')

    print(diff2)
    #now evaluate at maxima
    x = max_val

    fisher = -eval(diff2,{"x": 1})
    print(fisher)
    return()
#fisher()