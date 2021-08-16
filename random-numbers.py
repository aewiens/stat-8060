#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)   


def plot_histogram(sample, title):
    """ Plot a histogram of an array samples and output to screen.
    
    Parameters
    ----------
    sample : ndarray
        1D array (length = size) of generated samples.
    title : string
        Plot title for the histogram.
    
    Return
    ------
    None
    """
    n_bins = 50
    
    # premake the plot.
    fig, ax = plt.subplots()   

    # histogram of the sample
    n, bins, patches = ax.hist(sample, n_bins, normed=1)

    # name the axes
    ax.set_xlabel('Generated values')
    ax.set_ylabel('Probability density')
    ax.set_title(title)
    
    # plot
    plt.show()



# Write your own random number generator for the following distributions:

print("-------------------------------------------------------------------")
print(" Unif[0,1] distribution ")
print("-------------------------------------------------------------------")

def my_uniform(size, low=0.0, high=1.0):
    """ 
    Draw random samples from a uniform distribution (default Unif[0,1]). This 
    is a multiplicative congruential generator. It may sometimes fail due to
    overflow.
    
    Parameters
    ----------
    size : int
        Number of random uniform samples to generate
    low : float
        Lower bound of uniform distribution (default 0.0)
    high : float
        Upper bound of uniform distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a uniform distribution 
        
    """
    # minimum standard constants
    m = 2**31-1
    a = 7**5
     
    # algorithm
    sample = np.zeros(size)
    value = 1     # seed
    for i in range(size):
        value = (a * value) % m 
        sample[i] = value / m 
    
    # print sample mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(1/2 * (low + high)))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}\n".format(1/12 *(high - low)**2))
    
    # plot a histogram
    plot_title = r"Unif[0,1] Distribution"
    plot_histogram(sample, plot_title)
            
    return sample

# generate sample for (2a)
my_uniform(size=1000, low=0, high=1)


print("----------------------------------------------------------")
print(" Exp(beta) distribution ")
print("----------------------------------------------------------")


def my_random_exponential(size, beta):
    """ Draw random samples from an exponential distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    beta : float
        Scale parameter (1/rate) of exponential distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from an exponential distribution 
        
    """
    # algorithm
    uniform = np.random.uniform(0, 1, size)
    sample = [-beta * np.log(1 - i) for i in uniform]
    
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(beta))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(beta**2))
    
    # plot a histogram
    plot_title = r"Exponential ($\beta$ = {:1.1f}) Distribution".format(beta)
    plot_histogram(sample, plot_title)
    
    return sample

my_random_exponential(size=1000, beta=1.2)



print("------------------------------------")
print(" (2c) Standard Cauchy distribution ")
print("------------------------------------")

def my_standard_cauchy(size):
    """ Draw random samples from a standard cauchy distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate

    Return
    ------
    ndarray
        1D array (length = size) of samples from a standard cauchy distribution 
    """
    # algorithm
    uniform = np.random.uniform(0,1,size)
    sample = [np.tan(np.pi*(u-0.5)) for u in uniform]
    
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: Undefined")
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: Undefined")
    
    # plot a histogram
    plot_title = r"Standard Cauchy Distribution"
    plot_histogram(sample, plot_title)
    
    return sample

my_standard_cauchy(size=1000)

##############################################################################
# 2d) Normal (but not standard normal) distribution

print("------------------------------------")
print(" (2d) Normal distribution ")
print("------------------------------------")

def my_normal(size, mu=0.0, sigma=1.0):
    """ Draw random samples from a normal distribution 
    
    Parameters
    ----------
    size : int
        Number of  samples to generate
    mu : float
        mean of the normal distribution (default 0.0)
    sigma : float
        standard deviation of the normal distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a normal distribution 
    """
    # size must be even
    uniform = np.random.uniform(0, 1, size)
    sample = np.zeros(size)
    
    # algorithm (Box-Mueller Transformation)
    for i in range(0, size, 2):
        
        u1 = uniform[i]
        u2 = uniform[i+1]
        
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        
        sample[i]   = sigma * z1 + mu
        sample[i+1] = sigma * z2 + mu
        
    # print mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(mu))

    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(sigma**2))
    
    # plot a histogram
    plot_title = r"N({:1.1f}, {:1.1f}) Distribution".format(mu,sigma)
    plot_histogram(sample, plot_title)
    
    return sample

# generate sample for (2d)
my_normal(size=1000, mu=1.0, sigma=0.5)


print("------------------------------------------------------")
print(" Lognormal distribution ")
print("------------------------------------------------------")


def my_logNormal(size, mu=0.0, sigma=1.0):
    """ Draw random samples from a normal distribution 
    
    Parameters
    ----------
    size : int
        Number of  samples to generate
    mu : float
        mean of the normal distribution (default 0.0)
    sigma : float
        standard deviation of the normal distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a normal distribution 
    """
    Z = np.random.randn(size)
    X = mu + sigma * Z
    sample = np.exp(X)
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    theoretical_mean = np.exp(mu + sigma**2/2)
    theoretical_var = (np.exp(sigma**2) - 1) * (np.exp(2*mu + sigma**2))
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(theoretical_mean))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(theoretical_var))
    
    # plot a histogram
    plot_title = r'LogN({:3.1f}, {:3.1f}) Distribution'.format(mu, sigma)
    plot_histogram(sample, plot_title)
    
    return sample

logN = my_logNormal(10000, mu=0, sigma=0.5)


print("------------------------------------------------------")
print(" Chi-square distribution ")
print("------------------------------------------------------")

def my_chi_squared(size, k):
    """ Draw random samples from a chi squared distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    k : int
        Degrees of freedom for chi squared distribution
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a chi squared distribution  
    """
    # algorithm 
    sample = np.zeros(size)
    
    for i in range(size):
        
        Z = np.random.randn(5)
        X = sum([z**2 for z in Z])
        
        sample[i] = X
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(k))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(2*k))
    
    # plot a histogram
    plot_title = r'$\chi^2$ ({:d}) Distribution'.format(k)
    plot_histogram(sample, plot_title)
    
    return sample

# generate sample for (2e)
my_chi_squared(size=1000, k=5)


print("------------------------------------------------------")
print(" Student's t distribution ")
print("------------------------------------------------------")


def my_students_t(size, n):
    """ Draw random samples from a normal distribution 
    
    Parameters
    ----------
    size : int
        Number of  samples to generate
    mu : float
        mean of the normal distribution (default 0.0)
    sigma : float
        standard deviation of the normal distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a normal distribution 
    """
    Z = np.random.standard_normal(size=size)
    X = np.random.chisquare(df=n, size=size)
    sample = Z / np.sqrt(1/n * X)
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(0))  # n > 1 only 
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(n / (n-2))) # n > 2 only
    
    # plot a histogram
    plot_title = r"Student's T(df={:3.1f}) Distribution".format(n)
    plot_histogram(sample, plot_title)
    
    return sample


T = my_students_t(size=10000, n=5)


print("------------------------------------------------------")
print(" F distribution ")
print("------------------------------------------------------")


def my_random_F(size, m, n):
    """ Draw random samples from a normal distribution 
    
    Parameters
    ----------
    size : int
        Number of  samples to generate
    mu : float
        mean of the normal distribution (default 0.0)
    sigma : float
        standard deviation of the normal distribution (default 1.0)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a normal distribution 
    """
    X1 = np.random.chisquare(df=m, size=size)
    X2 = np.random.chisquare(df=n, size=size)
    F = n / m * (X1 / X2)
    sample = F
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    theoretical_mean = n / (n-2)
    theoretical_var = 2 * n**2 * (m + n -2) / m / (n-2)**2 / (n-4)
    
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(theoretical_mean))  # n > 2 only 
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(theoretical_var)) # n > 4 only
    
    # plot a histogram
    plot_title = r"F({:3.0f}, {:3.0f}) Distribution".format(m, n)
    plot_histogram(sample, plot_title)
    
    return sample


F = my_random_F(size=10000, m=100, n=100)
    

print("------------------------------------------------------")
print(" Bernoulli distribution ")
print("------------------------------------------------------")


def my_random_bernoulli(size, p):
    """ Draw random samples from a Bernoulli distribution  
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    p : float
        probability of success in each trial    
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Bernoulli(p) distribution 
    """ 
    uniform = np.random.uniform(0,1,size)
    bernoulli = np.zeros(size)
    
    for i, u in enumerate(uniform):
        if u <= p:
            bernoulli[i] = 1
    
    sample = bernoulli
    
    """
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(p))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}\n".format(p * (1 - p)))
    
    
    # plot a histogram
    plot_title = r"Bernoulli (p = {:1.1f}) Distribution".format(p)
    plot_histogram(sample, plot_title)
    """
    return bernoulli

Bernoulli = my_random_bernoulli(size=1000, p=0.6)


print("------------------------------------------------------")
print(" Binomial distribution ")
print("------------------------------------------------------")


def my_random_binomial(size, n, p):
    """ Draw random samples from a binomial distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    n : int
        number of trials in binomial distribution
    p : float
        probability of success in each trial    
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Bin(n,p) distribution 
    """
    # algorithm
    sample = [sum(my_random_bernoulli(n, p)) for i in range(size)]
    
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(n * p))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}\n".format(n * p * (1 - p)))
    
    # plot a histogram
    plot_title = r"Binomial (n = {:d}, p = {:1.1f}) Distribution".format(n,p)
    plot_histogram(sample, plot_title)
    
    return sample


binomial = my_random_binomial(size=1000, n=100, p=0.5)


print("------------------------------------------------------")
print(" Discrete uniform distribution ")
print("------------------------------------------------------")
def my_discrete_uniform(size, m):
    """ Draw random samples from a Discrete uniform (1,...,m) distribution 
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    m : float
        Upper bound of discrete uniform distribution    
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Bernoulli(p) distribution 
    """ 
    uniform = np.random.uniform(0, m, size)
    sample = np.array([np.floor(ui) + 1 for ui in uniform])
    
    
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format((1+m)/2))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}\n".format((m**2 - 1) / 12))
    
    
    # plot a histogram
    plot_title = r"Discrete Uniform (1,...,{:d}) Distribution".format(m)
    plot_histogram(sample, plot_title)
    
    return sample

discrete = my_discrete_uniform(size=10000, m=5)


def my_geometric_slow(size, p):
    """ Draw random samples from a geometric distribution via Bernoulli trials
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    p : float
        success probability
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Geo(p) distribution  
    """
    sample = np.zeros(size)

    # algorithm
    for i in range(size):
        flag = 0
        count = 0
        while flag == 0:
            y = my_random_bernoulli(size=1, p=p)[0]
            if y == 1:
                flag = 1
            count += 1

        sample[i] = count

    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(1/p))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format((1 - p)/p**2))
    
    # generate and plot a histogram
    plot_title = r"Geometric (p = {:1.1f}) Distribution (Slow)".format(p)
    plot_histogram(sample, plot_title)
    
    return sample


print("------------------------------------------------------")
print(" Geometric distribution ")
print("------------------------------------------------------")



def my_random_geometric(size, p):
    """ Draw random samples from a geometric distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    p : float
        success probability
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Geo(p) distribution  
    """
    # algorithm
    Lambda = -np.log(1-p)
    beta = 1/Lambda
    exp = np.random.exponential(beta, size)
    sample = [np.floor(i) + 1 for i in exp]
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(1/p))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format((1 - p)/p**2))
    
    # generate and plot a histogram
    plot_title = r"Geometric (p = {:1.1f}) Distribution".format(p)
    plot_histogram(sample, plot_title)
    
    return sample

# generate sample for (2g)
my_random_geometric(size=1000, p=0.5)


print("------------------------------------------------------")
print(" Poisson distribution ")
print("------------------------------------------------------")


def my_poisson(size, L):
    """ Draw random samples from a geometric distribution
    
    Parameters
    ----------
    size : int
        Number of samples to generate
    L : float
        Lambda (Parametrization of distribution)
    
    Return
    ------
    ndarray
        1D array (length = size) of samples from a Geo(p) distribution  
    """
    # algorithm
    sample = np.zeros(size)

    for i in range(size):
        s = 0
        count = 0
        while s < 1:
            s += np.random.exponential(scale=1/L, size=None)
            count += 1

        sample[i] = count - 1
        
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(L))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(L))
    
    # generate and plot a histogram
    plot_title = r"Poisson (lambda = {:1.1f}) Distribution".format(L)
    plot_histogram(sample, plot_title)
    
    return sample

test = my_poisson(size=1000, L=4)


# Sample from the Beta(2,2) distribution using the rejection method
# I sampled from Unif[0,1] and derived the necessary constant.

print("----------------------------------------------------------------------")
print(" Sample from a Beta(2,2) distribution using accept-reject")
print("----------------------------------------------------------------------")


def my_random_beta22(size):
    """ Draw random samples from a Beta(2,2) distribution
    
     Parameters
    ----------
    a (alpha) : int
        shape parameter of the gamma distribution.
    b (beta) : float
        rate parameter of the gamma distribution.
        
    Return
    ------
    ndarray
        1d numpy array (length = size)
    """
    # algorithm
    sample = np.zeros(size)                                                                                                                         
    for i in range(size):
        while 1:
            x = np.random.uniform(low=0, high=1, size=None)
            y = np.random.uniform(low=0, high=1, size=None)
            if y < 4*x*(1 - x):
              break
        sample[i] = x
    
    # print observed mean and variance
    mean   = np.mean(sample)
    var    = np.var(sample)
    print("Sample mean: {:13.4f}".format(mean))
    print("Expected mean: {:11.4f}\n".format(1/2))
    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(4/ ((5)*(4)**2)))

    # generate and plot a histogram
    plot_title = r"Beta(2,2) Distribution ({:d} points)".format(size)
    plot_histogram(sample, plot_title)
    return sample

# generate sample for problem 3
my_random_beta22(size=1000)

# this one didn't look very good with 1000 points so I used more just to
# show that my algorithm works 
my_random_beta22(size=100000)


##############################################################################
#                                                                            #
#                               PROBLEM 4                                    #
#                                                                            #
##############################################################################

# Repeat (1h) using Whittaker's method.

# 4) Implement a generator for drawing random samples from a Gamma distribution

print("--------------------------------------------------")
print(" (4) Gamma distribution using Whitaker's method ")
print("--------------------------------------------------")

# start with the naive gamma implementation
def naive_gamma(size, a, b):
    """ Draw random samples from a gamma(a,b) distribution -- with integer a.
    Parameters
    ----------
    a (alpha) : int
        shape parameter of the gamma distribution.
    b (beta) : float
        rate parameter of the gamma distribution.
        
    Return
    ------
    ndarray
        1d numpy array (length = size)
    """
    return [sum(np.random.exponential(b, a)) for i in range(size)]


def my_random_gamma(size, a, b):
    """ Draw random samples from a gamma(a,b) distribution.
    Parameters
    ----------
    a (alpha) : float
        shape parameter of the gamma distribution.
    b (beta) : float
        rate parameter of the gamma distribution.
        
    Return
    ------
    ndarray
        1d numpy array (length = size).
    """
    start = naive_gamma(size, int(np.floor(a)), b)
    sample = np.zeros(size)
    
    # algorithm
    p = a - np.floor(a)
    for i, x1 in enumerate(start):
        
        s1 = 1
        s2 = 1
        while s1 + s2 > 1:

            u1 = np.random.uniform(size=None)
            u2 = np.random.uniform(size=None)
            s1 = u1 ** (1/p)
            s2 = u2 ** (1/(1-p))
            
        x2 = s1 / (s1 + s2)
        u3 = np.random.uniform(size=None)
        x3 = -b * x2 * np.log(u3)
        
        sample[i] = x1 + x3  
    
    mean   = np.mean(sample)
    var    = np.var(sample)
    
    
    print("Sample mean: {:13.4f} ".format(mean))
    print("Expected mean: {:11.4f}\n".format(a * b))

    print("Sample variance: {:9.4f}".format(var))
    print("Expected variance: {:7.4f}".format(a * b**2))
    
    plot_title = r"Gamma ($\alpha$={:1.1f}, $\beta$={:1.1f})".format(a,b)
    plot_title += r" Distribution"
    plot_histogram(sample, plot_title)
    
    return sample

# generate sample for (4)
my_random_gamma(1000, a=1.5, b=2)
