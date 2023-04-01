#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
from scipy.stats import norm
import numpy as np
import datetime as dt

def d1(S, K, r, q, sigma, T):
    return (math.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, r, q, sigma, T):
    return d1(S, K, r, q, sigma, T) - sigma * math.sqrt(T)

def black_scholes_merton(S, K, r, q, sigma, T):
    """
    S: stock price
    K: strike price
    r: risk-free interest rate
    q: continuously compounding dividend yield
    sigma: volatility
    T: time to expiration (in years)
    """
    d1_val = d1(S, K, r, q, sigma, T)
    d2_val = d2(S, K, r, q, sigma, T)
    call_price = S * math.exp(-q * T) * norm.cdf(d1_val) - K * math.exp(-r * T) * norm.cdf(d2_val)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2_val) - S * math.exp(-q * T) * norm.cdf(-d1_val)
    return call_price, put_price

def delta_call(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return math.exp((q - r) * T) * norm.cdf(d1_val)

def delta_put(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return math.exp((q - r) * T) * (norm.cdf(d1_val)-1)

def gamma(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return math.exp((q - r) * T) * norm.pdf(d1_val) / (S * sigma * math.sqrt(T))

def theta_call(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    d2_val = d2(S, K, r, q, sigma, T)
    return -((S*math.exp((q - r) * T)*norm.pdf(d1_val) * sigma)/(2 * math.sqrt(T)))-(q - r)*S*math.exp((q - r)*T)*norm.cdf(d1_val) - r*K*math.exp((-r)*T)*norm.cdf(d2_val)

def theta_put(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    d2_val = d2(S, K, r, q, sigma, T)
    return -((S*math.exp((q - r) * T)*norm.pdf(d1_val) * sigma)/(2 * math.sqrt(T)))+(q - r)*S*math.exp((q - r)*T)*norm.cdf(-d1_val) + r*K*math.exp((-r)*T)*norm.cdf(-d2_val)

def vega(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return S * math.exp((q - r) * T) * norm.pdf(d1_val) * math.sqrt(T)

def rho_call(S, K, r, q, sigma, T):
    d2_val = d2(S, K, r, q, sigma, T)
    return T * K * math.exp(-r * T) * norm.cdf(d2_val)

def rho_put(S, K, r, q, sigma, T):
    d2_val = d2(S, K, r, q, sigma, T)
    return -T * K * math.exp(-r * T) * norm.cdf(-d2_val) 

def carry_rho_call(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return T * S * math.exp((q - r) * T) * norm.cdf(d1_val)

def carry_rho_put(S, K, r, q, sigma, T):
    d1_val = d1(S, K, r, q, sigma, T)
    return -T * S * math.exp((q - r) * T) * norm.cdf(-d1_val) 


S = 165.0
K = 165.0
r = 0.0425
q = 0.0053
sigma = 0.20
T = (dt.datetime(2023,4,15) - dt.datetime(2023,3,13)).days / 365

delta_call = delta_call(S, K, r, q, sigma, T)
gamma_call = gamma(S, K, r, q, sigma, T)
theta_call = theta_call(S, K, r, q, sigma, T)
vega_call = vega(S, K, r, q, sigma, T)
rho_call = rho_call(S, K, r, q, sigma, T)
carry_rho_call = carry_rho_call(S, K, r, q, sigma, T)

delta_put = delta_put(S, K, r, q, sigma, T)
gamma_put = gamma(S, K, r, q, sigma, T)
theta_put = theta_put(S, K, r, q, sigma, T)
vega_put = vega(S, K, r, q, sigma, T)
rho_put = rho_put(S, K, r, q, sigma, T)
carry_rho_put = carry_rho_put(S, K, r, q, sigma, T)

print("Call:")
print("Delta: ", delta_call)
print("Gamma: ", gamma_call)
print("Theta: ", theta_call)
print("Vega: ", vega_call)
print("Rho: ", rho_call)
print("Carry rho: ", carry_rho_call)

print("\nPut:")
print("Delta: ", delta_put)
print("Gamma: ", gamma_put)
print("Theta: ", theta_put)
print("Vega: ", vega_put)
print("Rho: ", rho_put)
print("Carry rho: ", carry_rho_put)


# In[52]:


def black_scholes_merton(S, K, r, q, sigma, T):
    """
    S: stock price
    K: strike price
    r: risk-free interest rate
    q: continuously compounding dividend yield
    sigma: volatility
    T: time to expiration (in years)
    """
    d1_val = d1(S, K, r, q, sigma, T)
    d2_val = d2(S, K, r, q, sigma, T)
    call_price = S * math.exp(-q * T) * norm.cdf(d1_val) - K * math.exp(-r * T) * norm.cdf(d2_val)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2_val) - S * math.exp(-q * T) * norm.cdf(-d1_val)
    return call_price,put_price

def delta_fd_call(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S + h, K, r, q, sigma, T)[0] - black_scholes_merton(S - h, K, r, q, sigma, T)[0]) / (2 * h)

def gamma_fd_call(S, K, r, q, sigma, T, h):
     return (black_scholes_merton(S + h, K, r, q, sigma, T)[0] - 2 * black_scholes_merton(S, K, r, q, sigma, T)[0] + black_scholes_merton(S - h, K, r, q, sigma, T)[0]) / h ** 2

def theta_fd_call(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q, sigma, T - h)[0] - black_scholes_merton(S, K, r, q, sigma, T + h)[0]) / (2 * h)

def vega_fd_call(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q, sigma + h, T)[0] - black_scholes_merton(S, K, r, q, sigma - h, T)[0]) / (2 * h)

def rho_fd_call(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r + h, q, sigma, T)[0] - black_scholes_merton(S, K, r - h, q, sigma, T)[0]) / (2 * h)

def carry_rho_fd_call(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q - h, sigma, T)[0] - black_scholes_merton(S, K, r, q + h, sigma, T)[0]) / (2 * h)


h1 = 0.00001
delta_call_d = delta_fd_call(S, K, r, q, sigma, T, h1)
gamma_call_d = gamma_fd_call(S, K, r, q, sigma, T, h1)
theta_call_d = theta_fd_call(S, K, r, q, sigma, T, h1)
vega_call_d = vega_fd_call(S, K, r, q, sigma, T, h1)
rho_call_d = rho_fd_call(S, K, r, q, sigma, T, h1)
carry_rho_call_d = carry_rho_fd_call(S, K, r, q, sigma, T, h1)
print("Call:")
print("Delta: ", delta_call_d)
print("Gamma: ", gamma_call_d)
print("Theta: ", theta_call_d)
print("Vega: ", vega_call_d)
print("Rho: ", rho_call_d)
print("Carry rho: ", carry_rho_call_d)


# In[20]:


def delta_fd_put(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S + h, K, r, q, sigma, T)[1] - black_scholes_merton(S - h, K, r, q, sigma, T)[1]) / (2 * h)

def gamma_fd_put(S, K, r, q, sigma, T, h):
     return (black_scholes_merton(S + h, K, r, q, sigma, T)[1] - 2 * black_scholes_merton(S, K, r, q, sigma, T)[1] + black_scholes_merton(S - h, K, r, q, sigma, T)[1]) / h ** 2

def theta_fd_put(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q, sigma, T - h)[1] - black_scholes_merton(S, K, r, q, sigma, T + h)[1]) / (2 * h)

def vega_fd_put(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q, sigma + h, T)[1] - black_scholes_merton(S, K, r, q, sigma - h, T)[1]) / (2 * h)

def rho_fd_put(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r + h, q, sigma, T)[1] - black_scholes_merton(S, K, r - h, q, sigma, T)[1]) / (2 * h)

def carry_rho_fd_put(S, K, r, q, sigma, T, h):
    return (black_scholes_merton(S, K, r, q - h, sigma, T)[1] - black_scholes_merton(S, K, r, q + h, sigma, T)[1]) / (2 * h)

h1 = 0.00001
delta_put_d = delta_fd_put(S, K, r, q, sigma, T, h1)
gamma_put_d = gamma_fd_put(S, K, r, q, sigma, T, h1)
theta_put_d = theta_fd_put(S, K, r, q, sigma, T, h1)
vega_put_d = vega_fd_put(S, K, r, q, sigma, T, h1)
rho_put_d = rho_fd_put(S, K, r, q, sigma, T, h1)
carry_rho_put_d = carry_rho_fd_put(S, K, r, q, sigma, T, h1)
print("Put:")
print("Delta: ", delta_put_d)
print("Gamma: ", gamma_put_d)
print("Theta: ", theta_put_d)
print("Vega: ", vega_put_d)
print("Rho: ", rho_put_d)
print("Carry rho: ", carry_rho_put_d)


# In[22]:


delta_diff_call = (delta_call_d - delta_call)/delta_call
gamma_diff_call = (gamma_call_d - gamma_call)/gamma_call
theta_diff_call = (theta_call_d - theta_call)/theta_call
vega_diff_call = (vega_call_d - vega_call)/vega_call
rho_diff_call = (rho_call_d - rho_call)/rho_call
carry_rho_diff_call = (carry_rho_call_d - carry_rho_call)/carry_rho_call
delta_diff_put = (delta_put_d - delta_put)/delta_put
gamma_diff_put = (gamma_put_d - gamma_put)/gamma_put
theta_diff_put = (theta_put_d - theta_put)/theta_put
vega_diff_put = (vega_put_d - vega_put)/vega_put
rho_diff_put = (rho_put_d - rho_put)/rho_put
carry_rho_diff_put = (carry_rho_put_d - carry_rho_put)/carry_rho_put

print("Percentage difference between closed form and finite difference")
print("Delta Call: {:.2f}%".format(delta_diff_call*100))
print("Gamma Call: {:.2f}%".format(gamma_diff_call*100))
print("Theta Call: {:.2f}%".format(theta_diff_call*100))
print("Vega Call: {:.2f}%".format(vega_diff_call*100))
print("Rho Call: {:.2f}%".format(rho_diff_call*(-100)))
print("Carry Rho Call: {:.2f}%".format(carry_rho_diff_call*100))
print("\n")
print("Delta Put: {:.2f}%".format(delta_diff_put*100))
print("Gamma Put: {:.2f}%".format(gamma_diff_put*100))
print("Theta Put: {:.2f}%".format(theta_diff_put*(-100)))
print("Vega Put: {:.2f}%".format(vega_diff_put*100))
print("Rho Put: {:.2f}%".format(rho_diff_put*(-100)))
print("Carry Rho Put: {:.2f}%".format(carry_rho_diff_put*100))


# In[23]:


def binomial_tree_american_continous(S0, K, T, r, q, sigma, N=200, option_type='call'):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp((r-q)*dt)-d)/(u-d)
    pd = 1-pu
    df = np.exp(-r*dt)
    z = 1 if option_type == 'call' else -1
    def nNodeFunc(n):
        return (n+2)*(n+1)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
    nNodes = nNodeFunc(N)
    optionValues = np.empty(nNodes, dtype = float)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i,j)
            price = S0*u**i*d**(j-i)
            optionValues[idx] = max(0,z*(price-K))
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
    return optionValues[0]

S = 165.0
K = 165.0
r = 0.0425
q = 0.0053
sigma = 0.20
T = (dt.datetime(2023,4,15) - dt.datetime(2023,3,13)).days / 365
binomial_tree_american_continous(S, K, T, r, q, sigma, N=200, option_type='call')
binomial_tree_american_continous(S, K, T, r, q, sigma, N=200, option_type='put')
print("Price of Call with no Dividend: " ,binomial_tree_american_continous(S, K, T, r, q, sigma, N=100, option_type='call'))
print("Price of Put with no Dividend: " ,binomial_tree_american_continous(S, K, T, r, q, sigma, N=100, option_type='put'))


# In[24]:


def binomial_tree_american_discrete(S0, K, r, T, sigma, N, option_type, dividend_dates=None, dividend_amounts=None):
    if dividend_dates is None or dividend_amounts is None or (len(dividend_amounts)==0) or (len(dividend_dates)==0):
        return binomial_tree_american_continous(S0, K, T, r, 0, sigma, N, option_type)
    elif dividend_dates[0] > N:
        return binomial_tree_american_continous(S0, K, T, r, 0, sigma, N, option_type)

    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(r*dt)-d)/(u-d)
    pd = 1-pu
    df = np.exp(-r*dt)
    z = 1 if option_type == 'call' else -1
    
    def nNodeFunc(n):
        return (n+2)*(n+1)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
   
    nDiv = len(dividend_dates)
    nNodes = nNodeFunc(dividend_dates[0])

    optionValues = np.empty(nNodes, dtype = float)

    for j in range(dividend_dates[0],-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = S0*u**i*d**(j-i)       
            
            if j < dividend_dates[0]:
                #times before the dividend working backward induction
                optionValues[idx] = max(0,z*(price-K))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
                
            else:
                no_ex= binomial_tree_american_discrete(price-dividend_amounts[0], K, r, T-dividend_dates[0]*dt, sigma, N-dividend_dates[0], option_type, [x- dividend_dates[0] for x in dividend_dates[1:nDiv]], dividend_amounts[1:nDiv] )
                ex =  max(0,z*(price-K))
                optionValues[idx] = max(no_ex,ex)

    return optionValues[0]

S = 165.0
K = 165.0
r = 0.0425
q = 0.0053
sigma = 0.20
T = (dt.datetime(2023,4,15) - dt.datetime(2023,3,13)).days / 365
N = 100
dividend_dates = [round((dt.datetime(2023,4,11)-dt.datetime(2023,3,13)).days/(dt.datetime(2023,4,15)-dt.datetime(2023,3,13)).days*N)]
dividend_amounts = [0.88]

print("Price of Call with Dividend: " ,binomial_tree_american_discrete(S, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts))
print("Price of Put with Dividend: " ,binomial_tree_american_discrete(S, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts))


# In[67]:


delta_call_bi_nodiv = (binomial_tree_american_continous(S+h1, K, T, r, q, sigma, N=100, option_type='call') - binomial_tree_american_continous(S-h1, K, T, r, q, sigma, N=100, option_type='call')) / (2 * h1)
delta_put_bi_nodiv = (binomial_tree_american_continous(S+h1, K, T, r, q, sigma, N=100, option_type='put') - binomial_tree_american_continous(S-h1, K, T, r, q, sigma, N=100, option_type='put')) / (2 * h1)
gamma_call_bi_nodiv = (binomial_tree_american_continous(S+h1, K, T, r, q, sigma, N=100, option_type='call') - 2*binomial_tree_american_continous(S, K, T, r, q, sigma, N=100, option_type='call')+ binomial_tree_american_continous(S-h1, K, T, r, q, sigma, N=100, option_type='call')) / ( h1**2)
gamma_put_bi_nodiv = (binomial_tree_american_continous(S+h1, K, T, r, q, sigma, N=100, option_type='put') - 2*binomial_tree_american_continous(S, K, T, r, q, sigma, N=100, option_type='put')+ binomial_tree_american_continous(S-h1, K, T, r, q, sigma, N=100, option_type='call')) / ( h1**2)
theta_call_bi_nodiv = (binomial_tree_american_continous(S, K, T-h1, r, q, sigma, N=100, option_type='call') - binomial_tree_american_continous(S, K, T+h1, r, q, sigma, N=100, option_type='call')) / (2 * h1)
theta_put_bi_nodiv = (binomial_tree_american_continous(S, K, T-h1, r, q, sigma, N=100, option_type='put') - binomial_tree_american_continous(S, K, T+h1, r, q, sigma, N=100, option_type='put')) / (2 * h1)
vega_call_bi_nodiv = (binomial_tree_american_continous(S, K, T, r, q, sigma+h1, N=100, option_type='call') - binomial_tree_american_continous(S, K, T, r, q, sigma-h1, N=100, option_type='call')) / (2 * h1)
vega_put_bi_nodiv = (binomial_tree_american_continous(S, K, T, r, q, sigma+h1, N=100, option_type='put') - binomial_tree_american_continous(S, K, T, r, q, sigma-h1, N=100, option_type='put')) / (2 * h1)
rho_call_bi_nodiv = (binomial_tree_american_continous(S, K, T, r+h1, q, sigma, N=100, option_type='call') - binomial_tree_american_continous(S, K, T, r-h1, q, sigma, N=100, option_type='call')) / (2 * h1)
rho_put_bi_nodiv = (binomial_tree_american_continous(S, K, T, r+h1, q, sigma, N=100, option_type='put') - binomial_tree_american_continous(S, K, T, r-h1, q, sigma, N=100, option_type='put')) / (2 * h1)


delta_call_bi_div = (binomial_tree_american_discrete(S+h1, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S-h1, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts)) / (2 * h1)
delta_put_bi_div = (binomial_tree_american_discrete(S+h1, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S-h1, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts)) / (2 * h1)
gamma_call_bi_div = (binomial_tree_american_discrete(S+h1, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts) - 2*binomial_tree_american_discrete(S, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts)+ binomial_tree_american_discrete(S-h1, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts)) / ( h1**2)
gamma_put_bi_div = (binomial_tree_american_discrete(S+h1, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts) - 2*binomial_tree_american_discrete(S, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts)+ binomial_tree_american_discrete(S-h1, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts)) / ( h1**2)
theta_call_bi_div = (binomial_tree_american_discrete(S, K, r, T-h1, sigma, N, 'call', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r, T+h1, sigma, N, 'call', dividend_dates, dividend_amounts)) / (2 * h1)
theta_put_bi_div = (binomial_tree_american_discrete(S, K, r, T-h1, sigma, N, 'put', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r, T+h1, sigma, N, 'put', dividend_dates, dividend_amounts)) / (2 * h1)
vega_call_bi_div = (binomial_tree_american_discrete(S, K, r, T, sigma+h1, N, 'call', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r, T, sigma-h1, N, 'call', dividend_dates, dividend_amounts)) / (2 * h1)
vega_put_bi_div = (binomial_tree_american_discrete(S, K, r, T, sigma+h1, N, 'put', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r, T, sigma-h1, N, 'put', dividend_dates, dividend_amounts)) / (2 * h1)
rho_call_bi_div = (binomial_tree_american_discrete(S, K, r+h1, T, sigma, N, 'call', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r-h1, T, sigma, N, 'call', dividend_dates, dividend_amounts)) / (2 * h1)
rho_put_bi_div = (binomial_tree_american_discrete(S, K, r+h1, T, sigma, N, 'put', dividend_dates, dividend_amounts) - binomial_tree_american_discrete(S, K, r-h1, T, sigma, N, 'put', dividend_dates, dividend_amounts)) / (2 * h1)

print("Delta for call without dividend: ", delta_call_bi_nodiv)
print("Delta for put without dividend: ", delta_put_bi_nodiv)
print("Gamma for call without dividend: ", gamma_call_bi_nodiv)
print("Gamma for put without dividend: ", gamma_put_bi_nodiv)
print("Theta for call without dividend: ", theta_call_bi_nodiv)
print("Theta for put without dividend: ", theta_put_bi_nodiv)
print("Vega for call without dividend: ", vega_call_bi_nodiv)
print("Vega for put without dividend: ", vega_put_bi_nodiv)
print("Rho for call without dividend: ", rho_call_bi_nodiv)
print("Rho for put without dividend: ", rho_put_bi_nodiv)
print("\n")
print("Delta for call with dividend: ", delta_call_bi_div)
print("Delta for put with dividend: ", delta_put_bi_div)
print("Gamma for call with dividend: ", gamma_call_bi_div)
print("Gamma for put with dividend: ", gamma_put_bi_div)
print("Theta for call with dividend: ", theta_call_bi_div)
print("Theta for put with dividend: ", theta_put_bi_div)
print("Vega for call with dividend: ", vega_call_bi_div)
print("Vega for put with dividend: ", vega_put_bi_div)
print("Rho for call with dividend: ", rho_call_bi_div)
print("Rho for put with dividend: ", rho_put_bi_div)
print("\n")

dividend_amounts = [0.88]
first_call = binomial_tree_american_discrete(S, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts)
first_put = binomial_tree_american_discrete(S, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts)
dividend_amounts = [1.88]
second_call = binomial_tree_american_discrete(S, K, r, T, sigma, N, 'call', dividend_dates, dividend_amounts)
second_put = binomial_tree_american_discrete(S, K, r, T, sigma, N, 'put', dividend_dates, dividend_amounts)
sensitivity_call = second_call - first_call
sensitivity_put = second_put - first_put
print("Sensitivity for call: ", sensitivity_call)
print("Sensitivity for put: ", sensitivity_put)


# In[ ]:




