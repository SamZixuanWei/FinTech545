#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np

n = 1000
burn_in = 50
y = np.empty(n)

yt_last = 1.0
e = np.random.normal(0, 0.1, n+burn_in)

for i in range(n+burn_in):
    y_t = 1.0 + 0.5*yt_last + e[i]
    yt_last = y_t
    if i >= burn_in:
        y[i-burn_in] = y_t

print("Mean and Var of Y: {}, {}".format(np.mean(y),np.var(y)))
print("Expected values Y: {}, {}".format(2.0,.01/(1-.5**2)))

plt.plot(y)
plt.title("AR (1)")

ar1 = SARIMAX(y, order=(1,0,0), trend='c')
ar1_results = ar1.fit()
print(ar1_results.summary())


# In[91]:


import numpy as np

n = 1000
burn_in = 50
y = np.empty(n)
yt_last = 1.0
yt_last1 = 1.0
e = np.random.normal(0, 0.1, n+burn_in)

for i in range(n+burn_in):
    y_t = 1.0 + 0.5*yt_last + 0.4*yt_last1 + e[i]
    yt_last1 = yt_last
    yt_last = y_t
    
    if i >= burn_in:
        y[i-burn_in] = y_t

print("Mean and Var of Y: {}, {}".format(np.mean(y),np.var(y)))
print("Expected values Y: {}, {}".format(2.0,.01/(1-.5**2)))
plt.plot(y)
plt.title("AR 2")

ar2 = SARIMAX(y, order=(2,0,0), trend='c')
ar2_results = ar2.fit()
print(ar2_results.summary())


# In[92]:


import numpy as np

n = 1000
burn_in = 50
y = np.empty(n)

yt_last = 1.0
yt_last1 = 1.0
yt_last2 = 1.0
e = np.random.normal(0, 0.1, n+burn_in)

for i in range(n+burn_in):
    y_t = 1.0 + 0.5*yt_last + 0.4*yt_last1 + 0.3*yt_last2 + e[i]
    yt_last2 = yt_last1
    yt_last1 = yt_last
    yt_last = y_t
    if i >= burn_in:
        y[i-burn_in] = y_t

print("Mean and Var of Y: {}, {}".format(np.mean(y),np.var(y)))
print("Expected values Y: {}, {}".format(1.8,.01/(1-.5*.4*.3)))

plt.plot(y)
plt.title("AR (3)")

ar3 = SARIMAX(y, order=(3,0,0), trend='c')
ar3_results = ar3.fit()
print(ar3_results.summary())


# In[ ]:





# In[ ]:





# In[51]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
phi = 0.5
n = 100

y = np.zeros(n)
d = np.random.normal(0, 0.1, n)

for i in range(1, n):
    y[i] = phi * y[i-1] + d[i]

plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of AR(1) process")
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of AR(1) process")
plt.show()


# In[50]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
phi = 0.5
phi1 = 0.4
n = 100

y = np.zeros(n)
d = np.random.normal(0, 0.1, n)

for i in range(1, n):
    y[i] = phi * y[i-1] + phi1 * y[i-2] + d[i]


plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of AR(2) process")
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of AR(2) process")
plt.show()


# In[43]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
phi1 = 0.5
phi2 = 0.4
phi3 = 0.3
n = 100

y = np.zeros(n)
d = np.random.normal(0, 0.1, n)

for i in range(1, n):
    y[i] = phi1 * y[i-1] + phi2 * y[i-2] + phi3 * y[i-3] + d[i]



plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of AR(3) process")
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of AR(3) process")
plt.show()


# In[68]:


n = 1000
burn_in = 50
y = np.empty(n)
yt_last = 1.0
d = np.random.normal(0, 0.1, n+burn_in)

for i in range(1, n+burn_in):
    y_t = 1.0 + 0.5*d[i-1] + d[i]
    if i > burn_in:
        y[i-burn_in-1] = y_t
    yt_last = y_t

print(f"Mean and Var of Y: {np.mean(y)}, {np.var(y)}")
print(f"Expected values Y: {1.0}, {(1+.5**2)*.01}")

plt.figure()
plt.plot(y)
plt.title("Time Series")
plt.xlabel("Time")
plt.ylabel("Value")


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the ACF
plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of MA(1) process")

# Plot the PACF
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of MA(1) process")


# In[70]:


n = 1000
burn_in = 50
y = np.empty(n)
yt_last = 1.0
d = np.random.normal(0, 0.1, n+burn_in)

for i in range(1, n+burn_in):
    y_t = 1.0 + 0.5*d[i-1] + 0.4*d[i-2] + d[i]
    if i > burn_in:
        y[i-burn_in-1] = y_t
    yt_last = y_t

print(f"Mean and Var of Y: {np.mean(y)}, {np.var(y)}")
print(f"Expected values Y: {1.0}, {(1+.5**2)*.01}")

plt.figure()
plt.plot(y)
plt.title("Time Series")
plt.xlabel("Time")
plt.ylabel("Value")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the ACF
plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of MA(2) process")

# Plot the PACF
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of MA(2) process")

plt.show()


# In[71]:


n = 1000
burn_in = 50
y = np.empty(n)
yt_last = 1.0
d = np.random.normal(0, 0.1, n+burn_in)

for i in range(1, n+burn_in):
    y_t = 1.0 + 0.5*d[i-1] + 0.4*d[i-2] + 0.3*d[i-3] + d[i]
    if i > burn_in:
        y[i-burn_in-1] = y_t
    yt_last = y_t

print(f"Mean and Var of Y: {np.mean(y)}, {np.var(y)}")
print(f"Expected values Y: {1.0}, {(1+.5**2)*.01}")

plt.figure()
plt.plot(y)
plt.title("Time Series")
plt.xlabel("Time")
plt.ylabel("Value")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the ACF
plt.figure()
plot_acf(y, lags=30)
plt.title("ACF of MA(3) process")

# Plot the PACF
plt.figure()
plot_pacf(y, lags=30)
plt.title("PACF of MA(3) process")

plt.show()


# In[100]:


import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms

import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        plt.tight_layout()
    return
# Simulate an AR(1) process 

np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
    
_ = tsplot(x, lags=30)

# Simulate an AR(2) process

n = int(1000)
alphas = np.array([0.6, -0.5])
betas = np.array([0.])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
_ = tsplot(ar2, lags=30)

# Simulate an AR(3) process

n = int(1000)
alphas = np.array([0.6, -0.5, -0.2])
betas = np.array([0.])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
_ = tsplot(ar3, lags=30)


# In[103]:


# Simulate an MA(1) process

n = int(1000)

# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.6])

ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
_ = tsplot(ma1, lags=30)

# Simulate MA(2) process 

n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6,0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma2, lags=30)

# Simulate MA(3) process 

n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6,0.4,0.2])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=30)


# In[ ]:





# In[ ]:




