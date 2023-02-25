#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t

data = pd.read_csv('problem1.csv')

mu, std = norm.fit(data['x'])

var_5percent = norm.ppf(0.05, mu, std)
print('VaR at 5% level:', -var_5percent)

es_5percent = -norm.pdf(norm.ppf(0.05)) * std / 0.05
print('Expected Shortfall at 5% level:', -es_5percent)


def VaR_ES(x, alpha=0.05, dist=None):
    xs = sorted(x)
    n = alpha*len(xs)
    iup = int(round(n))
    idn = int(n)
    VaR = (xs[iup] + xs[idn])/2
    if dist is None:
        ES = -sum([x for x in xs[0:idn]])/idn
    else:
        dist_obj = t(*dist)
        ES = -dist_obj.expect(lambda x: x, lb=-float('inf'), ub=-VaR)/alpha
    return -VaR, -ES

data = pd.read_csv('problem1.csv')

gen_t_dist = t.fit(data['x'])

genTVaR, genTES = VaR_ES(data['x'], alpha=0.05, dist=gen_t_dist)

print('\nGeneralized T Distribution:')
print('VaR:', genTVaR)
print('ES:', -genTES)

# Create a histogram of the data
plt.hist(data['x'], bins=20, density=True, alpha=0.6)

# Create the PDFs for the Normal distribution and the Generalized T distribution
x = np.linspace(-0.3, 0.3, 1000)
norm_pdf = norm.pdf(x, mu, std)
gen_t_pdf = t.pdf(x, *gen_t_dist)

# Plot the PDFs on the same plot
plt.plot(x, norm_pdf, 'r-', label='Normal')
plt.plot(x, gen_t_pdf, 'b-', label='Generalized T')

# Add vertical lines at the VaR and ES levels for each distribution
plt.axvline(var_5percent, color='r', linestyle='--', label=f'Normal VaR={-var_5percent:.4f}')
plt.axvline(es_5percent, color='g', linestyle='-.', label=f'Normal ES={-es_5percent:.4f}')
plt.axvline(-genTVaR, color='b', linestyle='--', label=f'Gen T VaR={genTVaR:.4f}')
plt.axvline(genTES, color='y', linestyle='-.', label=f'Gen T ES={-genTES:.4f}')

# Set plot title and axis labels
plt.title('Histogram and PDFs of the Data')
plt.xlabel('x')
plt.ylabel('Density')

# Show the legend and plot
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




