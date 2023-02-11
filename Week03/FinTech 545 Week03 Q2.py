#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = a.copy()

    #calculate the correlation matrix if we got a covariance
    if (np.abs(np.diag(out) - 1) < 1e-10).sum() != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(np.dot(invSD, out), invSD)
    else:
        invSD = None

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.diag(1.0 / (vals * np.square(np.linalg.norm(vecs, axis=0))))
    T = np.sqrt(T)
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)
    return out


def chol_psd(root, a):
    n = a.shape[0]
    root[:] = 0.0
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        temp = a[j, j] - s
        if temp < 1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            root[j+1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir


def nearest_psd(a, epsilon=0.0):
    n = a.shape[0]
    s, u = np.linalg.eigh(a)
    s = np.maximum(s, epsilon)
    s = np.diag(s)
    b = np.dot(u, np.dot(s, u.T))
    return b


# In[43]:


import numpy as np

n = 
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1.0)
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357


near_psd_matrix = near_psd(sigma,epsilon=1e-8)
higham_psd_matrix = nearest_psd(sigma,epsilon=1e-8)

def is_psd(matrix):
    eigvals = np.linalg.eigvals(matrix)
    return all(eigval >= 0 for eigval in eigvals)

if is_psd(near_psd_matrix):
    print("The matrix after near psd is PSD.")
else:
    print("The matrix after near psd is not PSD.")
    
if is_psd(higham_psd_matrix):
    print("The matrix after higham is PSD.")
else:
    print("The matrix after higham is not PSD.")
    

frobenius_norm = np.linalg.norm(near_psd_matrix, 'fro')
print("Frobenius norm of the near psd matrix: ", frobenius_norm)
frobenius_norm = np.linalg.norm(higham_psd_matrix, 'fro')
print("Frobenius norm of the higham psd matrix: ", frobenius_norm)

import time

start = time.time()
near_psd_result = nearest_psd(sigma)
end = time.time()
print("Time taken for near_psd: ", end - start)

start = time.time()
higham_result = nearest_psd(sigma)
end = time.time()
print("Time taken for higham nearest_psd: ", end - start)


# In[ ]:





# In[ ]:





# In[ ]:




