#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt




class Cal_Var(object):
    def __init__(self):
        
        
        df1 = pd.read_csv("DailyPrices.csv")
        df2 = pd.read_csv("portfolio.csv")
        df1.set_index(["Date"], inplace=True)   
        
        portfolio_type=['A','B','C']
        tickers=[]
        Holding=[] 
        self.portfolio_value_list=[]
        for i in portfolio_type:   
            for j in range(len(df2)):
                if df2.iloc[j]['Portfolio']==i:
                    tickers.append(df2.iloc[j]['Stock'])
                    Holding.append(df2.iloc[j]['Holding'])
            
            df=df1[tickers]
            price_frist=df.loc['2/14/2022 0:00'].to_list()
            price_list=[]
            for k in range(len(Holding)):
                price=Holding[k]*price_frist[k]
                price_list.append(price)
            portfolio_value=sum(price_list) 
            self.portfolio_value_list.append({i:portfolio_value})
        self.portfolio_value_A=self.portfolio_value_list[0]['A']
        self.portfolio_value_B=self.portfolio_value_list[1]['B']
        self.portfolio_value_C=self.portfolio_value_list[2]['C']
        self.portfolio_value_ALL=(self.portfolio_value_A+self.portfolio_value_B+self.portfolio_value_C)
        self.df1=df1
        self.df2=df2
 
        
    
    def way1_A(self,port_type):
        
        tickers=[]
        Holding=[]   
        for i in range(len(self.df2)):
            if self.df2.iloc[i]['Portfolio']==port_type:
            
                tickers.append(self.df2.iloc[i]['Stock'])
                Holding.append(self.df2.iloc[i]['Holding'])
        
        weights=Holding/sum(Holding)

        
        df=self.df1[tickers]


        
        daily_returns = df.pct_change()



       
        returns_matrix = np.array(daily_returns.dropna())
   
        
        
   
        lambda_value = 0.94
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        

        
        
        portfolio_returns = np.dot(cov_matrix, weights)
    

        VaR = np.percentile(portfolio_returns, 100 * (1-lambda_value))


        return portfolio_returns,VaR
    def way2_A(self,port_type):
       
        tickers=[]
        Holding=[]   
        for i in range(len(self.df2)):
            if self.df2.iloc[i]['Portfolio']==port_type:
            
                tickers.append(self.df2.iloc[i]['Stock'])
                Holding.append(self.df2.iloc[i]['Holding'])

        
        df=self.df1[tickers]  
        
  
        daily_returns = df.pct_change()

  
        returns_matrix = np.array(daily_returns.dropna())
        
                
       
        port_returns = returns_matrix
        
   
        conf_level = 0.95
        VaR = norm.ppf(1 - conf_level, np.mean(port_returns), np.std(port_returns))
      

        return port_returns,VaR
    

       
        

    def way1_ALL(self):
        
        tickers=[]
        Holding=[]   
        for i in range(len(self.df2)):
            tickers.append(self.df2.iloc[i]['Stock'])
            Holding.append(self.df2.iloc[i]['Holding'])
        
        weights=Holding/sum(Holding)

       
        df=self.df1[tickers]


       
        daily_returns = df.pct_change()


       
        returns_matrix = np.array(daily_returns.dropna())
   
        
        
     
        
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
    
        
        
        portfolio_returns = np.dot(cov_matrix, weights)
        

        lambda_value = 0.94

        VaR = np.percentile(portfolio_returns, 100 * (1-lambda_value))


        return portfolio_returns,VaR
    
    def way2_ALL(self):
        
        tickers=[]
        Holding=[]   
        for i in range(len(self.df2)):
            
            tickers.append(self.df2.iloc[i]['Stock'])
            Holding.append(self.df2.iloc[i]['Holding'])

        
        df=self.df1[tickers]  
        
  

        daily_returns = df.pct_change()

  
        returns_matrix = np.array(daily_returns.dropna())
        
          
        port_returns = returns_matrix
        
        
        conf_level = 0.95
        VaR = norm.ppf(1 - conf_level, np.mean(port_returns), np.std(port_returns))
      

        return port_returns,VaR
if __name__=="__main__":
    CVAR=Cal_Var()
    portfolio_type=['A','B','C']
    
    
    portfolio_value_A=CVAR.portfolio_value_A
    portfolio_value_B=CVAR.portfolio_value_B
    portfolio_value_C=CVAR.portfolio_value_C
    portfolio_value_ALL=CVAR.portfolio_value_ALL
    
    
    portfolio_returns_A1,VaR_A1=CVAR.way1_A('A')
    portfolio_returns_B1,VaR_B1=CVAR.way1_A('B')
    portfolio_returns_C1,VaR_C1=CVAR.way1_A('C')
    portfolio_returns_ALL,VaR_ALL1=CVAR.way1_ALL()
    fig= plt.subplots(nrows=1, ncols=4, sharex=True)
    plt.subplot(1, 4, 1)
    plt.plot(portfolio_returns_A1, 'b.',label="A")
    plt.axhline(VaR_A1, color='r', linestyle='-', label='VaR_A')
    plt.subplot(1, 4, 2)
    plt.plot(portfolio_returns_B1, 'b.',label="B")
    plt.axhline(VaR_B1, color='r', linestyle='-', label='VaR_B')

    plt.subplot(1, 4, 3)
    plt.plot(portfolio_returns_C1, 'b.',label="C")
    plt.axhline(VaR_C1, color='r', linestyle='-', label='VaR_C')
    
    plt.subplot(1, 4, 4)
    plt.plot(portfolio_returns_ALL, 'b.',label="ALL")
    plt.axhline(VaR_ALL1, color='r', linestyle='-', label='VaR_ALL')
    
    
    fig, ax = plt.subplots()
    portfolio_returns_A2,VaR_A2=CVAR.way2_A('A')
    portfolio_returns_B2,VaR_B2=CVAR.way2_A('B')
    portfolio_returns_C2,VaR_C2=CVAR.way2_A('C')
    portfolio_returns_ALL,VaR_ALL2=CVAR.way2_ALL()
    xA = np.linspace(np.min(portfolio_returns_A2), np.max(portfolio_returns_A2), 100)
    yA = norm.pdf(xA, np.mean(portfolio_returns_A2), np.std(portfolio_returns_A2))
    xB = np.linspace(np.min(portfolio_returns_B2), np.max(portfolio_returns_B2), 100)
    yB = norm.pdf(xB, np.mean(portfolio_returns_B2), np.std(portfolio_returns_B2))
    xC = np.linspace(np.min(portfolio_returns_C2), np.max(portfolio_returns_C2), 100)
    yC = norm.pdf(xC, np.mean(portfolio_returns_C2), np.std(portfolio_returns_C2))
    xALL = np.linspace(np.min(portfolio_returns_ALL), np.max(portfolio_returns_ALL), 100)
    yALL = norm.pdf(xC, np.mean(portfolio_returns_ALL), np.std(portfolio_returns_ALL))
    ax.plot(xA, yA, label='A')
    ax.plot(xB, yB, label='B')
    ax.plot(xC, yC, label='C')
    ax.plot(xALL, yALL, label='ALL')
    
    ax.axvline(x=VaR_A2, linestyle='--', color='g', label='VaR_A')
    ax.axvline(x=VaR_B2, linestyle='--', color='r', label='VaR_B')
    ax.axvline(x=VaR_C2, linestyle='--', color='y', label='VaR_C')
    ax.axvline(x=VaR_ALL2, linestyle='--', color='b', label='VaR_ALL')
    ax.legend()
    plt.show()
    
    print("VaR Using Exponentially weighted covariance :")
    print(VaR_A1*portfolio_value_A)
    print(VaR_B1*portfolio_value_B)
    print(VaR_C1*portfolio_value_C)
    print(VaR_ALL1*portfolio_value_ALL)
    print("VAR Using normal distribution:")
    print(-(VaR_A2*portfolio_value_A))
    print(-(VaR_B2*portfolio_value_B))
    print(-(VaR_C2*portfolio_value_C))
    print(-(VaR_ALL2*portfolio_value_ALL))


# In[ ]:





# In[17]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


df1 = pd.read_csv("DailyPrices.csv")
df2 = pd.read_csv("portfolio.csv")


df1.set_index(["Date"], inplace=True)       


def get_VaR(df1,df2,portfolio,confidence_level=0.95):
    if portfolio=="ALL":
        tickers=list(df2['Stock'])
        Holding=list(df2['Holding']) 
        
        weights=np.array([(a/sum(Holding)) for a in Holding])

        df=df1[tickers]
        price_frist=df1.loc['2/14/2022 0:00'].to_list()
        price_list=[]
        for i in range(len(Holding)):
            price=Holding[i]*price_frist[i]
            price_list.append(price)
        portfolio_value=sum(price_list)
        
    else:
        tickers=[]
        Holding=[]   
        for i in range(len(df2)):
            if df2.iloc[i]['Portfolio']==portfolio:
            
                tickers.append(df2.iloc[i]['Stock'])
                Holding.append(df2.iloc[i]['Holding'])
        
        weights=Holding/sum(Holding)

        df=df1[tickers]
        price_frist=df.loc['2/14/2022 0:00'].to_list()
        price_list=[]
        for i in range(len(Holding)):
            price=Holding[i]*price_frist[i]
            price_list.append(price)
        portfolio_value=sum(price_list)    
    
    daily_returns = df.pct_change()

    
    returns_matrix = np.array(daily_returns.dropna())

    
    mu = returns_matrix.mean()
    sigma = returns_matrix.std()
    
    
    lambda_value = 0.94
    cov_matrix = np.cov(returns_matrix, rowvar=False)
    ewma_cov_matrix = lambda_value * cov_matrix + (1 - lambda_value) * np.outer(returns_matrix[-1], returns_matrix[-1])

    
    portfolio_mean_return = np.sum(weights * np.mean(returns_matrix, axis=0))
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(ewma_cov_matrix, weights)))


    
    Z_score = norm.ppf(confidence_level)

    portfolio_value = 1000000  
    VaR = (portfolio_mean_return * portfolio_value + portfolio_std_dev * portfolio_value * Z_score)
    
    print("portfolio:"+portfolio)
    print(f"VaR Using Exponentially weighted variance: {VaR:.10f}")
    return VaR,mu,sigma,returns_matrix





def draw(mu,sigma,returns_matrix):
    
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma))
    
    
    plt.hist(returns_matrix, bins=30, density=True)
    
     
    plt.xlabel('Daily returns')
    plt.ylabel('Frequency')
    plt.title('Distribution of daily returns')
    
    
    
    plt.show()
    

if __name__=="__main__":
    
    VaR_A,mu_A,sigma_A,returns_matrix_A=get_VaR(df1=df1,df2=df2,portfolio="A",confidence_level=0.95)
    VaR_B,mu_B,sigma_B,returns_matrix_B=get_VaR(df1=df1,df2=df2,portfolio="B",confidence_level=0.95)
    VaR_C,mu_C,sigma_C,returns_matrix_C=get_VaR(df1=df1,df2=df2,portfolio="C",confidence_level=0.95)
    VaR_All,mu_All,sigma_All,returns_matrix_ALL=get_VaR(df1=df1,df2=df2,portfolio="ALL",confidence_level=0.95)

    #draw(mu,sigma)


# In[ ]:




