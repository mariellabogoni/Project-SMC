# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:44:55 2021

@author: Mariella
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns     #for plotting densities
import scipy.stats 


#modules from particles
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles import collectors as col


#defining the class of the model
class myModel(ssm.StateSpaceModel):
    def PX0(self): 
        return dists.IndepProd(dists.Normal(loc=self.mu0[0], scale=self.sigmaX), #beta0
                               dists.Normal(loc=self.mu0[1], scale=self.sigmaX), #beta1
                               dists.Normal(loc=self.mu0[2], scale=self.sigmaX), #beta2
                               dists.Normal(loc=self.mu0[3], scale=self.sigmaX)) #beta3
    def PX(self, t, xpast):
        return dists.IndepProd(dists.Normal(loc=xpast[:, 0], scale=self.sigmaX), #beta0
                               dists.Normal(loc=xpast[:, 1], scale=self.sigmaX), #beta1
                               dists.Normal(loc=xpast[:, 2], scale=self.sigmaX), #beta2
                               dists.Normal(loc=xpast[:, 3], scale=self.sigmaX)) #beta3    #distribution of X_t|X_t-1
    def PY(self, t, xpast, x):
        prob = 1.0/(1+np.exp(-np.dot(x,self.x_cov[t,])))
        return dists.Binomial(n=1, p=prob)
        
        
###### Input data: covariate matrix  (assuming only one observation at each time t)

p = 4     #size of beta    
T=500
t=range(T)

X_aux=np.zeros((T,(p-1)))
for i in range(T):
    X_aux[i,:] = np.array([np.random.normal(loc=0.,scale=1.,size=(p-1))])
ones_vector = np.ones((T,1))
X = np.hstack((ones_vector,X_aux))



###### Simulating the model 
# x_true = np.zeros((T,p))
# y = np.zeros((1,T))

# for t in range(T): 
#     x_true[t,:] = scipy.stats.multivariate_normal.rvs([0,0,0,0],[1,1,1,1]) if t==0 else scipy.stats.multivariate_normal.rvs(x_true[t-1,:],[1,1,1,1]) 
    
#     prob = 1.0/(1+np.exp(-np.dot(X[t,:],x_true[t,:])))
#     y[:,t] = np.random.binomial(n=1,p=prob,size=1)
  
  
model = myModel(mu0 = np.array([0,0,0,0,0]), sigmaX = 1., x_cov= X)
x_true, y = model.simulate(T) 
x_true = np.array(x_true).squeeze()

model_boot= ssm.Bootstrap(ssm=model, data=y)     #obtaining the associated Bootstrap Feynman-Kac model
my_filter=particles.SMC(fk=model_boot,N=4000,resampling="multinomial", ESSrmin=0.5,
                        collect=[col.Moments()])
my_filter.run()


loglike = my_filter.summaries.logLts
my_filter_means = np.array([f['mean'] for f in my_filter.summaries.moments]).squeeze()
my_filter_var  = np.array([f['var'] for f in my_filter.summaries.moments]).squeeze()



from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('logistic_model_case1.pdf') as pdf:
    plt.style.use("ggplot")
    plt.figure()
    plt.scatter(t,y, c="blue")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.title('Observations')
    pdf.savefig()  
    plt.close()
    
    plt.figure()
    plt.plot(t, x_true[:, 0], label = "Beta0")
    plt.plot(t, x_true[:, 1], label = "Beta1")
    plt.plot(t, x_true[:, 2], label = "Beta2")
    plt.plot(t, x_true[:, 3], label = "Beta3")
    plt.legend()
    plt.title('Betas')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    plt.plot(t, my_filter_means[:, 0], label = "Beta0")
    plt.plot(t, my_filter_means[:, 1], label = "Beta1")
    plt.plot(t, my_filter_means[:, 2], label = "Beta2")
    plt.plot(t, my_filter_means[:, 3], label = "Beta3")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    plt.plot(t, my_filter_means[:, 0], label = "Filtered Beta0")
    plt.plot(t, x_true[:, 0], label = "Beta0")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    plt.plot(t, my_filter_means[:, 1], label = "Filtered Beta1")
    plt.plot(t, x_true[:, 1], label = "Beta1")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    plt.plot(t, my_filter_means[:, 2], label = "Filtered Beta2")
    plt.plot(t, x_true[:, 2], label = " Beta2")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    plt.plot(t, my_filter_means[:, 3], label = "Filtered Beta3")
    plt.plot(t, x_true[:, 3], label = " Beta3")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    sns.distplot(my_filter.summaries.logLts, hist=False)
    plt.title('Density of Log-Likelihood')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Density')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()