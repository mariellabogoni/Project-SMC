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
        return dists.MvNormal(loc = self.mu0, scale=1,cov = self.B.dot(self.B.T)+np.diag(self.sigmaX) )
             
    def PX(self, t, xpast):
        N,d = xpast.shape
        pi_prob=[]
        z=[]
        p=0.9
        for i in range(d): 
            pi_prob.append(p*np.exp((-1/(2*self.sigmaX[i]))*np.power(xpast[:, i],2)))
            z.append(np.random.binomial(n=1,p=1-pi_prob[i])) 
        z=np.array(z)
        return dists.LinearD(dists.MvNormal(loc = xpast, scale=1, cov = self.B.dot(self.B.T)+np.diag(self.sigmaX)), a = z.T)
    
    def PY(self, t, xpast, x):
        prob=1.0/(1+ np.exp(-x_cov[t].dot(x.T)))
        
        return dists.IndepProd( dists.Binomial(n=1, p=prob[0]),
                                dists.Binomial(n=1, p=prob[1]),
                                dists.Binomial(n=1, p=prob[2]),
                                dists.Binomial(n=1, p=prob[3]),
                                dists.Binomial(n=1, p=prob[4]))
        
        
###### Input data: covariate matrix  (assuming nt observation at each time t)

p = 8     #size of beta    
T=500
t=range(T)
nt=np.repeat(5,T)   #number of observations at each time t

X=[]
S=[]

for i in range(T):
    X_aux=np.zeros((nt[i],(p-1)))  #allocating the matrix
    ones_vector = np.ones((nt[i],1))
    for j in range(nt[i]): X_aux[j,:] = np.random.binomial(n=1,p=0.3,size=(p-1))   #np.random.normal(loc=0.,scale=1.,size=(p-1))
    X.append(np.hstack((ones_vector,X_aux)))
    S.append(csr_matrix(np.hstack((ones_vector,X_aux))))


###### Simulating the model 
  
k=5 #k<<p
B=np.zeros((p,k))
for j in range(p): B[j,:] = np.random.normal(loc = 0.0, scale = 1., size = k)
B=np.tril(B,k=0)   #get the lower triangular version of B

sigmaX = np.ones(p) 
mu = np.zeros(p)

model = myModel(mu0 = mu, sigmaX = sigmaX , B=B, x_cov= S, nt=nt)
x_true, y = model.simulate(T) 
x_true = np.array(x_true).squeeze()


###### Running the Filter
model_boot= ssm.Bootstrap(ssm=model, data=y)     #obtaining the associated Bootstrap Feynman-Kac model
my_filter=particles.SMC(fk=model_boot,N=5000,resampling="multinomial", ESSrmin=0.5,
                        collect=[col.Moments()])
my_filter.run()


loglike = my_filter.summaries.logLts
my_filter_means = np.array([f['mean'] for f in my_filter.summaries.moments]).squeeze()
my_filter_var  = np.array([f['var'] for f in my_filter.summaries.moments]).squeeze()



from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('logistic_model_case1.pdf') as pdf:
    plt.figure()
    plt.plot(t, x_true[:, 0], label = "Beta0")
    plt.plot(t, x_true[:, 1], label = "Beta1")
    plt.plot(t, x_true[:, 2], label = "Beta2")
    plt.plot(t, x_true[:, 3], label = "Beta3")
    plt.plot(t, x_true[:, 4], label = "Beta4")
    plt.plot(t, x_true[:, 5], label = "Beta5")
    plt.plot(t, x_true[:, 6], label = "Beta6")
    plt.plot(t, x_true[:, 7], label = "Beta7")
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
    plt.plot(t, my_filter_means[:, 4], label = "Beta4")
    plt.plot(t, my_filter_means[:, 5], label = "Beta5")
    plt.plot(t, my_filter_means[:, 6], label = "Beta6")
    plt.plot(t, my_filter_means[:, 7], label = "Beta7")
    plt.legend()
    plt.title('Particle Filter with Adaptative Resampling')
    plt.xlabel("Time")
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    plt.figure()
    fig, axs = plt.subplots(4,2)
    fig.suptitle('True and Filtered betas')
    axs[0,0].plot(t, my_filter_means[:, 0], label = "Filtered Beta0")
    axs[0,0].plot(t, x_true[:, 0], label = "Beta0")
    axs[0,0].set_title("Beta0")

    axs[0,1].plot(t, my_filter_means[:, 1], label = "Filtered Beta1")
    axs[0,1].plot(t, x_true[:, 1], label = "Beta1")
    axs[0,1].set_title("Beta1")

    axs[1,0].plot(t, my_filter_means[:, 2], label = "Filtered Beta2")
    axs[1,0].plot(t, x_true[:, 2], label = "Beta2")
    axs[1,0].set_title("Beta2")

    axs[1,1].plot(t, my_filter_means[:, 3], label = "Filtered Beta3")
    axs[1,1].plot(t, x_true[:, 3], label = "Beta3")
    axs[1,1].set_title("Beta3")
    #fig.tight_layout()
    
    axs[2,0].plot(t, my_filter_means[:, 4], label = "Filtered Beta0")
    axs[2,0].plot(t, x_true[:, 4], label = "Beta0")
    axs[2,0].set_title("Beta4")

    axs[2,1].plot(t, my_filter_means[:, 5], label = "Filtered Beta1")
    axs[2,1].plot(t, x_true[:, 5], label = "Beta1")
    axs[2,1].set_title("Beta5")

    axs[3,0].plot(t, my_filter_means[:, 6], label = "Filtered Beta2")
    axs[3,0].plot(t, x_true[:, 6], label = "Beta2")
    axs[3,0].set_title("Beta6")

    axs[3,1].plot(t, my_filter_means[:, 7], label = "Filtered Beta3")
    axs[3,1].plot(t, x_true[:, 7], label = "Beta3")
    axs[3,1].set_title("Beta7")
    fig.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    sns.distplot(my_filter.summaries.logLts, hist=False)
    plt.title('Density of Log-Likelihood')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Density')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
