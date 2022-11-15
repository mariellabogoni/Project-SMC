# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns     
from scipy.sparse import csr_matrix
import time
from scipy.special import expit
import joypy
import pandas as pd


#modules from particles
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles import collectors as col
import os


###### Input data: covariate matrix  (assuming nt observation at each time t)

np.random.seed(10)
p = 100  #size of beta    
T = 10
t=range(T)
nt=np.random.poisson(lam = 500 , size=T)+1


X=[]
S=[]


for i in range(T):
    X_aux=np.zeros((nt[i],(p-1)))  
    tf=np.zeros((nt[i],(p-1)))
    ones_vector = np.ones((nt[i],1))
    for j in range(nt[i]): 
        X_aux[j,:] = np.random.binomial(n=5,p=0.05,size=(p-1))   
        tf[j,:]=X_aux[j,:]/p
    if 0 in np.sum(X_aux,axis=0) : 
        coll =[np.where(np.sum(X_aux,axis=0)==0)]
        for z in coll:
            row = np.random.choice(range(nt[i]), size = 1)
            X_aux[row,z]=1
            tf[row,z]=X_aux[row,z]/p
    X.append(np.hstack((ones_vector,tf)))
    S.append(csr_matrix(X[i]))

true_sigmaX0 = np.ones(p) 
true_sigmaX = np.repeat(0.5,p)
true_mu = np.zeros(p)

x_cov=S


phi=0.6
#defining the class of the model
class myModel(ssm.StateSpaceModel):
    
    def PX0(self): 

        return dists.MvNormal(loc = self.mu0, scale=1,cov= np.diag(self.sigmaX0) )
             
    def PX(self, t, xpast):
        
        N,d = xpast.shape
        pi_prob=[]
        z=[]
        p=0.9
       
        for i in range(d): 
            pi_prob.append(p*np.exp((-1/(2*self.sigmaX[i]))*np.power(xpast[:, i],2)))
            z.append(np.random.binomial(n=1,p=1-pi_prob[i])) 
        z=np.array(z)
        
        return dists.LinearD(dists.MvNormal(loc = phi*xpast, scale=1, cov = np.diag(self.sigmaX)),
                             a = z.T)
    
    def PY(self, t, xpast, x):
        prob = expit(x_cov[t].dot(x.T))    #inverse of logit function
        return dists.IndepProdGen([dists.Binomial(n=1, p=prob[i,:]) for i in range(nt[t])])
    
model = myModel(mu0 = true_mu, sigmaX0 = true_sigmaX0 , sigmaX=true_sigmaX, nt=nt)
x_true, y = model.simulate(T)
x_true = np.array(x_true).squeeze()    


######## Particle Metroplis Hastings

prior_dict = {'mu0': dists.MvNormal(loc = np.repeat(2,p), scale = 1., cov = np.diag(np.repeat(1,p))),
              
              'sigmaX0' : dists.IID(dists.LogNormal(loc=-0.1, scale=0.1),p), 
              
              'sigmaX'  : dists.IID(dists.LogNormal(loc=-0.7, scale=0.1),p)  
             }


my_prior = dists.StructDist(prior_dict)

pmcmc = mcmc.PMMH(ssm_cls=myModel, prior=my_prior, data=y, Nx=3000, 
                  niter =10000, scale =1, adaptive=True, smc_options={ "resampling" : "multinomial"})
pmcmc.run()

burnin = 4000  
 
estimates_mu0=np.mean(pmcmc.chain.theta["mu0"][burnin:,],0)
estimates_sigmaX0=np.mean(pmcmc.chain.theta["sigmaX0"][burnin:,],0)
estimates_sigmaX=np.mean(pmcmc.chain.theta["sigmaX"][burnin:,],0)
log_like=pmcmc.chain.lpost[burnin:]


erro_mu=np.mean(estimates_mu0 - true_mu)
erro_sigmaX=np.mean(estimates_sigmaX - true_sigmaX)
erro_sigmaX0=np.mean(estimates_sigmaX0 - true_sigmaX0)


######## Running the filter with the estimates 


model = myModel(mu0 = estimates_mu0, sigmaX0 = estimates_sigmaX0, sigmaX=estimates_sigmaX, nt=nt)
model_boot= ssm.Bootstrap(ssm=model, data=y)     #obtaining the associated Bootstrap Feynman-Kac model
my_filter=particles.SMC(fk=model_boot,N=3000,resampling="multinomial", ESSrmin=0.5,
                        collect=[col.Moments()])

my_filter.run()
loglike = my_filter.summaries.logLts
my_filter_means = np.array([f['mean'] for f in my_filter.summaries.moments]).squeeze()
my_filter_var  = np.array([f['var'] for f in my_filter.summaries.moments]).squeeze()


