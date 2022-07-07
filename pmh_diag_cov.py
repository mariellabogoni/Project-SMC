# -*- coding: utf-8 -*-
"""
Particle Metropolis Hastings for the model with diagonal covariance matrix 
for the regression coefficients.

@author: Mariella
"""

from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import expit
import pandas as pd
import tarfile 
import scipy as sc
import os 
from matplotlib.backends.backend_pdf import PdfPages

#modules from particles
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles import collectors as col

    
##### reading the data

os.chdir('C:\\Users\\Mariella\\Documents\\sequentialMonteCarlo\\dados_tfidf')

year=list(np.arange(1980,2021,1))

T=len(year)
p=1000   #dim of beta
y=[]
S=[]     #sparse matrices of each time t
words=[]

for file in year:
        data= pd.read_csv("y_epu_year_"+str(file)+".csv")
        data.columns=["index","y"]
        y.append(np.array(data.y[0:5000]))
        
        my_tar = tarfile.open("x_sparsedata_year_"+str(file)+".tar")
        my_tar.extract('x_sparse.npz')
        
        aux = sc.io.mmread("x_sparse.npz")
        aux=aux.tocsr()
        palavras=np.random.choice(np.shape(aux)[1],size=p, replace=False)
        S.append(aux[0:5000,palavras])

        my_tar.extract("col_names.csv")
        colnames=pd.read_csv("col_names.csv")
        words.append(np.array(colnames.V1[palavras]))
     
        my_tar.close()
        
#sample size and number of words in each time t        
nt=np.repeat(0,T)
for i in range(T):  
    nt[i] = len(y[i]) 

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
    


######## Particle Metroplis Hastings

prior_dict = {'mu0': dists.MvNormal(loc = np.repeat(-8.0,p), scale = 1., cov = np.diag(np.repeat(0.01,p))),
              
              'sigmaX0' : dists.IID(dists.LogNormal(loc=-0.7, scale=0.1),p),
              
              'sigmaX'  : dists.IID(dists.LogNormal(loc=-0.7, scale=0.1),p)
               }

my_prior = dists.StructDist(prior_dict)

pmcmc = mcmc.PMMH(ssm_cls=myModel, prior=my_prior, data=y, Nx=5000, 
                  niter =1000, adaptive=True, smc_options={ "resampling" : "multinomial"})

start_time=time.time()
pmcmc.run()
end_time = time.time()
print("--- %s minutes ---" %((end_time - start_time)/60))

burnin = 0    
 
estimates_mu0=np.mean(pmcmc.chain.theta["mu0"][burnin:,],0)
estimates_sigmaX0=np.mean(pmcmc.chain.theta["sigmaX0"][burnin:,],0)
estimates_sigmaX=np.mean(pmcmc.chain.theta["sigmaX"][burnin:,],0)
log_like=pmcmc.chain.lpost[burnin:]

######## Running the filter with the estimates 

model = myModel(mu0 = estimates_mu0, sigmaX0 = estimates_sigmaX0, sigmaX=estimates_sigmaX, nt=nt)
model_boot= ssm.Bootstrap(ssm=model, data=y)     #obtaining the associated Bootstrap Feynman-Kac model
my_filter=particles.SMC(fk=model_boot,N=5000,resampling="multinomial", ESSrmin=0.5,
                        collect=[col.Moments()])

my_filter.run()
loglike = my_filter.summaries.logLts
my_filter_means = np.array([f['mean'] for f in my_filter.summaries.moments]).squeeze()
my_filter_var  = np.array([f['var'] for f in my_filter.summaries.moments]).squeeze()



with PdfPages('aplicaçao_semB_dados_tfidf.pdf') as pdf:
    plt.style.use('ggplot')
    
    plt.figure(figsize=(10,10))
    plt.scatter(range(p), estimates_mu0)
    plt.title('Estimates of mu0')
    plt.ylabel('Posterior Mean')
    pdf.savefig()  
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.scatter(range(p), estimates_sigmaX0)
    plt.title('Estimates of SigmaX0')
    plt.ylabel('Posterior Mean')
    pdf.savefig()  
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.scatter(range(p), estimates_sigmaX)
    plt.title('Estimates of SigmaX')
    plt.ylabel('Posterior Mean')
    pdf.savefig()  
    plt.close()
    
    plt.figure(figsize=(25, 25))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Beta's estimates in each year", fontsize=18, y=0.95)
    for i in range(T):
          plt.subplot(7,6,i+1)
          plt.scatter(range(p),my_filter_means[i,])
          plt.xlabel('Year:'+str(year[i]))
          plt.legend()
    pdf.savefig()  
    plt.close()