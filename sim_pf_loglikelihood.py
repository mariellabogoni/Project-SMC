# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:12:17 2022

@author: Mariella
"""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import expit
from scipy.sparse import csr_matrix


# modules from particles
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import collectors as col


###### Input data: covariate matrix  (assuming nt observation at each time t)

np.random.seed(10)
p = 500  #size of beta    
T = 41
t=range(T)
nt=np.random.poisson(lam = 500 , size=T)+1


X=[]
S=[]


for i in range(T):
    tf=np.zeros((nt[i],(p-1)))
    ones_vector = np.ones((nt[i],1))
    for j in range(nt[i]): 
        X_aux = np.random.binomial(n=5,p=0.05,size=(p-1))   #np.random.normal(loc=0.,scale=1.,size=(p-1))
        tf[j,:]=X_aux/p
    if 0 in np.sum(tf,axis=0) : 
        coll =[np.where(np.sum(tf,axis=0)==0)]
        for z in coll:
            row = np.random.choice(range(nt[i]), size = 1)
            tf[row,z]=1/p
    X.append(np.hstack((ones_vector,tf)))
    S.append(csr_matrix(X[i]))

true_sigmaX0 = np.ones(p) 
true_sigmaX = np.repeat(0.5,p)
true_mu = np.zeros(p)
x_cov=S


phi=0.6
# defining the class of the model
class myModel(ssm.StateSpaceModel):

    def PX0(self):

        return dists.MvNormal(loc=self.mu0, scale=1, cov=np.diag(self.sigmaX0))

    def PX(self, t, xpast):

        N, d = xpast.shape
        pi_prob = []
        z = []
        p = 0.9

        for i in range(d):
            pi_prob.append(
                p*np.exp((-1/(2*self.sigmaX[i]))*np.power(xpast[:, i], 2)))
            z.append(np.random.binomial(n=1, p=1-pi_prob[i]))
        z = np.array(z)

        return dists.LinearD(dists.MvNormal(loc=phi*xpast, scale=1, cov=np.diag(self.sigmaX)),
                             a=z.T)

    def PY(self, t, xpast, x):
        prob = expit(x_cov[t].dot(x.T))  # inverse of logit function

        return dists.IndepProdGen([dists.Binomial(n=1, p=prob[i, :]) for i in range(nt[t])])

model = myModel(mu0 = true_mu, sigmaX0 = true_sigmaX0 , sigmaX=true_sigmaX, nt=nt)
x_true, y = model.simulate(T)
x_true = np.array(x_true).squeeze() 


loglike = np.zeros((1000, T))

# Particle Filter

mu0=true_mu
sX=true_sigmaX
sX0=true_sigmaX0

start_time=time.time()
for i in range(1000):
        print("Filtro:", i)
        model = myModel(mu0=mu0, sigmaX0=sX0, sigmaX=sX, nt=nt)
        model_boot = ssm.Bootstrap(ssm=model, data=y)
        my_filter = particles.SMC(fk=model_boot, N=5000, resampling="multinomial", ESSrmin=0.5,
                                  collect=[col.Moments()])
        my_filter.run()

        loglike[i, :] = my_filter.summaries.logLts
   
end_time = time.time()
print("--- %s minutes ---" %((end_time - start_time)/60))

my_filter_means = np.array([f['mean'] for f in my_filter.summaries.moments]).squeeze()
my_filter_var  = np.array([f['var'] for f in my_filter.summaries.moments]).squeeze()

print("Variance:", np.var(loglike[:,T-1]))

bias = np.zeros((T,p))
bias_at_t=np.zeros(T)
for k in range(T):
    bias[k,:] = my_filter_means[k,:] - x_true[k,:]
    bias_at_t[k] = np.mean(bias[k,:])

with PdfPages('logverosimilhanca_semB1.pdf') as pdf:
    
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.boxplot(loglike[:,T-1])
    plt.title("Log-likelihood estimates over 1000 runs")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.boxplot(bias_at_t)
    plt.title("Bias")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.boxplot(bias.T)
    plt.title("Bias at each time t")
    plt.xlabel("Time")
    pdf.savefig()
    plt.close()
    
