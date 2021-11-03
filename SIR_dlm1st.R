setwd("~/Documentos/sequentialMonteCarlo")
#############################################
#
# 1st ORDER DYNAMIC LINEAR MODEL 
#
#  For t=1,...,T,
#
#        y(t) ~ N(x(t);sdy)                          
#        x(t) ~ N(x(t-1);sdx)                          
#
#  Prior distributions:
#
#       x(0) ~ N(m0,sd0)
#
#############################################
library(gridExtra)
library(ggplot2)
library(cubature)
set.seed(1234)

T=500  
x_true<-NULL
y_true<-NULL
m0<- 0
sd0<- 1
sdx<-1
sdy<-5

##############################################
#            Generating the data
########################  ######################

x_true[1]<-rnorm(1, mean = m0, sd = sd0)
y_true[1]<-rnorm(1, mean = x_true[1], sd = sdy)

for (i in 2:T) {
  x_true[i]<-rnorm(1, mean = x_true[i-1],sd = sdx)
  y_true[i]<-rnorm(1, mean = x_true[i], sd = sdy)
}


ggplot(data = as.data.frame(cbind(y=y_true,t=c(1:T), x=x_true)))+
  geom_line(mapping = aes(x=t,y=x))+
  geom_point(mapping = aes(x=t,y=y), color = "red") 

############################################
#      Sequential Important Sampling with     
#                 Resampling  
############################################    

N<-1000
X<-matrix(0, nrow = T+1, ncol = N)  
w<-matrix(0, nrow = T+1, ncol = N)
W<-matrix(0, nrow = T+1, ncol = N)
X_hat<-NULL
w_hat<-NULL
ESS<-NULL

#For time t=0 
for(i in 1:N){ 
    X[1,i]<-rnorm(1, m0, sd0)
    w[1,i]<-dnorm(y_true[1], X[1,i], sdy)#1/N
}
W[1,]<-w[1,]/sum(w[1,])


#For time t>0 
for (t in 2:(T+1)) {
  for (i in 1:N ){
    #new_sd <- sqrt(sdy^(-2) + sdx^(-2))
    #new_mean  <- (new_sd^2)*(sdy^(-2)*y_true[t-1]+sdx^(-2)*X[t-1,i]) 
    X[t,i]<- rnorm(1,X[t-1,i], sdx)   #proposal distributin q(X_t|X_t-1) = f(X_t|X_t-1) 
    w[t,i]<- dnorm(y_true[t-1],X[t-1,i],sdy,log=F)  #importance weight
  }
  W[t,]<-w[t,]/sum(w[t,])
  ESS[t]<-1/sum(W[t,]^2)
  if(ESS[t]<N/2){
      index<-sample(1:N, size = N, replace = T, prob = W[t,])
      X[t,]<-X[t,index]
      W[t,]<-rep(1/N,N)
  }
  X_hat[t-1]<-sum(X[t,]*W[t,])     #filtered state estimate. 
  w_hat[t-1]<-mean(w[t,])
} 
  
pdf("output2.pdf", width = 10, height = 10)
ggplot(data = as.data.frame(cbind(y=X_hat,t=c(1:T), x=x_true)))+
  geom_line(mapping = aes(x=t,y=y, colour = "EST"))+
  geom_line(mapping = aes(x=t,y=x, colour = "TRUE"))+
  scale_colour_manual(values = c("EST"="red","TRUE"="black"),name = '', 
                      labels = c("Estimated States","True States"))
dev.off()




