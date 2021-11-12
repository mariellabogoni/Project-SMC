setwd("~/Documentos/sequentialMonteCarlo")
#############################################
#
# 1st ORDER DYNAMIC LINEAR MODEL 
#
#  For t=1,...,T,
#
#        y(t) ~ N(x(t);Vt)                          
#        x(t) ~ N(x(t-1);Wt)                          
#
#  Prior distributions:
#
#       x(0) ~ N(m0,c0)
#
#############################################
library(gridExtra)
library(ggplot2)
library(cubature)
set.seed(1234)

T=500  
x_true<-NULL
y<-NULL
m0<- 0
c0<- 1
W<-1
V<-5

##############################################
#            Generating the data
########################  ######################

x_true[1]<-rnorm(1, mean = m0, sd = c0)
y[1]<-rnorm(1, mean = x_true[1], sd = V)

for (i in 2:T) {
  x_true[i]<-rnorm(1, mean = x_true[i-1],sd = W)
  y[i]<-rnorm(1, mean = x_true[i], sd = V)
}


ggplot(data = as.data.frame(cbind(y=y_true,t=c(1:T), x=x_true)))+
  geom_line(mapping = aes(x=t,y=x))+
  geom_point(mapping = aes(x=t,y=y), color = "red") 


###############################################
#            Kalman Filter                    
###############################################
a<-NULL     #mean of dist Xt|y1:t-1
R<-NULL     #var of dist Xt|y1:t-1
f<-NULL     #mean of dist Yt|yt-1
Q<-NULL     #var of dist Yt|yt-1
m<-NULL     #mean of dist Xt|y1:t
C<-NULL     #var of dist Xt|y1:t
loglik<-NULL

#time t=1
a[1]<-0
R[1]<-10
f[1]<-a[1]
Q[1]<-R[1] + V
m[1]<-a[1] + (R[1]/Q[1])*(y[1]-f[1])
C[1]<-R[1] - (R[1]^2)/Q[1]
loglik[1]<- dnorm(y[1], f[1], sqrt(Q[1]), log = TRUE)

for (t in 2:T) {
   a[t]<-m[t-1]
   R[t]<-C[t-1] + W
   f[t]<-a[t]
   Q[t]<-R[t] + V
   m[t]<-a[t] + (R[t]/Q[t])*(y[t]-f[t])
   C[t]<-R[t] - (R[t]^2)/Q[t]
    loglik[t]<- loglik[t-1] + dnorm(y[t], f[t], sqrt(Q[t]), log = TRUE)
}

#plotting the filtered states
ggplot(data = as.data.frame(cbind(t=c(1:T), x=x_true, y=m)))+
  geom_line(mapping = aes(x=t,y=y, colour = "EST"))+
  geom_line(mapping = aes(x=t,y=x, colour = "TRUE"))+
  scale_colour_manual(values = c("EST"="red","TRUE"="black"),name = '', 
                      labels = c("Estimated States","True States"))




