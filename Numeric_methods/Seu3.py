import numpy as np
#from numpy import linalg as la
from matplotlib import pyplot as pp
from scipy import linalg as la
from itertools import cycle
import matplotlib.patches as mps


print "wanna n"
n=input()
def  ND(n):#schitat
    m=[]
    i=0.0
    for _ in range(0,n):
        i=(input())
        m.append(i)
    return m

print "Wanna m"
M=np.diag(ND(n))
print "wanna b"
B=np.diag(ND(n))
print "Wanna k"
k=ND(n)

K=np.diag(k)
k.pop(0)
k.append(0)
K=K+np.diag(k)
k.pop()
K=K-np.diag(k,1)
K=K-np.diag(k,-1)

tau = 1

print "std x0 and xt0?(1/0)" #y0 consists of n values (x0) and n values of xt0
d=input()
if d==0:
    y0=np.array(ND(2*n))
if d==1:
    y0=np.array(np.diag(np.ones((2*n,2*n))))

x1=[]
for i in range(0,n):
    x1.append(y0[i]+y0[i+n]*tau)

X=[]
xnm1=y0[:n]
xn=x1
X.append(xnm1)
X.append(xn)
for n in range(0,100):
    ch1=np.linalg.inv((M+B))
    ch21=np.dot(-K,np.dot(xn,tau*tau))
    ch22=np.dot(M,(np.dot(2,xn)-xnm1))
    ch23=np.dot(B,xnm1)
    #xnp1=np.dot(np.linalg.inv((M+B)),((np.dot(-K,xn*tau*tau)+np.dot(M,(2*xn-xnm1))+np.dot(B,(xnm1)))))
    xnp1=np.dot(ch1,ch21+ch22+ch23)
    X.append(xnp1)
    xnm1=xn
    xn=xnp1
