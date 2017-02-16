import numpy as np
#from numpy import linalg as la
from matplotlib import pyplot as pp
from scipy import linalg as la
from itertools import cycle
import matplotlib.patches as mps
from scipy.interpolate import interp1d

#Warning! k and b *-1 to obtain oscillations
##COLOURS="
##d={a:a**2 a in range
print "wanna n"
n=input()
##eps=0.0
tmax=100.0
tau=0.3
raznrange=int(tmax/tau)
#xt0=[1.0,-1.0]
#x0=[0.0,0.0]

print "hey"
##print "wanna eps"
##eps=(input())
def  ND(n):#schitat
    m=[]
    i=0.0
    for _ in range(0,n):
        i=(input())
        m.append(i)
    return m

print "std y0?(1/0)"
d=input()
if d==0:
    y0=np.array(ND(2*n))
if d==1:
    y0=np.array(np.diag(np.ones((2*n,2*n))))
print "Wanna m"
M=np.diag(ND(n))
print "wanna b"
B=np.diag(ND(n))
print "Wanna k"
k=ND(n)
print "Which coords to draw?(0..n)"
a=-2
CTD=[]
while a!= -1:
    if a==-9:
        for r in range (0,n):
            CTD.append(r)
        break
    if 0<=int(a)<n:
        CTD.append(int(a))
    else:
        print "no such coord"
    a=input()
#CTD.pop(0)
print "Which v to draw?(0..n)"
a=-2
VTD=[]
while a!= -1:
    if a==-9:
        for r in range (0,n):
            CTD.append(n+r)
        break
    if 0<=a<n:
        CTD.append(n+a)
    else:
        print "no such coord"
    a=int(input())
    
print "Calculating..."
#k=-k
k2 = k[:]

#k = [-i for i in k2]

Ko=np.diag(k)
k.pop(0)
k.append(0)
Ko=Ko+np.diag(k)
k.pop()
Ko=Ko-np.diag(k,1)
Ko=Ko-np.diag(k,-1)
#B=-B
Mi=la.inv(M)#M^-1
E =np.diag(np.diag(np.ones((n,n))))#beru diag i iz nee diag
A1=np.concatenate((-np.dot(Mi,B),-np.dot(Mi,Ko)),axis=1)
A2=np.concatenate((E,np.zeros((n,n))),axis=1)
A=np.concatenate((A1,A2),axis=0)

### (n+1)!/A^(n+1) = 1+At/((1-A^2t^2)e) -> n
#Ch=(1+A*t)/((1-(A**2)*(t**2))*eps)

def y(t,i):
   #return np.dot(la.expm(A*t),y0)[i]#can be expm3(A, ordrer Taylor)
  #return np.dot(la.expm3(A*t,n),y0)[i]
    return np.dot(la.expm2(A*t),y0)[i]
#t=np.arange(0,10)
t=0.0

t = [i for i in range(0,int(tmax), 1)]

#To gnuplot plot "Tplt.txt" using 1:2, ".." using 1:3 ...
# cd with '..' !

##toplot=open("Tplt.txt",'w')
##for j in t:
##    toplot.write(str(j))
##    toplot.write(' ')
##    for i in CTD:
##        wrt=y(j,i)
##        toplot.write(str(wrt))
##        toplot.write(' ')
##    toplot.write("\n")

#end tognuplot
#colors = cycle(["#FF00DD", "#DDDDD0", "#FFFF34","k","#BBBBFF","m","c"])

#ZADACHA3ZADACHA3ZADACHA3ZADACHA3ZADACHA3ZADACHA3ZADACHA3
k = [i for i in k2]

K=np.diag(k)
k.pop(0)
k.append(0)
K=K+np.diag(k)
k.pop()
K=K-np.diag(k,1)
K=K-np.diag(k,-1)
B=B

x1=[]
for i in range(0,n):
    x1.append(y0[i]-abs(y0[i+n])*tau)

X=[]
xnm1=y0[:n]
#xnm1=[-i for i in xnm1]
xn=[i for i in x1]

X.append(xnm1)
X.append(xn)

t2=[0,tau]


tek=tau+tau
tauk=tau*tau
ch1=np.linalg.inv((M+B))
for n in range(2,(raznrange)):
    t2.append(tek)
    tek=tek+tau
    ch21=np.dot(-K,np.dot(xn,tauk))
    ch22=np.dot(M,(np.dot(2,xn)-xnm1))
    ch23=np.dot(B,xnm1)
    #B=-B??
    #xnp1=np.dot(np.linalg.inv((M-B)),((np.dot(K,xn*tau*tau)+np.dot(M,(np.dot(2,xn)-xnm1))-np.dot(B,(xnm1)))))
    xnp1=np.dot(ch1,ch21+ch22+ch23)
    X.append(0.5*xnp1)#*0.5
    xnm1=xn
    xn=xnp1

##for j in X:
##    for g in j:
##        g=g+5
        
X=np.array(X)
X=X.transpose()
def y3(t,i):
    return X[t][i]





#pp.figure()
res=[]
k=0
for i in CTD:
    k+=1
    ins=[]
    res.append(ins)
    for j in t:
        wrt=y(j,i)
        res[k-1].append(wrt)#res[&] - one coordinate evolution through all time
k=0
#every i is conformed its own k and its own row in res
cmap=pp.get_cmap('jet_r')
patches=[]
for i in CTD:
    k+=1
    
    col=cmap(float(k)/(2*n))
    lin=pp.plot(t, res[k-1],color=col)
    #pp.scatter(t, res[k-1], color=col,marker='k')
    if(i>=n):
        patches.append(mps.Patch(color=col,label="v"+str(i-n)))
    else:
        patches.append(mps.Patch(color=col,label="x"+str(i)))
#leg1=pp.legend(handles=patches)


##y_arr = [y(i, 2*n-1) for i in t]
##y_arr2=[y(i, 2*n-2) for i in t]
##lin1,lin2=pp.plot(t, y_arr,'(0,123,2)--',t,y_arr2, 'r--')
##pp.legend((lin1,lin2),(u'x1 ',u'x2 '),loc='best')
pp.grid()
#for _ in range(0,tmax):
#    t=t+1
#    pp.plot(t,y(t,n+n-1),'bo')
pp.ylim([-10,10])
pp.xlim([0,100])


pp.draw()

#pp.figure()

#pp.plot(res[0],res[n])
# DAT WAS PHASE PLANE DAT WAS PHASE PLANE DAT WAS PHASE PLANE DAT WAS PHASE PLANE
######k=0
######patches=[]
######while k<n:
######    k+=1
######    col=cmap(float(k)/(2*n))
######    lin=pp.plot(res[k-1],res[k-1+n] ,color=col)
######    #pp.scatter(t, res[k-1], color=col,marker='k')
######    patches.append(mps.Patch(color=col,label="No "+str(k)))
######pp.legend(handles=patches)
#######pp.legend()
######pp.grid()
######pp.show()
######print "done"
t2=np.array(t2)


#cmap=pp.get_cmap('jet_r')



##patches=[]
for i in CTD:
    
    if(i>=n):
        continue
        
    col=cmap(float(i)/(2*n))
    lin=pp.plot(t2, X[i],color="b") #x[n-1] - koord over time
    f = interp1d(t2, X[i], kind="cubic")
    space = np.linspace(0, int(tmax-1), 200)
    pp.plot(space, [f(i) for i in space],color="r")
    #pp.scatter(t, res[k-1], color=col,marker='k')
    patches.append(mps.Patch(color=col,label="x"+str(i)))
#leg1=pp.legend(handles=patches)


pp.show()    
