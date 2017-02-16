import numpy as np
from scipy import linalg as la
n=0
print "n"
n=int(input())
def  ND(n):#schitat
    m=[]
    i=0.0
    for _ in range(0,n):
        i=(input())
        m.append(i)
    return m

print "Wanna k"
k=ND(n)
K=np.diag(k)
k.pop(0)
k.append(0)
K=K+np.diag(k)
k.pop()
K=K-np.diag(k,1)
K=K-np.diag(k,-1)

T,Z=la.schur(K)
#T-up triangle Z-unitarn Z-1=Z*(ermSoprazh=trans&soprazh elem)
ev,evecr=la.eig(K)
#The normalized right
#eigenvector corresponding to the eigenvalue w[i]
#is the column vr[:,i]. Only returned if right=True.

f = open("matrixT.txt","wb")

for i in T:
    for j in i:
        f.write(str(j)+' ')
    f.write('\r\n')
f.close()
f = open("matrixZ.txt","wb")
for i in Z:
    for j in i:
        f.write(str(j)+' ')
    f.write('\r\n')
f.close()
