import numpy as np
import time
import librosa

def J(D,lmda):
    return max(np.linalg.norm(D, ord=2), np.linalg.norm(D.flatten(), ord=np.inf)/lmda)


def Se(X,e):
    E = np.ones((X.shape))*e
    C1 = (X > E) 
    C2 = C1*X
    C1 = C1*e
    C1 = C2 - C1

    C3 = (X < -1*E)
    C4 = C3*X
    C3 = C3*e
    C3 = C3 + C4

    return C1 + C3

def rpca(D,k):
    m,n = D.shape
    lmda = k/np.sqrt(max(m,n))
    M,P = librosa.magphase(D)
    #P = np.angle(D)
    #M = np.abs(D)
    Y = M/J(M,lmda)
    E = np.zeros((m,n))
    E_prev = E
    mu = 1.25/np.linalg.norm(M,2)
    cond1 = 1
    cond2 = 1
    i = 0
    while(not((cond1 < 1e-7) and (cond2 < 1e-5))):
        start = time.time()
        U,S,V = np.linalg.svd(M - E + (Y/mu))
        end = time.time()
        S = Se(S,1/mu)
        #print (end - start)
        if(m > n):
            S = np.concatenate((np.diagflat(S), np.zeros((m-n,n))), axis = 0)
        elif(m < n):
            S = np.concatenate((np.diagflat(S), np.zeros((m,n-m))), axis = 1)
        else:
            S = np.diagflat(S)
        A = U.dot(S).dot(V)
        E_prev = E
        E = Se(M - A + (Y/mu),(lmda/mu))
        Y = Y + mu*(M - A - E)
        cond1 = np.linalg.norm(M - A - E, "fro")/np.linalg.norm(M, "fro")
        cond2 = mu*(np.linalg.norm(E - E_prev, "fro")/np.linalg.norm(M,"fro"))
        if(cond2 < 1e-5):
            mu *= 1.6
            #print ("yo")
        #mu *= 1.6
        #print (mu)
        if(mu == np.inf):
            break
        i += 1
        #if(i%100 == 0):
        #    print (cond1, cond2) 
    return A*P, E*P
    #return A*np.exp(P*1j),E*np.exp(P*1j)

def tf_mask(M,L,S,gain):
    Mb = (np.abs(S) > gain*np.abs(L))
    Xs = Mb*M
    Xm = (1-Mb)*M
    return Xs,Xm

def soft_mask(M,L,S,gain):
    W = np.minimum(np.abs(M),np.abs(S))
    Ms = 1/(1 + np.exp(-15*(W/np.abs(M+1e-20) - np.sqrt(gain**2/(1+gain**2)))))
    Xs = Ms*M
    Xm = (1-Ms)*M
    return Xs,Xm

def speech(Y,k,gain):
    L,S = rpca(Y,k)
    #print(np.sum(Y-L-S))
    #Xs,Xm = tf_mask(Y,L,S,gain)
    Xs,Xm = soft_mask(Y,L,S,gain)
    return Xs, Xm
