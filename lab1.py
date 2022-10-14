from re import U
import numpy as np

n, m = 3, 3
eps_u = 0.001
eps_A = 0.01

np.random.seed(1000)
det_A = 0
while det_A == 0:
    A = np.random.uniform(0, 1, (n,m)) * 1
    det_A = np.linalg.det(A)

np.random.seed(1000)
u = np.random.uniform(0, 1, (n,1)) 

f = A.dot(u)

A = np.vstack([A, A[0] + A[n-1]])
A = np.vstack([A, A[1] + A[n-2]])

f = np.append(f, f[0] + f[n-1])
f = np.append(f, f[1] + f[n-2])

np.random.seed(1000)
A_err = A * (1 + np.random.uniform(-1, 1, (n + 2, n)) * eps_A)
f_err = f * (1 + np.random.uniform(-1, 1, (n + 2, 1)) * eps_u)

h = np.linalg.norm(A - A_err)
sigma = np.linalg.norm(f - f_err)
psi = h * np.linalg.norm(u) + sigma
mu = np.linalg.norm(A_err.dot(u) - f_err)

u_pribl = np.zeros(n,float)


def Newton(m, n, k, h, A, u, f):

    u_pribl[0] = u[0]
    for i in range(1000):
        for k in range(0, n):
            if i != 0:
                dd = diff1(m, n, k, h, A, u, f) / diff2(u, h, A, f, m, n, k)

                u_pribl[k] = u_pribl[k-1] - dd
                #print(dd, u_pribl[k])
    print(u_pribl)
    print(u)
    

def diff1(m, n, k, h, a, u, f):
    p = np.zeros(2, float)
    p[0] = h * u[k] / sum(u**2)**0.5

    p[1] = (sum([a[i][k] * (a[i][j] * u[j] - f[i]) for j in range(n) for i in range(n)]) / 
        sum([(a[i][j] * u[j] - f[i])**2 for j in range(n) for i in range(n)])**0.5)

    return p[0] + p[1]

def diff2(u, h, a, f, g, n, k):
    p = np.zeros(4, float)

    p[0] = h / sum(u * u)**0.5

    p[1] = h * u[k]**2 / sum(u * u)**1.5

    p[2] = (sum([a[i][k]**2 for i in range(n)]) / 
        (sum([(a[i][j] * u[j] - f[i])**2 for j in range(n) for i in range(n)])**0.5))

    p[3] = ((sum([a[i][k] * (a[i][j] * u[j] - f[i]) for j in range(n) for i in range(n)]))**2 / 
        (2. * sum([(a[i][j] * u[j] - f[i])**2 for j in range(n) for i in range(n)])**1.5))
    

    return p[0] - p[1] + p[2] - p[3]


Newton(m, n, n, h, A, u, f)
print(A)