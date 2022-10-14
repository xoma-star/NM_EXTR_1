import numpy as np
from sympy import N

np.random.seed(1)


a = -10  # min элемент в матрице
b = 10  # max элемент в матрице
n = 3  # размер матрицы
m = n + 2
A = a + (b - a) * np.random.rand(n, n)
f = a + (b - a) * np.random.rand(n, 1)
u = np.dot(A, f)
# print(A)
A = np.append(A, [A[0] + A[n - 1]], axis=0)
A = np.append(A, [A[1] + A[n - 2]], axis=0)
f = np.append(f, [f[0] + f[n - 1]], axis=0)
f = np.append(f, [f[1] + f[n - 2]], axis=0)
# print(A)
# print(f)
# print(u)
e_A = 0.001
rand_val = -1 + 2 * np.random.rand()  # [-1, 1)
A_ = A * (1 + rand_val * e_A)
e_h = 0.001
f_ = f * (1 + rand_val * e_h)
h = np.linalg.norm(A - A_, ord='fro')
sigma = np.linalg.norm(f - f_, ord=2)
psi = h * np.linalg.norm(u, ord=2) + sigma
print(h, sigma, psi)


def Newton(F, dF, x_prev, delta):
    err = 1
    x = np.zeros_like(x_prev)
    iters = 0
    while err > delta:
        iters += 1
        delta_x_prev = np.linalg.solve(dF(x_prev), -F(x_prev)).reshape(n, )
        x = x_prev + delta_x_prev
        print(dF(x_prev))
        print(-F(x_prev))
        print(delta_x_prev)
        err = np.linalg.norm(x - x_prev, ord=1)
        print(err)
        x_prev = x
    return x


def kron(q, i):
    if q == i:
        return 1
    else:
        return 0


def diff1(m, n, k, h, a, v, f):
    buff = np.zeros(5, float)
    # 1 3
    # 2 4
    for i in range(m):
        buff[0] = 0
        for j in range(n):
            buff[0] += a[i][j]*v[j]-f[i]
        buff[1] += a[i][k] * buff[0]
        buff[2] += np.power(buff[0], 2)

    buff[2] = np.sqrt(buff[2])
    buff[3] = h*v[k]
    for j in range(n):
        buff[4] += np.power(v[j], 2)
    buff[4] = np.sqrt(buff[4])

    return buff[1]/buff[2] + buff[3]/buff[4]


def diff2(m, n, k, q, h, a, v, f):
    buff = np.zeros(8, float)
    #
    #(0 * 2 - (3*4)/2 ) * "1"/5 + "h"/6 * ( "kron(k,q)" * 7 - "u[q]*u[k]"/7 )
    #

    for i in range(m):
        buff[0] += a[i][k]*a[i][q]

    for i in range(m):
        buff[1] = 0
        for j in range(n):
            buff[1] += a[i][j]*v[j]-f[i]
        buff[2] += np.power(buff[1], 2)
        buff[3] += a[i][q] * buff[1]
        buff[4] += a[i][k] * buff[1]

    buff[5] = buff[2]
    buff[2] = np.sqrt(buff[2])

    for j in range(n):
        buff[6] += np.power(v[j], 2)

    buff[7] = np.sqrt(buff[6])

    return (buff[0] * buff[2] - (buff[3]*buff[4])/buff[2]) / buff[5] + (h/buff[6]) * (kron(k, q)*buff[7] - v[q]*v[k]/buff[7])


def F(x):
    return np.array([[diff1(m, n, d, h, A_, x, f_)] for d in range(n)])


def dF(x):
    buff = []
    for i in range(n):
        buff2 = []
        for j in range(n):
            buff2.append([diff2(m, n, i, j, h, A_, x, f_)])
        buff.append(buff2)
    return np.squeeze(np.array(buff))


'''def F(x):
    return np.array([[-(x[0] + 3 * math.log(x[0], 10) - x[1])],
                     [-(2 * x[0] * x[0] - x[0] * x[1] - 5 * x[0] + 1)]])


def dF(x):
    return np.array([[1 + 3 * 0.43429 / x[0], -2 * x[1]],
                     [4 * x[0] - x[1] - 5, -x[1]]])'''


x_0 = np.squeeze(u)
answer = Newton(F, dF, x_0, 1e-2)
print(answer)
