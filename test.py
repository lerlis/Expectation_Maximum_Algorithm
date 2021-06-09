import numpy as np

one = np.array([5, 5, 5])
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.hstack((a, b))
d = np.vstack((a, b))
e = 100
f = np.dot(a, b)
g = np.dot(one, a)
h = a * a
print(c)
print(d)
print(d[0], '\n', d[1])
print(f)
print(g)
print(h)