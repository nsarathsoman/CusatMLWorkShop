import numpy as np
a = np.array([1,2,3])
print a
print a.shape

a = np.array([[1,2],[3,4]])
print a

b = a.reshape(3,2)
print b

np.row_stack((a, [3,4,3,2])
np.row_stack((a, [3,4,3]))

np.zeros((5,2))
np.ones((5,2))

print a[1,1]

print a[:,2]

print a[-1]

print a[-1, -1]

x = np.array([[1,2], [3,4]])
y = np.array([[1,2,3],[3,4,5]])

np.dot(x, y)

y.transpose()

np.linalg.inv(y)

np.linalg.inv(np.dot(x, x.transpose()))

np.linalg.pinv(y)