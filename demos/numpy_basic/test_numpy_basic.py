from numpy_basic import foo
import numpy as np

x = np.zeros((5, 7))
print "Before:"
print x

foo(x)

print "After:"
print x

