import framework.tensor as tensor

"""
Creating tensors and performing mathematical operations with them
"""

a = tensor.Tensor(5)
b = tensor.Tensor(10)

c = a * b

d = c ** tensor.Tensor(3)

print (d, d.gradient(a)) # d.gradient(a) : Find the derivative of "d" with respect to "a"


m1 = tensor.Tensor([[2, 2, 2], [3, 3, 3]])
m2 = tensor.Tensor([[2, 2],[2, 2], [2, 2]])

result = m1 @ m2 # Matrix multiply m1 and m2

print (result, result.gradient(m2)) # result.gradient(m2) : Find the derivative of "result" with respect to "m2"

"""
### Output ###

Tensor(value=[[125000]]) Tensor(value=[[75000]])
Tensor(value=[[12 12]
 [18 18]]) Tensor(value=[[5. 5.]
 [5. 5.]
 [5. 5.]])

"""
