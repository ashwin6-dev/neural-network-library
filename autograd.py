import framework.tensor as tensor

a = tensor.Tensor(5)
b = tensor.Tensor(10)

c = a * b

d = c ** tensor.Tensor(3)

print (d, d.gradient(a))


m1 = tensor.Tensor([[2, 2, 2], [3, 3, 3]])
m2 = tensor.Tensor([[2, 2],[2, 2], [2, 2]])

result = m1 @ m2

print (result, result.gradient(m2))