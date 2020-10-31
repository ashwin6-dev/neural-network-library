import numpy as np

class Tensor(object):
    """Tensor class"""

    def __init__(self, value):
        self.value = np.array(value)

        if self.value.ndim < 2:
            while self.value.ndim < 2:
                self.value = np.expand_dims(self.value, axis=0)

        self.parents = []
        self.backward = None
        self.visited_parents = set()
        self.path_cache = {}
        self.ones = np.ones(self.value.shape)

    def has_path(self, tensor):
            if self == tensor:
                return True

            queue = self.parents.copy()

            while queue != []:
                current_tensor = queue[0]
                queue = queue[1:]

                if current_tensor == tensor:
                    self.path_cache[tensor] = True

                    return True

                queue = queue + current_tensor.parents


            return False


    def gradient(self, tensor, ignore_visited=True):
        current_tensor = self
        grad = Tensor(self.ones)

        while current_tensor != tensor:
            
            if current_tensor.parents[0].has_path(tensor) and ((current_tensor.parents[0] not in current_tensor.visited_parents and ignore_visited == False) or ignore_visited):
                grad = current_tensor.backward(current_tensor.parents[0], current_tensor.parents, grad)
                if ignore_visited:
                    current_tensor.visited_parents = set()

                current_tensor.visited_parents.add(current_tensor.parents[0])
                current_tensor = current_tensor.parents[0]
                
            elif current_tensor.parents[1].has_path(tensor) and ((current_tensor.parents[1] not in current_tensor.visited_parents and ignore_visited == False) or ignore_visited):
                grad = current_tensor.backward(current_tensor.parents[1], current_tensor.parents, grad)
                if ignore_visited:
                    current_tensor.visited_parents = set()

                current_tensor.visited_parents.add(current_tensor.parents[1])
                current_tensor = current_tensor.parents[1]
            else:
                return Tensor([[0]])

            

        grad = add(grad, self.gradient(tensor, ignore_visited=False))

        for i,d in enumerate(tensor.value.shape):
            if d == 1:
                grad.value = np.sum(grad.value, axis=i, keepdims=True)

        return grad

    def __str__(self):
        return f"Tensor(value={self.value})"

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)
   
    def __matmul__(self, other):
        return matmul(self, other)

    def __pow__(self, other):
        return pow(self, other)

    @staticmethod
    def randn(*args):
        value = np.random.randn(*args)
        return Tensor(value)

    def dispose(self):
        del self


def add(t1, t2):
    if type(t1) != Tensor:
        t1 = Tensor(t1)
    if type(t2) != Tensor:
        t2 = Tensor(t2)

    t = Tensor(t1.value + t2.value)
    t.parents = [t1, t2]

    def add_backward(v, parents, grad):
       local_grad = 0
       local_grad = int(v == parents[0]) + int(v == parents[1])

       grad.value = grad.value * local_grad

       return grad

    t.backward = add_backward
    return t

def sub(t1, t2):
    if type(t1) != Tensor:
        t1 = Tensor(t1)
    if type(t2) != Tensor:
        t2 = Tensor(t2)

    t = Tensor(t1.value - t2.value)
    t.parents = [t1, t2]

    def sub_backward(v, parents, grad):
       local_grad = 0
       local_grad = int(v == parents[0]) + int(v == parents[1]) * -1

       grad.value = grad.value * local_grad

       return grad

    t.backward = sub_backward
    return t

def mul(t1, t2):
    if type(t1) != Tensor:
        t1 = Tensor(t1)
    if type(t2) != Tensor:
        t2 = Tensor(t2)

    t = Tensor(t1.value * t2.value)
    t.parents = [t1, t2]

    def mul_backward(v, parents, grad):
       local_grad = 0
       if v == parents[0]:     
           local_grad += parents[1].value
       if v == parents[1]:
           local_grad += parents[0].value

       grad.value = grad.value * local_grad

       return grad

    t.backward = mul_backward
    return t

def div(t1, t2):
    if type(t1) != Tensor:
        t1 = Tensor(t1)
    if type(t2) != Tensor:
        t2 = Tensor(t2)

    t = Tensor(t1.value + t2.value)
    t.parents = [t1, t2]

    def div_backward(v, parents, grad):
       local_grad = 0
       if v == parents[0]:
           local_grad += 1 / parents[1].value
       if v == parents[1]:
           local_grad += -(parents[0] / (parents[1] ** 2))

       grad.value = grad.value * local_grad

       return grad

    t.backward = div_backward
    return t

def matmul(t1, t2):
    t = Tensor(t1.value.dot(t2.value))
    t.parents = [t1, t2]

    def matmul_backward(v, parents, grad):
       local_grad = 0
       if v == parents[0]:
           local_grad = np.dot(grad.value, parents[1].value.T)
       elif v == parents[1]:
           local_grad = np.dot(parents[0].value.T, grad.value)

       grad.value = local_grad

       return grad

    t.backward = matmul_backward
    return t

def pow(t1, t2):
    t = Tensor(t1.value ** t2.value)
    t.parents = [t1, t2]

    def pow_backward(v, parents, grad):
        local_grad = 0
        if v == parents[0]:
            local_grad = t2.value * (t1.value ** (np.subtract(t2.value, 1))) 

        grad.value = local_grad

        return grad
    
    t.backward = pow_backward
    return t
