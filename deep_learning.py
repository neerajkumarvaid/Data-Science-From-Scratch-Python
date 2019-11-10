
Tensor = list

from typing import List

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

print(f"shape([1,2,3]) = {shape([1,2,3])}")
print(f"shape([[1,2],[3,4],[5,6]]) = {shape([[1,2],[3,4],[5,6]])}")
print(f"shape([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]],[[15,16,17],[18,19,20]]]) = {shape([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]],[[15,16,17],[18,19,20]]])}")

def is_1d(tensor: Tensor) -> bool:
    """If tensor[0] is a list, it's a higher-order tensor.
    Otherwise, tensor is 1-dimensional (that is, a vector)."""
    return not isinstance(tensor[0], list)

print(f"is_1d([1,2,3]) = {is_1d([1,2,3])}")
print(f"is_1d([[1,2],[3,4]]) = {is_1d([[1,2],[3,4]])}")

def tensor_sum(tensor: Tensor) -> float:
    """Sums up all the values in a tensor"""
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)

from typing import Callable
def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Applies f element-wise"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]
    
print(f"tensor_apply(lambda x: x + 1, [1,2,3]) = {tensor_apply(lambda x: x + 1, [1,2,3])}")
print(f"tensor_apply(lambda x: 2*x, [[1,2],[3,4]]) = {tensor_apply(lambda x: 2*x, [[1,2],[3,4]])}")


def zero_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

print(f"zero_like([1,2,3]) = {zero_like([1,2,3])}")
print(f"zero_like([[1,2],[3,4]]) = {zero_like([[1,2],[3,4]])}")


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Applies f to corresponding elements of t1 and t2"""
    if is_1d(t1):
        return [f(x,y) for x,y in zip(t1,t2)]
    else:
        return [tensor_combine(f, tensor_i, tensor_j)
               for tensor_i, tensor_j in zip(t1, t2)]
    
import operator

print("tensor_combine(operator.add, [1,2,3], [4,5,6])" +  
      f" = {tensor_combine(operator.add, [1,2,3], [4,5,6])}")

from typing import Iterable, Tuple

class Layer:
    """
    Our neural networks will be composed of Layers, each of which
    knows how to do some computation on its inputs in the "forward"
    direction and propagate gradients in the "backward" direction.
    """
    def forward(self, input):
        """
        Note the lack of types. We're not going to be prescriptive
        about what kinds of inputs layers can take and what kinds
        of outputs they can return.
        """
        raise NotImplementedError

    def backward(self, gradient):
        """
        Similarly, we're not going to be prescriptive about what the
        gradient looks like. It's up to you the user to make sure
        that you're doing things sensibly.
        """
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """
        Returns the parameters of this layer. The default implementation
        returns nothing, so that if you have a layer with no parameters
        you don't have to implement this.
        """
        return ()

    def grads(self) -> Iterable[Tensor]:
        """
        Returns the gradients, in the same order as params()
        """
        return ()
