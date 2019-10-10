
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

