def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

from vector_operations import Vector, dot;

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """Returns 1 if the perceptron fires, else returns 0"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)
    
and_weights = [2., 2.]
and_bias = -3.

print(f"perceptron_output(and_weights, and_bias, [1, 1]) = {perceptron_output(and_weights, and_bias, [1, 1])}")
print(f"perceptron_output(and_weights, and_bias, [0, 1]) = {perceptron_output(and_weights, and_bias, [0, 1])}")
print(f"perceptron_output(and_weights, and_bias, [1, 0]) = {perceptron_output(and_weights, and_bias, [1, 0])}")
print(f"perceptron_output(and_weights, and_bias, [0, 0]) = {perceptron_output(and_weights, and_bias, [0, 0])}")


or_weights = [2., 2.]
or_bias = -1.

print(f"perceptron_output(or_weights, or_bias, [1, 1]) = {perceptron_output(or_weights, or_bias, [1, 1])}")
print(f"perceptron_output(or_weights, or_bias, [0, 1]) = {perceptron_output(or_weights, or_bias, [0, 1])}")
print(f"perceptron_output(or_weights, or_bias, [1, 0]) = {perceptron_output(or_weights, or_bias, [1, 0])}")
print(f"perceptron_output(or_weights, or_bias, [0, 0]) = {perceptron_output(or_weights, or_bias, [0, 0])}")

not_weights = [-2.]
not_bias = 1.
print(f"perceptron_output(not_weights, not_bias, [1]) = {perceptron_output(not_weights, not_bias, [1.])}")
print(f"perceptron_output(not_weights, not_bias, [0]) = {perceptron_output(not_weights, not_bias, [0.])}")
