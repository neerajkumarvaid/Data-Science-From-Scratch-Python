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

import math

def sigmoid(t: float) -> float:
    return 1.0 / (1 + math.exp(-t))


t = [i for i in range(-10,11,1)]
sigmoid_t = [sigmoid(x) for x in t]
step_t = [step_function(x) for x in t]

import matplotlib.pyplot as plt
plt.plot(t, sigmoid_t, label = 'sigmoid')
plt.plot(t, step_t, 'r--', label = 'step function')
plt.legend()
plt.show()

def neuron_output(weights: Vector, inputs: Vector) -> float:
    """Weights include a bias terms, input includes a 1."""
    return sigmoid(dot(weights, inputs))

from typing import List

def feed_forward(neural_network: List[List[Vector]],
                input_vector: Vector) -> List[Vector]:
    
    """Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one)."""
    
    outputs: List[Vector] = []
        
    for layer in neural_network:
        input_with_bias = input_vector + [1.0] # Adds a constant for bias
        output = [neuron_output(input_with_bias, neuron)
                 for neuron in layer]
        outputs.append(output)
        
        # Then the input to the next layer is the output of this layer
        input_vector = output
        
    return outputs
