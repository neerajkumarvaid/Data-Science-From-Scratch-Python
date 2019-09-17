from vector_operations import dot, Vector;

def predict(x: Vector, beta: Vector) -> float:
    """Assumes that the first element of x is 1"""
    return dot(x, beta)

predict([1,49,4,0],[3,5,2,4])

