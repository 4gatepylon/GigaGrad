from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, List, Tuple, Optional, Callable
from uuid import uuid4
from random import random

# NOTE we only support matrices for now, only linked-list computational graphs
class Matrix:
    """Simple matrix operations library that enables you to do the following operations:
        - Addition
        - Pointwise Multiplication
        - Transpose
        - Matrix Multiplication

    It provides
        - Simple checking via assertions (for sizes to match)
        - Broadcasting
        - A simple interface

    It does not provide tensor or higher dimensional operations.

    # NOTE that this library is meant, purely, to enable us to calculate values.
    """
    ZERO_INITIALIZER = lambda: 0.0
    RANDOM_INITIALIZER = lambda: random()
    def __init__(self, height: int = None, width: int = None, values: Optional[list[list[float]]] = None, initializer: Optional[Callable[[], float]] = ZERO_INITIALIZER) -> None:
        assert (height is not None and width is not None) or (
            values is not None
        )
        assert values is None or (
            type(values) == list and len(values) >= 1 and type(values[0]) == list and len(values[0]) >= 1 and type(values[0][0]) in {float, int}
        )
        assert values is None or (
            all([len(values[i]) == len(values[0]) for i in range(len(values))])
        )
        
        self.height = height if height is not None else len(values)
        self.width = width if width is not None else len(values[0])
        assert self.height is not None and self.width is not None

        initializer = initializer if initializer else Matrix.ZERO_INITIALIZER
        self.values = values if values else [[initializer() for _ in range(width)] for _ in range(height)]
        
        assert self.values is not None
        assert len(self.values) == self.height
        assert all([len(self.values[i]) == self.width for i in range(self.height)])
    
    def __eq__(self, __value: object) -> bool:
        return type(__value) == Matrix and (
            self.height == __value.height and self.width == __value.width) and (
            all([
                self.values[i][j] == __value.values[i][j] for i in range(self.height) for j in range(self.width)])
            )
    
    def lmatmul(self, other: Matrix) -> Matrix:
        # Our matrix is left
        return Matrix.matmul(self, other)
    def rmatmul(self, other: Matrix) -> Matrix:
        # Our matrix is right
        return Matrix.matmul(other, self)
    def plus(self, other: Matrix) -> Matrix:
        # Order does not matter
        return Matrix.add(self, other)
    def times(self, other: float) -> Matrix:
        # Use broadcasting
        other_matrix = Matrix(1, 1, values=[[other]])
        return Matrix.pmult(self, other_matrix)
    def neg(self) -> Matrix:
        return self.times(-1.0)
    def minus(self, other: Matrix) -> Matrix:
        return Matrix.add(self, other.neg())
    def inverted(self) -> Matrix:
        return Matrix(self.height, self.width, values=[[1.0 / self.values[i][j] for j in range(self.width)] for i in range(self.height)])
    def divided_by(self, other: Matrix) -> Matrix:
        return Matrix.pmult(self, other.inverted())
    def transpose(self) -> Matrix:
        # Swap the i and j indices
        return Matrix(
            self.width,
            self.height, 
            [[self.values[j][i] for j in range(self.height)] for i in range(self.width)]
        )
    def mask(self, include: Callable[[float, int, int], bool], default_value: float = 0.0) -> Matrix:
        return Matrix(
            self.height,
            self.width,
            [[self.values[i][j] if include(self.values[i][j], i, j) else default_value for j in range(self.width)] for i in range(self.height)]
        )
    
    @staticmethod
    def matmul(left: Matrix, right: Matrix) -> Matrix:
        assert left.width == right.height
        # Multiply two matrices using meaningful multiplication

        dot_dim = left.width
        out = [[0.0 for _ in range(right.width)] for _ in range(left.height)]
        for orow in range(left.height):
            for ocol in range(right.width):
                for dentry in range(dot_dim):
                    left_row = lambda j: left.values[orow][j]
                    right_col = lambda i: right.values[i][ocol]
                    out[orow][ocol] += left_row(dentry) * right_col(dentry)
        return Matrix(left.height, right.width, values=out)
    
    @staticmethod
    def add(left: Matrix, right: Matrix) -> Matrix:
        return Matrix.preduce(left, right, lambda a, b: a + b)
    @staticmethod
    def pmult(left: Matrix, right: Matrix) -> Matrix:
        return Matrix.preduce(left, right, lambda a, b: a * b)

    @staticmethod
    def preduce(left: Matrix, right: Matrix, pop: Callable[[float, float], float], allow_broadcast: bool = True) -> Matrix:
        # Pointwise reducer for two matrices (i.e. pointwise multiply or add or average or...)
        # Allows broadcasting

        allowed_rshapes = [] # Declare so non-local to for scope (enables assertion later)
        for _ in range(2):
            lh, lw = left.height, left.width
            lshape = (lh, lw)
            rshape = (right.height, right.width)

            # Right must match left, right will be reduced onto left (left is template shape)
            allowed_rshapes = [lshape]
            if allow_broadcast:
                allowed_rshapes.append((1, lw))
                allowed_rshapes.append((lh, 1))
                allowed_rshapes.append((1, 1))
            if rshape not in allowed_rshapes:
                left, right = right, left
            else:
                break
        assert rshape in allowed_rshapes, f'{rshape} not in {allowed_rshapes}'

        # Left is template
        height = left.height
        width = left.width
        out = [[0.0 for _ in range(width)] for _ in range(height)]

        def get_right_value(i: int, j: int) -> float:
            if rshape == (1, 1):
                return right.values[0][0]
            elif rshape == (1, width):
                return right.values[0][j]
            elif rshape == (height, 1):
                return right.values[i][0]
            elif rshape == (height, width):
                return right.values[i][j]
            else:
                raise ValueError(f'Invalid rshape {rshape} when multiplying, are you sure you checked?')

        for i in range(height):
            for j in range(width):
                out[i][j] = pop(left.values[i][j], get_right_value(i, j))
        return Matrix(height, width, values=out)

    @staticmethod
    def almost_equal(a: Matrix, b: Matrix, rtol: float = 1e-5) -> bool:
        # Return whether the absolute difference is smaller than this for all pointwise entries
        assert a.height == b.height and a.width == b.width
        for i in range(a.height):
            for j in range(a.width):
                x = a.values[i][j]
                y = b.values[i][j]
                if abs(x - y) > rtol:
                    return False
        return True

class GradientLib:
    """Gradient lib is just a place to house/namespace pure functions that calculate gradient w.r.t.
    the output of a function for a set of incoming parameters matrices. The gradient outputs are always
    matrices that have the exact same shape and same order as the incoming parameters.

    - The inputs to every function are the inputs to the computation that was done on the CG.
    - The outputs of every function are the derivatives of those inputs' corr. variables w.r.t. the loss.
    """
    def MSE(left: Matrix, right: Matrix) -> tuple[Matrix, Matrix]:
        ldiff = left.minus(right).times(2.0)
        rdiff = right.minus(left).times(2.0)
        return ldiff, rdiff
    def KL(left: Matrix, right: Matrix) -> tuple[Matrix, Matrix]:
        # KL = sum(left * log(left / right))
        pass
    def Matmult(left: Matrix, right: Matrix) -> tuple[Matrix, Matrix]:
        pass
    def MatAdd(left: Matrix, right: Matrix) -> tuple[Matrix, Matrix]:
        pass
    def MatPmult(left: Matrix, right: Matrix) -> tuple[Matrix, Matrix]:
        pass
    def ReLU(m: Matrix) -> tuple[Matrix, Matrix]:
        return m.mask(lambda v, _, __: v >= 0.0, default_value=0.0)

class CGNodeState(Enum):
    """A CGNodeState stores the state of a node during its computational graph gradient descent
    backprop. (etc). NOTE: CG means Computational Graph, generally.
    - Forward flowing is meant to coordinate propagation of output values.
    - Backward flowing is meant to coordinate propagation of derivatives w.r.t. the loss.

    Every node will
    1. Start out empty, nothing is happening
    2. Enter a state in which its predecessors are flowing forward (i.e. at least one predecessor has reached
        the FORWARD_FLOWED state but not all).
    3. Only once ALL predecessors have reached the FORWARD_FLOWED state will the node itself reach the
        FORWARD_FLOWED state. At this point all the outputs of its predecessors are known and ready to be used.
        This means that in this state it can calculate the output value of this node.
    4. Once the node has calculated its output value, it will stay in FORWARD_FLOWED until at least one of
        its successors is BACKWARD_FLOWED. Then it will enter the SUCCESSORS_BACKWARD_FLOWING state until
        all of its successors are BACKWARD_FLOWED.
    5. Once all of its successors are BACKWARD_FLOWED, it will enter the BACKWARD_FLOWED state. At this point
        the gradient with respect to any parameters this node may use can be calculated. Moreover, the gradient
        w.r.t. the output of this node and each of the inputs to this node can be handed off to the predecessors.
    6. Once all of the predecessors are BACKWARD_FLOWED, the node will be available to enter the stepped state
        by calling a step function. The step function just updates the parameters of the node using the gradients.
    
    NOTE: all CGNodes are ONLY able to traverse state in the order
    EMPTY -> PREDECESSORS_FORWARD_FLOWING -> FORWARD_FLOWED -> SUCCESSORS_BACKWARD_FLOWING -> BACKWARD_FLOWED -> STEPPED ->
        back to EMPTY.
    """

    EMPTY = 0
    PREDECESSORS_FORWARD_FLOWING = 1
    FORWARD_FLOWED = 2
    SUCCESSORS_BACKWARD_FLOWING = 3
    BACKWARD_FLOWED = 4
    STEPPED = 5

# class GradientLib:
#     @staticmethod
#     def MSE():
#         pass

# class LossCG:
#     def __init__(self, l: List[Tuple[Function, Optional[Parameter]]]) -> None:
#         self.funcs = []
#         self.params = []
#         for func, param in l:
#             self.funcs.append(func)
#             if param is not None:
#                 self.params.append(param)
#                 self.funcs[-1].parameter = param
#         for i in range(len(self.funcs) -1):
#             self.funcs[i].next = self.funcs[i+1]
#             self.funcs[i+1].prev = self.funcs[i]
#     def forward(self, input: Matrix) -> Matrix:
#         return self.funcs[0].forward(input)
#     def backward(self, loss: Matrix) -> None:
#         self.funcs[-1].backward(loss)
#     def step(self) -> None:
#         for param in self.params:
#             param.step()
    
#     def clear(self) -> None:
#         for func in self.funcs:
#             func.state = CGNodeState.EMPTY
#             func.derivative_wrt_loss = None
#         for param in self.params:
#             param.state = CGNodeState.EMPTY
#             param.derivative_wrt_loss = None

# class CGNode(ABC):
#     def __init__(self) -> None:
#         self.state = CGNodeState.EMPTY
#         self.id = uuid4()
#         self.derivative_wrt_loss: Matrix = None
#         self.output_value: Callable[[], Matrix] = lambda: None

#     def uid(self) -> int:
#         return self.id
#     def backward(self, next_derivative: Any = None) -> Any:
#         raise NotImplementedError
#     def step(self) -> None:
#         raise NotImplementedError

# class Parameter(CGNode):
#     def __init__(self, function: Optional[Function] = None) -> None:
#         super().__init__()
#         # self.state, self.id, self.derivative_wrt_loss
#         # (already present)
#         self.lr = 0.01
#         self.function = function
    
#     def backward(self, derivative: Matrix = None) -> Any:
#         self.derivative_wrt_loss = derivative
#     def step(self) -> None:
#         output_value = self.output_value()

#         # They must be matries
#         assert type(output_value) == list and len(output_value) >= 1
#         assert all([type(output_value[i]) == list and len(output_value[i]) >= 1 for i in range(len(output_value))])

#         # The matrix shaps must match
#         assert len(self.derivative_wrt_loss) == len(output_value)
#         assert all(len(self.derivative_wrt_loss[i]) == len(output_value[i]) for i in range(len(self.derivative_wrt_loss)))
        
#         # One for one step
#         for i in range(len(self.derivative_wrt_loss)):
#             for j in range(len(self.derivative_wrt_loss[i])):
#                 output_value[i][j] -= self.lr * self.derivative_wrt_loss[i][j]

# class Function(CGNode):
#     def __init__(self, parameter: Optional[Parameter] = None, next: Optional[Function] = None, prev: Optional[Function] = None) -> None:
#         super().__init__()
#         # self.state, self.id, self.derivative_wrt_loss
#         # (already present)
#         self.parameter = parameter
#         self.next = next
#         self.prev = prev

#     # Push forward the computation
#     def _calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
#         raise NotImplementedError
#     def forward(self, input: Matrix) -> Matrix:
#         assert self.state == CGNodeState.EMPTY

#         outgoing = self._calc_forward(input, self.parameter.output_value() if self.parameter else None)
#         self.state = CGNodeState.FORWARD_FLOWED

#         if self.next is None:
#             return outgoing
#         return self.next.forward(outgoing)
    
#     # Calculate the derivative w.r.t. the loss for the parameter and for the input
#     def _calc_derivative_param(self, next_derivative: Matrix) -> Matrix:
#         raise NotImplementedError
#     def _calc_derivative_input(self, next_derivative: Matrix) -> Matrix:
#         raise NotImplementedError
#     def backward(self, next_derivative: Any = None) -> None:
#         assert self.state == CGNodeState.FORWARD_FLOWED
#         assert (next_derivative is None) == (self.next is None)

#         next_derivative = [[1.0]] if next_derivative is None else next_derivative
#         derivative_wrt_loss_input = self._calc_derivative_input(next_derivative)
#         derivative_wrt_loss_param = self._calc_derivative_param(next_derivative)
#         self.derivative_wrt_loss = derivative_wrt_loss_input

#         if self.parameter is not None:
#             # This just stores that derivative
#             self.parameter.backward(derivative_wrt_loss_param)
#         if self.prev is not None:
#             self.prev.backward(self.derivative_wrt_loss)

#     # NOTE: does not implement step
#     def step(self) -> None:
#         raise ValueError("You tried to call step on a function?!")

# class MatrixAffine(Parameter):
#     # NOTE that you must always multiple with input on the right so that
#     # we are interperting the columns as the vectors of the vector space
#     def __init__(self, height: int, width: int) -> None:
#         super().__init__()
#         self.matrix = [[random() for _ in range(width)] for _ in range(height)]
#         self.affine = [random() for _ in range(height)]
#         # List of rows, turn the affine into a row
#         self.output_value = lambda: self.matrix + [self.affine]
    
# class MatrixAffineMultiply(Function):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#     def _calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
#         # Must have a matrix and a vector (affine) parameter
#         assert param_value
#         assert len(param_value) >= 2

#         matrix = param_value[:-1]
#         affine_row = param_value[-1:]
#         assert len(affine_row) == 1
        
#         # Turn it into an affine column
#         affine = [[affine_row[0][i]] for i in range(len(affine_row[0]))]

#         # Adding enables broadcasting
#         mmed =  Matrix.matmul(matrix, input)
#         aed = Matrix.add(affine, mmed)
#         out = aed

#         return out


def TEST_MAT_LIB() -> None:
    # Check that you can make matrices with width and height
    Matrix(height=2, width=2)

    # Check that you can make matrices with values
    mat1 = Matrix(values=[[1, 2, 3], [4, 5, 6]])
    mat2 = Matrix(values=[[1, 0], [1, 0], [1, 0]])
    mat1a = Matrix(values=[[1, 1, 1], [1, 1, 1]])
    matb = Matrix(values=[[1, 1, 1]])

    # Check that matmul works
    assert Matrix.matmul(mat1, mat2).values == [[6, 0], [15, 0]]
    assert Matrix.matmul(mat2, mat1).values == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert Matrix.matmul(mat1, mat2) == mat1.lmatmul(mat2) and mat1.lmatmul(mat2) == mat2.rmatmul(mat1)
    assert Matrix.matmul(mat2, mat1) == mat2.lmatmul(mat1) and mat1.lmatmul(mat2) == mat2.rmatmul(mat1)
    # Check that transpose works
    assert matb.transpose().values == [[1], [1], [1]]
    # Check that addition works
    assert Matrix.add(mat1, mat1a).values == [[2, 3, 4], [5, 6, 7]]
    # Check that pointwise multiplication works
    assert Matrix.pmult(mat1, mat1a) == mat1
    # Check that broadcasting properly works
    assert Matrix.add(mat1, matb).values == [[2, 3, 4], [5, 6, 7]] and Matrix.add(matb, mat1) == Matrix.add(mat1, matb)
    assert Matrix.pmult(mat1, matb).values == [[1, 2, 3], [4, 5, 6]] and Matrix.pmult(matb, mat1) == Matrix.pmult(mat1, matb)
    assert Matrix.add(mat2, matb.transpose()).values == [[2, 1], [2, 1], [2, 1]] and Matrix.add(matb.transpose(), mat2) == Matrix.add(mat2, matb.transpose())
    assert Matrix.pmult(mat2, matb.transpose()).values == [[1, 0], [1, 0], [1, 0]] and Matrix.pmult(matb.transpose(), mat2) == Matrix.pmult(mat2, matb.transpose())
    
    # Check that other nifty operations work
    assert mat1a.minus(mat1).values == [[0, -1, -2], [-3, -4, -5]]
    assert mat1.times(2).values == [[2, 4, 6], [8, 10, 12]]
    assert mat1.divided_by(Matrix(values=[[0.2]])) == mat1.times(5)
    assert Matrix.almost_equal(Matrix.inverted(mat1), Matrix(values=[[1.0, 0.5, 1.0/3.0], [0.25, 0.2, 1.0/6.0]]))

def TEST_TRIVIAL_GRADIENT() -> None:
    pass

if __name__ == "__main__":
    """ Begin by testing our matrix code """
    TEST_MAT_LIB()
    
    """ Continue to test the propagation of our gradients on a simple example
        (the gradient value code and the loss graph traversal/topo-sort, etc... code is too
        simple to be really worth testing here)
    """

    # matmul_seq = [
    #     # Height, width for each
    #     # Cancelled: 1st reduce from 28x28 to 4x28
    #     # (MatrixAffineMultiply(), MatrixAffine(4, 28)),
    #     # 2nd reduce from 4x28 to 1x28
    #     (MatrixAffineMultiply(), MatrixAffine(1, 28)),
    # ]
    # # TODO not having transposes (or generally left vs. right mult.) is a real problem

    # matmul_graph = LossGraph(matmul_seq)
    # input = [[random() for _ in range(28)] for _ in range(28)]
    # print("---image---")
    # print(input)
    # print("-----------")

    # output = matmul_graph.forward(input)
    # print(output)