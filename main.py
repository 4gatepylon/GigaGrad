from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, List, Tuple, Optional, Callable
from uuid import uuid4
from random import random

# NOTE we only support matrices for now, only linked-list computational graphs
Matrix = List[List[float]]

class NodeState(Enum):
    EMPTY = 0
    FORWARD_FLOWED = 1
    BACKWARD_FLOWED = 2
    STEPPED = 3

class LossGraph:
    def __init__(self, l: List[Tuple[Function, Optional[Parameter]]]) -> None:
        self.funcs = []
        self.params = []
        for func, param in l:
            self.funcs.append(func)
            if param is not None:
                self.params.append(param)
                self.funcs[-1].parameter = param
        for i in range(len(self.funcs) -1):
            self.funcs[i].next = self.funcs[i+1]
            self.funcs[i+1].prev = self.funcs[i]
    def forward(self, input: Matrix) -> Matrix:
        return self.funcs[0].forward(input)
    def backward(self, loss: Matrix) -> None:
        self.funcs[-1].backward(loss)
    def step(self) -> None:
        for param in self.params:
            param.step()
    
    def clear(self) -> None:
        for func in self.funcs:
            func.state = NodeState.EMPTY
            func.derivative_wrt_loss = None
        for param in self.params:
            param.state = NodeState.EMPTY
            param.derivative_wrt_loss = None

class Node(ABC):
    def __init__(self) -> None:
        self.state = NodeState.EMPTY
        self.id = uuid4()
        self.derivative_wrt_loss: Matrix = None
        self.output_value: Callable[[], Matrix] = lambda: None

    def uid(self) -> int:
        return self.id
    def backward(self, next_derivative: Any = None) -> Any:
        raise NotImplementedError
    def step(self) -> None:
        raise NotImplementedError

class Parameter(Node):
    def __init__(self, function: Optional[Function] = None) -> None:
        super().__init__()
        # self.state, self.id, self.derivative_wrt_loss
        # (already present)
        self.lr = 0.01
        self.function = function
    
    def backward(self, derivative: Matrix = None) -> Any:
        self.derivative_wrt_loss = derivative
    def step(self) -> None:
        output_value = self.output_value()

        # They must be matries
        assert type(output_value) == list and len(output_value) >= 1
        assert all([type(output_value[i]) == list and len(output_value[i]) >= 1 for i in range(len(output_value))])

        # The matrix shaps must match
        assert len(self.derivative_wrt_loss) == len(output_value)
        assert all(len(self.derivative_wrt_loss[i]) == len(output_value[i]) for i in range(len(self.derivative_wrt_loss)))
        
        # One for one step
        for i in range(len(self.derivative_wrt_loss)):
            for j in range(len(self.derivative_wrt_loss[i])):
                output_value[i][j] -= self.lr * self.derivative_wrt_loss[i][j]

class Function(Node):
    def __init__(self, parameter: Optional[Parameter] = None, next: Optional[Function] = None, prev: Optional[Function] = None) -> None:
        super().__init__()
        # self.state, self.id, self.derivative_wrt_loss
        # (already present)
        self.parameter = parameter
        self.next = next
        self.prev = prev

    # Push forward the computation
    def _calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
        raise NotImplementedError
    def forward(self, input: Matrix) -> Matrix:
        assert self.state == NodeState.EMPTY

        outgoing = self._calc_forward(input, self.parameter.output_value() if self.parameter else None)
        self.state = NodeState.FORWARD_FLOWED

        if self.next is None:
            return outgoing
        return self.next.forward(outgoing)
    
    # Calculate the derivative w.r.t. the loss for the parameter and for the input
    def _calc_derivative_param(self, next_derivative: Matrix) -> Matrix:
        raise NotImplementedError
    def _calc_derivative_input(self, next_derivative: Matrix) -> Matrix:
        raise NotImplementedError
    def backward(self, next_derivative: Any = None) -> None:
        assert self.state == NodeState.FORWARD_FLOWED
        assert (next_derivative is None) == (self.next is None)

        next_derivative = [[1.0]] if next_derivative is None else next_derivative
        derivative_wrt_loss_input = self._calc_derivative_input(next_derivative)
        derivative_wrt_loss_param = self._calc_derivative_param(next_derivative)
        self.derivative_wrt_loss = derivative_wrt_loss_input

        if self.parameter is not None:
            # This just stores that derivative
            self.parameter.backward(derivative_wrt_loss_param)
        if self.prev is not None:
            self.prev.backward(self.derivative_wrt_loss)

    # NOTE: does not implement step
    def step(self) -> None:
        raise ValueError("You tried to call step on a function?!")

class MatrixAffine(Parameter):
    # NOTE that you must always multiple with input on the right so that
    # we are interperting the columns as the vectors of the vector space
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.matrix = [[random() for _ in range(width)] for _ in range(height)]
        self.affine = [random() for _ in range(height)]
        # List of rows, turn the affine into a row
        self.output_value = lambda: self.matrix + [self.affine]
    
class MatrixAffineMultiply(Function):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def _calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
        # Must have a matrix and a vector (affine) parameter
        assert param_value
        assert len(param_value) >= 2

        matrix = param_value[:-1]
        affine_row = param_value[-1:]
        assert len(affine_row) == 1
        
        # Turn it into an affine column
        affine = [[affine_row[0][i]] for i in range(len(affine_row[0]))]

        # The affine must be the height of the matrix
        assert len(matrix) == len(affine), f'{len(matrix)} {len(affine)}'

        def matmul(left: Matrix, right: Matrix) -> Matrix:
            left_height = len(left)
            left_width = len(left[0])
            right_height = len(right)
            right_width = len(right[0])

            assert left_width == right_height
            

            out = [[0.0 for _ in range(right_width)] for _ in range(left_height)]
            for i in range(left_height):
                for j in range(right_width):
                    # Calculate dot product of row i and column j
                    for k in range(left_width):
                        # Row varies in row dimension, column varies in column dimension
                        out[i][j] += left[i][k] * right[k][j]
            return out

        def broadcast_add(left: Matrix, right: Matrix) -> Matrix:
            left_height = len(left)
            left_width = len(left[0])
            right_height = len(right)
            right_width = len(right[0])

            assert left_height == right_height
            assert left_width == 1
            assert right_width >= 1

            out = [[right[i][j] for j in range(right_width)] for i in range(right_height)]

            # Must have same height
            for i in range(right_height):
                for j in range(right_width):
                    # Broadcast in the row dimension (i.e. height-match)
                    out[i][j] += left[i][0]
            
            return out
        
        mmed =  matmul(matrix, input)
        aed = broadcast_add(affine, mmed)
        out = aed

        return out

if __name__ == "__main__":
    matmul_seq = [
        # Height, width for each
        # Cancelled: 1st reduce from 28x28 to 4x28
        # (MatrixAffineMultiply(), MatrixAffine(4, 28)),
        # 2nd reduce from 4x28 to 1x28
        (MatrixAffineMultiply(), MatrixAffine(1, 28)),
    ]
    # TODO not having transposes (or generally left vs. right mult.) is a real problem

    matmul_graph = LossGraph(matmul_seq)
    input = [[random() for _ in range(28)] for _ in range(28)]
    print("---image---")
    print(input)
    print("-----------")

    output = matmul_graph.forward(input)
    print(output)