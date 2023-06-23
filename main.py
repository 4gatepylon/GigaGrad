from abc import ABC
from enum import Enum
from typing import Any, Optional, Callable
from uuid import uuid4
from random import random

from main import Matrix

# NOTE we only support matrices for now, only linked-list computational graphs
Matrix = list[list[float]]

class NodeState(Enum):
    EMPTY = 0
    FORWARD_FLOWED = 1
    BACKWARD_FLOWED = 2
    STEPPED = 3

class LossGraph:
    def __init__(self, list[tuple[Function, Optional[Parameter]]]) -> None:
        self.funcs = []
        self.params = []
        for func, param in list:
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

Function = Any
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
    def __calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
        raise NotImplementedError
    def forward(self, input: Matrix) -> Matrix:
        assert self.state == NodeState.EMPTY

        outgoing = self.__calc_forward(input, self.parameter.output_value() if self.parameter else None)
        self.state = NodeState.FORWARD_FLOWED

        if self.next is None:
            return outgoing
        return self.next.forward(outgoing)
    
    # Calculate the derivative w.r.t. the loss for the parameter and for the input
    def __calc_derivative_param(self, next_derivative: Matrix) -> Matrix:
        raise NotImplementedError
    def __calc_derivative_input(self, next_derivative: Matrix) -> Matrix:
        raise NotImplementedError
    def backward(self, next_derivative: Any = None) -> None:
        assert self.state == NodeState.FORWARD_FLOWED
        assert (next_derivative is None) == (self.next is None)

        next_derivative = [[1.0]] if next_derivative is None else next_derivative
        derivative_wrt_loss_input = self.__calc_derivative_input(next_derivative)
        derivative_wrt_loss_param = self.__calc_derivative_param(next_derivative)
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
    def __init__(self, parameter: MatrixAffine, input: Parameter, users: list[Parameter]) -> None:
        super().__init__()
        self.parameter = parameter
        self.input = input
        self.users = users
    def __calc_forward(self, input: Matrix, param_value: Optional[Matrix] = None) -> Matrix:
        # Must have a matrix and a vector (affine) parameter
        assert param_value
        assert len(param_value) >= 2

        matrix = param_value[:-1]
        affine_row = param_value[-1:]
        assert len(affine_row) == 1
        
        # Turn it into an affine column
        affine = [[affine_row[i]] for i in range(len(affine_row))]

        # The affine must be the height of the matrix
        assert len(matrix) == len(affine)

        def matmul(left: Matrix, right: Matrix) -> Matrix:
            assert len(left[0]) == len(right)

            out = [[0.0 for _ in range(len(right[0]))] for _ in range(len(left))]
            for i in range(len(left)):
                for j in range(right[0]):
                    for k in range(len(left[0])):
                        out[i][j] += left[i][k] * right[k][j]
            return out
        def broadcast_add(left: Matrix, right: Matrix) -> Matrix:
            out = [[0.0 for _ in range(len(left[i]))] for i in range(len(left))]
            # Must have same height
            assert len(left) == len(right)
            assert all([len(right[i]) == 1 for i in range(len(right))])
            for i in range(len(left)):
                for j in range(len(left[i])):
                    out[i][j] += out[i][0]
            return out
        
        return broadcast_add(matmul(matrix, input), affine)
