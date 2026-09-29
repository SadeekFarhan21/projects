"""tcm: a small tensor compiler that targets the Apple GPU through Metal."""

from .compiler import CompileOptions, Program, compile, device, run_graph
from .frontend import (
    Tensor,
    Tracer,
    constant,
    exp,
    gelu,
    layernorm,
    log,
    matmul,
    maximum,
    relu,
    softmax,
    sqrt,
    tanh,
    trace,
)
from .interp import interpret
from .ir import F16, F32, Graph, Node
from .passes import optimize

__all__ = [
    "CompileOptions",
    "F16",
    "F32",
    "Graph",
    "Node",
    "Program",
    "Tensor",
    "Tracer",
    "compile",
    "constant",
    "device",
    "exp",
    "gelu",
    "interpret",
    "layernorm",
    "log",
    "matmul",
    "maximum",
    "optimize",
    "relu",
    "run_graph",
    "softmax",
    "sqrt",
    "tanh",
    "trace",
]
