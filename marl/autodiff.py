"""
Minimal reverse-mode autodiff over NumPy arrays.

This is a small, dependency-free ("no PyTorch needed") automatic
differentiation engine, just powerful enough to train the small MLP
actor/critic networks used by the MAPPO experiment in this folder.

Supported ops: matmul, add, relu, tanh, exp, log, sum, mean, softmax,
elementwise multiply, subtraction, minimum, clip, and gather (index
selection). Each op records its inputs and a local backward function;
`Tensor.backward()` walks the resulting graph in reverse topological
order and accumulates gradients in `.grad`.
"""

from __future__ import annotations
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_children", "_backward", "_op", "requires_grad")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._children = _children
        self._backward = lambda: None
        self._op = _op
        self.requires_grad = requires_grad

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---- ops -----------------------------------------------------------
    def __add__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "add")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._wrap(other))

    def __rsub__(self, other):
        return Tensor._wrap(other) + (-self)

    def __mul__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "mul")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tensor._wrap(other)
        return self * (other ** -1.0)

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), "pow")

        def _backward():
            self.grad += _unbroadcast((power * self.data ** (power - 1)) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = Tensor._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-8), (self,), "log")

        def _backward():
            self.grad += out.grad / (self.data + 1e-8)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,), "softmax")

        def _backward():
            # dL/dz = p * (dL/dp - sum(dL/dp * p))
            g = out.grad
            dot = np.sum(g * p, axis=axis, keepdims=True)
            self.grad += p * (g - dot)
        out._backward = _backward
        return out

    def gather(self, indices):
        """Select one value per row: out[i] = self[i, indices[i]]."""
        indices = np.asarray(indices, dtype=np.int64)
        rows = np.arange(self.data.shape[0])
        out = Tensor(self.data[rows, indices], (self,), "gather")

        def _backward():
            g = np.zeros_like(self.data)
            g[rows, indices] += out.grad
            self.grad += g
        out._backward = _backward
        return out

    def clip(self, lo, hi):
        out = Tensor(np.clip(self.data, lo, hi), (self,), "clip")

        def _backward():
            mask = (self.data >= lo) & (self.data <= hi)
            self.grad += out.grad * mask
        out._backward = _backward
        return out

    def minimum(self, other):
        other = Tensor._wrap(other)
        out_data = np.minimum(self.data, other.data)
        out = Tensor(out_data, (self, other), "minimum")

        def _backward():
            mask_self = (self.data <= other.data).astype(np.float64)
            self.grad += _unbroadcast(out.grad * mask_self, self.data.shape)
            other.grad += _unbroadcast(out.grad * (1.0 - mask_self), other.data.shape)
        out._backward = _backward
        return out

    # ---- graph traversal ------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Adam:
    """Standard Adam optimizer over a list of Tensor parameters."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self, clip_norm=None):
        self.t += 1
        if clip_norm is not None:
            total = np.sqrt(sum(float(np.sum(p.grad ** 2)) for p in self.params))
            if total > clip_norm:
                scale = clip_norm / (total + 1e-8)
                for p in self.params:
                    p.grad *= scale
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
