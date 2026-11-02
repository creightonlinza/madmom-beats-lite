# encoding: utf-8
# cython: language_level=3, boundscheck=False, wraparound=False
"""
Native forward pass of an LSTM layer.

It performs the same floating point operations in the same order as the Python
implementation in :class:`madmom.ml.nn.layers.LSTMLayer` and therefore returns
bit-identical results: matrix products go through NumPy's dot implementation
with arguments of the same shape and memory layout, the activation functions
are the very inner loops of the NumPy and SciPy ufuncs, and additions and
multiplications are rounded separately (this module must be compiled without
floating point contraction).

"""

import numpy as np
from scipy.special import expit

cimport numpy as cnp

cnp.import_array()


cdef extern from "numpy/arrayobject.h":
    object PyArray_MatrixProduct2(object, object, cnp.ndarray)


cdef extern from "numpy/ufuncobject.h":
    ctypedef void (*PyUFuncGenericFunction)(char **, cnp.npy_intp *,
                                            cnp.npy_intp *, void *) noexcept
    ctypedef struct PyUFuncObject:
        int ntypes
        PyUFuncGenericFunction *functions
        void **data
        char *types


cdef struct FloatLoop:
    PyUFuncGenericFunction function
    void *data


cdef FloatLoop float_loop(object ufunc) except *:
    """Return the inner loop of a unary ufunc for single precision floats."""
    cdef PyUFuncObject *u = <PyUFuncObject *>ufunc
    cdef FloatLoop loop
    cdef int i
    for i in range(u.ntypes):
        if u.types[2 * i] == cnp.NPY_FLOAT and \
                u.types[2 * i + 1] == cnp.NPY_FLOAT:
            loop.function = u.functions[i]
            loop.data = u.data[i]
            return loop
    raise TypeError('%s has no single precision loop' % ufunc)


cdef inline void apply_loop(FloatLoop loop, float *x, float *y,
                            cnp.npy_intp n) noexcept:
    """Apply the inner loop to `n` values of `x`, store the results in `y`."""
    cdef char *args[2]
    cdef cnp.npy_intp steps[2]
    args[0] = <char *>x
    args[1] = <char *>y
    steps[0] = sizeof(float)
    steps[1] = sizeof(float)
    loop.function(args, &n, steps, loop.data)


cdef FloatLoop SIGMOID = float_loop(expit)
cdef FloatLoop TANH = float_loop(np.tanh)


cdef inline void gate_sum(object x, object weights, object prev,
                          object recurrent_weights, cnp.ndarray dot_data,
                          cnp.ndarray dot_prev, const float *bias,
                          const float *state, const float *peephole,
                          float *out, cnp.npy_intp n) except *:
    """
    Sum up the inputs of a gate, adding the terms in the order of `Gate.activate`.

    Without a peephole (a NULL pointer), the state term is left out.

    """
    PyArray_MatrixProduct2(x, weights, dot_data)
    PyArray_MatrixProduct2(prev, recurrent_weights, dot_prev)
    cdef const float *d = <const float *>cnp.PyArray_DATA(dot_data)
    cdef const float *r = <const float *>cnp.PyArray_DATA(dot_prev)
    cdef cnp.npy_intp j
    for j in range(n):
        out[j] = d[j] + bias[j]
    if peephole != NULL:
        for j in range(n):
            out[j] += state[j] * peephole[j]
    for j in range(n):
        out[j] += r[j]


def activate(layer, data):
    """
    Process `data` with the given :class:`LSTMLayer`, starting from its current
    state, and update the state of the layer.

    All arrays must be single precision and all gates must use peepholes and
    the default activation functions, see :meth:`LSTMLayer.activate`.

    Parameters
    ----------
    layer : :class:`LSTMLayer`
        LSTM layer.
    data : numpy array, shape (num_frames, num_inputs)
        Activate with this data.

    Returns
    -------
    numpy array, shape (num_frames, num_hiddens)
        Activations for this data.

    """
    input_gate, forget_gate = layer.input_gate, layer.forget_gate
    cell, output_gate = layer.cell, layer.output_gate
    w_input, r_input = input_gate.weights, input_gate.recurrent_weights
    w_forget, r_forget = forget_gate.weights, forget_gate.recurrent_weights
    w_cell, r_cell = cell.weights, cell.recurrent_weights
    w_output, r_output = output_gate.weights, output_gate.recurrent_weights
    cdef const float[::1] b_input = input_gate.bias
    cdef const float[::1] b_forget = forget_gate.bias
    cdef const float[::1] b_cell = cell.bias
    cdef const float[::1] b_output = output_gate.bias
    cdef const float[::1] p_input = input_gate.peephole_weights
    cdef const float[::1] p_forget = forget_gate.peephole_weights
    cdef const float[::1] p_output = output_gate.peephole_weights

    cdef cnp.npy_intp size = len(data)
    cdef cnp.npy_intp n = cell.bias.size
    cdef cnp.npy_intp i, j
    out = np.zeros((size, n), dtype=np.float32)
    cdef float[:, ::1] out_view = out

    cdef cnp.ndarray dot_data = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray dot_prev = np.empty(n, dtype=np.float32)
    # scratch space: gate input, input gate, forget gate, cell, output gate
    cdef float[:, ::1] work = np.empty((5, n), dtype=np.float32)
    cdef float *tmp = &work[0, 0]
    cdef float *ig = &work[1, 0]
    cdef float *fg = &work[2, 0]
    cdef float *cl = &work[3, 0]
    cdef float *og = &work[4, 0]
    state_array = np.array(layer._state, dtype=np.float32)
    cdef float[::1] state = state_array
    cdef float *st = &state[0]
    cdef float *o

    prev = layer._prev
    for i in range(size):
        x = data[i]
        # input gate
        gate_sum(x, w_input, prev, r_input, dot_data, dot_prev,
                 &b_input[0], st, &p_input[0], tmp, n)
        apply_loop(SIGMOID, tmp, ig, n)
        # forget gate
        gate_sum(x, w_forget, prev, r_forget, dot_data, dot_prev,
                 &b_forget[0], st, &p_forget[0], tmp, n)
        apply_loop(SIGMOID, tmp, fg, n)
        # cell
        gate_sum(x, w_cell, prev, r_cell, dot_data, dot_prev,
                 &b_cell[0], st, NULL, tmp, n)
        apply_loop(TANH, tmp, cl, n)
        # internal state
        for j in range(n):
            st[j] = cl[j] * ig[j] + st[j] * fg[j]
        # output gate
        gate_sum(x, w_output, prev, r_output, dot_data, dot_prev,
                 &b_output[0], st, &p_output[0], tmp, n)
        apply_loop(SIGMOID, tmp, og, n)
        # output
        o = &out_view[i, 0]
        apply_loop(TANH, st, o, n)
        for j in range(n):
            o[j] *= og[j]
        prev = out[i]
    layer._prev = prev
    layer._state = state_array
    return out
