import numpy as np
from dezero.core import Function, as_variable
from dezero import utils


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx =  gy * cos(x)
        return gx

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx


def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        return gx

class Log(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    else:
        return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        # x.dot(W) is equivalent to np.dot(x, W)
        # but it is faster than np.dot
        # because it uses BLAS (Basic Linear Algebra Subprograms)
        # which is a collection of low-level routines for performing basic linear algebra operations
        # such as matrix multiplication, vector multiplication, matrix addition, etc.
        # https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
        # 이 지점에서 x.dot은 np.dot을 호출하는 것이다.
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        # xp = dezero.cuda.get_array_module(gy)
        # gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        # if xp is np:
        #     np.add.at(gx, self.slices, gy)
        # else:
        #     xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)

def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


# =============================================================================
# Softmax
# =============================================================================
class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy
# =============================================================================

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)
        # return sum(diff ** 2) / len(diff)
        # 이러면 안되는 이유는?
        # sum은 numpy의 함수이고, diff ** 2는 Variable이다.
        ## Function 클래스에서 data에 numpy array를 저장하고, sum 함수는 .data 형태로 numpy array를 호출하여 사용한다.
        # Variable은 numpy의 함수를 호출할 수 없다.
        # 그래서 numpy의 함수를 호출하고 싶으면, numpy의 함수를 호출한 결과를 Variable로 감싸야 한다.
        # 그래서 위와 같이 (diff ** 2).sum()을 사용한다.
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]

        # logexpsum 함수는 max의 smooth approximation이다.
        # 벡터에 exp를 취하게 되면 숫자가 큰 값의 영향력이 지수적으로 커지게 되는데, 이들을 덧셈하면 원래 작았던 숫자의 영향력은 매우 갑소하게 된다.
        # 그다음 log를 취해서 exp의 역과정을 하게 되면, 사실상 영향력이 큰 숫자만 의미를 가지게 되므로 max와 유사한 기능을 한다고 볼 수 있다.
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z

        # 정답인 log proba 값만 추출
        log_p = log_p[np.arange(N), t.ravel()]

        # 미니배치 단위로 softmax cross entropy 스코어 평균내기
        y = -log_p.sum() / np.float32(N)

        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        # xp = cuda.get_array_module(t.data)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# clip
# =============================================================================
class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)