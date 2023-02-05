import heapq
import weakref
import contextlib
import numpy as np


class Config:
    enable_backprop = True


# -----
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}is not supported')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

        self.count = 0
    
    """
    """
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+' '*9)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    """
    """
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, self.count, f))
                self.count += 1
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)[2]

            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # forward, backward 함수 둘 다 리턴되는 값의 길이가 1일 경우 wrapping 해주는 코드가 삽입되어 있는데 꼭 이렇게 번거롭게 해야 하는 걸까?
            # 그리고 forward는 Function 클래스에서 wrap하고, backward는 Variable 클래스에서 wrap하고 있는데 개발하는 입장에서는 매우 불편한 설계인 것 같다

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    # f.outputs() 리스트가 약한 참조(weakref)이기 떄문에 y()로 사용
                    # 참조 카운트가 0이 되고 메모리에서 데이터가 삭제됨
                    y().grad = None


# -----
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        return NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


# -----
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function):
    pass

class Sub(Function):
    pass

class Div(Function):
    def forward(self, numer, denom):
        if denom.data == 0:
            raise ZeroDivisionError
        y = numer / denom
        return y

class Pow(Function):
    pass

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# -----
@contextlib.contextmanager
# with 블록에 들어갈 때 name으로 지정한 Config 클래스 속성이 value로 설정
# with 블록을 빠져나오면서 원래 값(old_value)로 복원
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    else:
        return Variable(obj)

def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)

def neg():
    pass

def sub():
    pass

def div(numer, denom):
    return Div()(numer, denom)

def rdiv():
    pass

def pow():
    pass

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


# -----
def setup_variable():
    Variable.__add__ = add
    Variable.__mul__ = mul

