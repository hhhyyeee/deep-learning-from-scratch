import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}is not supported')

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            # parameter 정의할 때와 argument로 사용할 때의 *(asterisk)의 동작 방식이 다름
            # parameter: positional arguments에 대한 가변 인자 활용
            # argument: list unpack

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # forward, backward 함수 둘 다 리턴되는 값의 길이가 1일 경우 wrapping 해주는 코드가 삽입되어 있는데 꼭 이렇게 번거롭게 해야 하는 걸까?
            # 그리고 forward는 Function 클래스에서 wrap하고, backward는 Variable 클래스에서 wrap하고 있는데 개발하는 입장에서는 매우 불편한 설계인 것 같다

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        return NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

