import types

def clone_function(fn, **overrides):
    result=types.FunctionType(fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__;result.__annotations__=fn.__annotations__;result.__doc__=fn.__doc__;result.__qualname__=fn.__qualname__
    return result
