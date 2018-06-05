# class Foo(object):
#     def __init__(self, func):
#         self._func = func
#
#     def __call__(self):
#         print('class decorator running')
#         self._func()
#         print('class decorator ending')
#
#
# @Foo
# def bar():
#     print('bar')


# bar()


from functools import wraps


def logged(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__)  # 输出 'f'
        print(func.__doc__)  # 输出 'does some math'
        return func(*args, **kwargs)

    return with_logging


@logged
def f(x):
    """does some math"""
    return x + x * x


f(2)
