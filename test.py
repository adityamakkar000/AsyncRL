from functools import partial

fs = dict()


def f(x):
    print(x)


for i in range(10):
    fs[i] = partial(f, x=i)

breakpoint()
