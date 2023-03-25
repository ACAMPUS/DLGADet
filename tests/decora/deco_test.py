import time
import numpy as np
from numba import jit

def outfunc(func):
    def infunc(times):
        t1 = time.time()
        sum = func(times)
        t2 = time.time() - t1
        print(t2)
        return sum

    return infunc

a=np.arange(99999999).reshape(33333333,3)


# @outfunc
@jit
def func1(times):
    sum = 0
    c,l=times.shape
    for i in range(c):
        for j in range(l):
            sum += times[i,j]
    return sum


if __name__ == '__main__':
    times = a
    # innera=outfunc(func1)
    # res=innera(times)
    # res = func1(times)
    # print(res)
    t1=time.time()
    func1(times)
    t2=time.time()
    print(t2-t1)
