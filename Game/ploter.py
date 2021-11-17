import numpy as np

def fun():
    obj = np.load('../Game/logs/evaluations.npz')
    print(obj)
    for item in obj:
        print(obj[item])
if __name__ == '__main__':
    fun()