import os
import pandas as pd
import helper as hlp

def func(dataPoint):

    seed = 0
    i = 0
    while True:
        seed ^= dataPoint[i][0] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= dataPoint[i][1] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= dataPoint[i][2] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        i+=1
        print("\n\nin")
        if i == 4:
            break
    return seed


if __name__ == "__main__":
    hash =  9932841145013996982
    x    =  205026338093898251492784180662
    l=[(121,122,0),(2,2,0),(128,128,0),(-1,-1,0)]
    print(func(l))
