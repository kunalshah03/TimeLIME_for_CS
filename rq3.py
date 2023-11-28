from planner import *
from othertools import *
import matplotlib.pyplot as plt


def main():
    scores_t = readfile('rq2_TimeLIME.csv')
    bcs_t = readfile('rq3_TimeLIME.csv')

    list1 = [scores_t]
    list2 = [bcs_t]
    names = ['TimeLIME']
    results=[]
    for i in range(len(names)):
        scores = list1[i]
        bcs = list2[i]
        dummy = []
        N = len(scores)
        for k in range(0, len(scores)):
            temp = 0
            for j in range(0, len(scores[k])):
                temp -= (bcs[k][j] * scores[k][j])
            total = -np.sum(bcs[k])
            dummy.append(100*(np.round(temp / total, 3)))
        # print(names[i],dummy)
        results.append(dummy)
    print(results)
    return results

if __name__ == "__main__":
    main()
