

def partition(lst, n):
    assert len(lst)<n,print("list length smaller than partition-n")
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def random_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(random.choice(g))
    return mat

def fix_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(g[-1])
    return mat