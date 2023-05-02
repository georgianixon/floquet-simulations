
def MaxDeltas2(ns, num_gaps = 4):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = list(enumerate(np.diff(ns)))
    pairso = max(pairs, key=lambda ab: ab[1])    
    first_gap = (pairso[1],( ns[pairso[0]], ns[pairso[0]+1]))
    pairs = np.delete(pairs, pairso[0], 0)
    pairso = max(pairs, key=lambda ab: ab[1])   
    second_gap = (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
    if num_gaps ==2:
        return first_gap, second_gap
    else:
        pairs = np.delete(pairs, pairso[0], 0)
        pairso = max(pairs, key=lambda ab: ab[1])
        third_gap =  (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
        if num_gaps == 3:
            return first_gap, second_gap, third_gap
        else:
            pairs = np.delete(pairs, pairso[0], 0)
            pairso = max(pairs, key=lambda ab: ab[1])
            fourth_gap =  (pairso[1],( ns[int(pairso[0])], ns[int(pairso[0]+1)]))
            return first_gap, second_gap, third_gap, fourth_gap
            

def SecondDelta(ns):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.
    '''
    pairs = [
        (abs(a - b), (a, b)) for a, b
        in zip(ns, ns[1:])
    ]
    delta = max(pairs, key=lambda ab: ab[0])
    d = dict(pairs)
    del d[delta[0]]
    second_max = max(d)
    return (second_max, d[second_max])

def MaxDelta(ns):
    '''Each of the maximally differing successive pairs
       in ns, each preceded by the value of the difference.'''
    pairs = [
        (abs(a - b), (a, b)) for a, b
        in zip(ns, ns[1:])
    ]
    delta = max(pairs, key=lambda ab: ab[0])
    return delta