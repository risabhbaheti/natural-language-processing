import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    #print N
    #print L
    emission_t = emission_scores.transpose()
    scores = np.zeros((L,N))
    tags = np.zeros((L,N), dtype = int)
    
    for i in range(L):
        scores[i][0] = emission_t[i][0]+start_scores[i]
        
    y = []
    for i in range(1,N):
        for j in range(L):
            score = np.zeros(L)
            #print i, j
            for k in range(L):
                score[k]=scores[k][i-1] + trans_scores[k][j]
            #print score
            pos_max = score.argmax()
            score_max = score.max()
            #print pos_max, score_max
            tags[j][i] = pos_max
            scores[j][i] = score_max +emission_t[j][i]
        

    #print tags
    #print scores
    score = np.zeros(L)
    for i in range(L):
        scores[i][N-1] +=end_scores[i]
        score[i] = scores[i][N-1]
    score_max_final = score.max()
    pos_max = score.argmax()
    #print pos_max
    y.append(pos_max)
    for i in range(N-1, 0,-1):
        pos_max_temp = tags[pos_max][i]
        pos_max = pos_max_temp
        #print pos_max
        y.append(pos_max)
    y.reverse()
    #print y
    return (score_max_final, y)

