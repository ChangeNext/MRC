def calc_distance(a, b):
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b:
        return 0 # 같으면 0을 반환
    
    a_len = len(a) # a 길이
    b_len = len(b) # b 길이
    if a == "":
        return b_len
    if b == "":
        return a_len
    
    matrix = [[] for i in range(a_len+1)]
    
    for i in range(a_len+1):
        matrix[i] = [0 for j in range(b_len+1)]
        
    for i in range(a_len+1):
        matrix[i][0] = i
        
    for j in range(b_len+1):
        matrix[0][j] = j
        
    for i in range(1, a_len+1):
        ac = a[i-1]
        for j in range(1, b_len+1):
            bc = b[j-1]
            cost = 0 if (ac == bc) else 1
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 삽입
                matrix[i][j-1] + 1,     # 문자 제거
                matrix[i-1][j-1] + cost # 문자 변경
            ])
            
    return matrix[a_len][b_len]

def val_calc(a, b):
    score = []
    for i in range(int(len(b))):
        s = calc_distance(a, b)
        score.append(s)
    return mean(score)