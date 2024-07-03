import pynvml

if __name__=='__main__':
    max_clock = 1597
    min_clock = 135
    clock=max_clock
    CLOCKS_GPU =[]
    while clock > min_clock:
        CLOCKS_GPU.append(clock)
        clock = clock-7
        CLOCKS_GPU.append(clock)
        clock = clock - 8
    CLOCKS_GPU.reverse()
    print(len(CLOCKS_GPU))
    currentFrequency = 1117
    t = 0
    for i in range(len(CLOCKS_GPU)):
        if currentFrequency == CLOCKS_GPU[i]:
            t = i
            break
    # t = int(t*1.125)
    print(CLOCKS_GPU[t] - CLOCKS_GPU[int(t*0.8)])

    
    

