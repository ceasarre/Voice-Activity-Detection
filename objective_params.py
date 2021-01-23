def analyze_objective_params(y, y_pred):

    
    noise = 0
    signal = 0

    # objective params
    fec = 0
    msc = 0
    over = 0
    nds = 0


    for i in range(len(y)-1):

        if i == 0:
            if y[i] == 0:
                noise += 1
            else:
                signal += 1
            continue

        if y[i] == 0:
            noise += 1
        else:
            signal += 1

        if y[i] == 1 and y_pred[i] == 0:
            if y[i-1] == 0 and y[i + 1] == 1:
                fec+=1

            elif y[i - 1] == 1 and y[i+1] == 1:
                msc += 1

        if y[i] == 0 and y_pred[i] == 1:
            if y[i-1] == 0 and y[i + 1] == 0:
                nds += 1

            elif y[i - 1] == 1 and y[i+1] == 0:
                over += 1

    return signal, noise, fec, msc, over, nds
        



