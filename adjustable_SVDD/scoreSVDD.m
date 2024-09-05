function s = scoreSVDD(Xtr, Ytr, X, Y, R, alpha, kernel, param, eta)

    s = -Y.*barrho(Xtr, Ytr, X, R, alpha, kernel, param, eta);



