function c = compute_c(Xtr, Ytr, Xvl, alpha, kernel, param, eta)
    
    K = KernelMatrix(Xtr, Xvl, kernel, param);
    c = eta*(alpha'*diag(Ytr))*K;
    c = c';