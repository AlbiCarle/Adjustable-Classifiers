function b_j = compute_b_j(Xtr, Ytr, Xvl, alpha_bar, kernel, param, eta)
    
    K = KernelMatrix(Xtr, Xvl, kernel, param);
    b_j = -eta*(alpha_bar'*diag(Ytr))*K;
    b_j = b_j';