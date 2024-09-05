function R = compute_R(Xtr, Ytr, Xvl, alpha, kernel, param, eta)
    
Kz = KernelMatrix(Xtr,Xvl,kernel,param);
Kv = KernelMatrix(Xvl,Xvl,kernel,param);
K =  KernelMatrix(Xtr,Xtr,kernel,param);
D = diag(Ytr);

R = sqrt((diag(Kv)-4*eta*Kz'*(D*alpha)+4*eta^2*alpha'*D*K*D*alpha));
