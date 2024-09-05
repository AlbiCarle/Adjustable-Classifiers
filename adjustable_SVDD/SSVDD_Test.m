function y = SSVDD_Test(Xtr, Ytr, Xts, alpha, R, rho, kernel, param, eta)

if isequal(kernel, 'linear')

    a = eta*(alpha'*diag(Ytr))*Xtr;

    % Watch out: new notation provided by Teo !
    y = sign((R^2 - rho) - diag((Xts-a)*(Xts-a)'));

else

    Kz = KernelMatrix(Xtr,Xts,kernel,param);
    Kt = KernelMatrix(Xts,Xts,kernel,param);
    K =  KernelMatrix(Xtr,Xtr,kernel,param);
    D = diag(Ytr);
    
    % Watch out: new notation provided by Teo !
    y = sign((R^2 - rho)-(diag(Kt)-4*eta*Kz'*(D*alpha)+4*eta^2*alpha'*D*K*D*alpha));

end