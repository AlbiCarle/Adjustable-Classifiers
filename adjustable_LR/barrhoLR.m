function r = barrhoLR(Xtr, Ytr, X, b, alpha, kernel, param, eta)

if isequal(kernel, 'linear')

    w = 2*eta*(alpha'*diag(Ytr))*Xtr;

    r = -b + X*w';

else

    K = KernelMatrix(Xtr,X,kernel,param);
    D = diag(Ytr);
    w = 2*eta*(K'*(D*alpha));

    r =  -b + w;
end
