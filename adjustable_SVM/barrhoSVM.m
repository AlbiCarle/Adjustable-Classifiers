function r = barrhoSVM(Xtr, Ytr, X, b, alpha, kernel, param, eta)

if isequal(kernel, 'linear')

    w = -eta*(alpha'*diag(Ytr))*Xtr;

    r = b - X*w';

else

    K = KernelMatrix(Xtr,X,kernel,param);
    D = diag(Ytr);
    w = -eta*(K'*(D*alpha));

    r =  b - w;
end
