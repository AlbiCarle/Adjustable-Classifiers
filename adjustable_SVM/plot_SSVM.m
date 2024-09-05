function p = plot_SSVM(Xtr, Ytr, Xvl, alpha_bar, b, b_eps, kernel, param, eta, mycolor)

dimGrid=100; % dimGrid*dimGrid

[K1, Z1] = meshgrid(linspace(min(Xvl(:,1))-1, max(Xvl(:,1))+1,dimGrid),...
                    linspace(min(Xvl(:,2))-1, max(Xvl(:,2))+1,dimGrid));

x=linspace(min(Xvl(:,1))-1, max(Xvl(:,1))+1, dimGrid);
y=linspace(min(Xvl(:,2))-1, max(Xvl(:,2))+1, dimGrid);
   
K1=K1(:); Z1=Z1(:);
E=[K1 Z1];

if isequal(kernel, 'linear')
  
    w = (alpha_bar'*diag(Ytr))*Xtr;
       
    ax = gca;
    x1 = ax.XLim(1); 
    y1 = (1/(w(2)))*(b - b_eps - w(1)*x1);
    y2 = ax.YLim(1);
    x2 = (1/(w(1)))*(b - b_eps - w(2)*y2);
    %y2 = (1/(w(2)))*(b - w(1)*x2);
    
    p = fill([x1 x2 x1],[y2 y2 y1], mycolor,'LineWidth',1,...
       'FaceAlpha',0.05, 'EdgeColor', mycolor, 'LineWidth',2);
    %end

else

    y_pred = SSVM_Test(Xtr, Ytr, E, alpha_bar, b, b_eps, kernel, param, eta);

       p = contourf(x, y, reshape(y_pred,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', mycolor, 'LineWidth', 2,'FaceAlpha', 0.05);
    
    axis equal

    mycolormap = [mycolor(1)*ones(256,1), mycolor(2)*ones(256,1), mycolor(3)*ones(256,1)];

    colormap(mycolormap)

end