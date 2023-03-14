data = dlmread('psi.0002');
x = data(:,1);
y = data(:,2);
z1 = data(:,3);
%z2 = data(:,4);
%z3 = data(:,5);

N = 1000;
xvec = linspace(min(x), max(x), N);
yvec = linspace(min(y), max(y), N);
Z1 = griddata(x, y, z1, xvec, yvec.');
%Z2 = griddata(x, y, z2, xvec, yvec.');
%Z3 = griddata(x, y, z3, xvec, yvec.');

%l = length(x);

%sparse = sparse(l,2);
%x_compl = sparse(:,1);
%y_compl = sparse(:,2);


%for k=1:l
%    w = x(k)+1j*y(k);
%    w = 1j*(w-1j)/(w+1j);
%    x_compl(k) = real(w);
%    y_compl(k) = imag(w);
%end
%whos x1 x2 fplot
figure
%pcolor(x,y,z);
pcolor(xvec, yvec, Z1);
colormap(jet)
shading interp
colorbar

% figure
% %pcolor(x,y,z);
% pcolor(xvec, yvec, Z2);
% colormap(jet)
% shading interp
% colorbar
% 
% figure
% %pcolor(x,y,z);
% pcolor(xvec, yvec, Z3);
% colormap(jet)
% shading interp
% colorbar