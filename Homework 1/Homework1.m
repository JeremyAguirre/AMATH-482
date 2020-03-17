%% AMATH 482: Homework #1
clear; close all; clc; 
load Testdata
L = 15;  % Spatial Domain 
n = 64;  % Fourier Modes 
x2 = linspace(-L,L,n+1); 
x = x2(1:n); 
y = x; 
z = x; 
k = (2*pi/(2*L)) * [0:(n/2-1) -n/2:-1]; 
ks = fftshift(k);
[X,Y,Z] = meshgrid(x,y,z); 
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%% Denoising Signal by Averaging
Utave = zeros(n,n,n);
for i=1:20 
    Un(:,:,:) = reshape(Undata(i,:),n,n,n); 
    Utave = fftn(Un) + Utave;
end
Utave = abs(fftshift(Utave)) / max(abs(Utave),[],'all');  % Denoised Signal

%% 3D Gaussian Filter Construction
[M,I] = max(Utave,[],'all','linear');
[Ix,Iy,Iz] = ind2sub([n n n],I);
a = Kx(Ix,Iy,Iz);  % Center Frequency for x
b = Ky(Ix,Iy,Iz);  % Center Frequency for Y
c = Kz(Ix,Iy,Iz);  % Center Frequency for Z
sigmax = 0.2;  % X-direction Bandwidth
sigmay = 0.2;  % Y-direction Bandwidth
sigmaz = 0.2;  % Z-direction Bandwidth
filter = exp(-sigmax*(Kx-a).^2 - sigmay*(Ky-b).^2 - sigmaz*(Kz-c).^2);
filter = fftshift(filter);  % 3D Gaussian Filter

%% Application of 3D Gaussian Filter to find Marble Locations
marble = zeros(20,3);
for j=1:20 
    Un(:,:,:) = reshape(Undata(j,:),n,n,n); 
    Unt = fftn(Un);
    Unft = filter .* Unt;
    Unf = ifftn(Unft);
    
    [M,J] = max(Unf,[],'all','linear');
    [Jx,Jy,Jz] = ind2sub([n n n],J);
    marblex = X(Jx,Jy,Jz);  % Marble X locations
    marbley = Y(Jx,Jy,Jz);  % Marble Y locations
    marblez = Z(Jx,Jy,Jz);  % Marble Z locations
    
    marble(j,1) = marblex;
    marble(j,2) = marbley;
    marble(j,3) = marblez;
end

%% Visualization of Marble Trajectory
plot3(marble(:,1),marble(:,2),marble(:,3),'linewidth',2), hold on
plot3(marble(20,1),marble(20,2),marble(20,3),'+','MarkerSize',15)
grid on
title('Marble Trajectory')
xlabel('x-coordinate')
ylabel('y-coordinate')
zlabel('z-coordinate')
print(gcf, '-dpng', 'marble_trajectory.png')