%% AMATH 482: Assignment #3
close all, clear, clc

load cam1_1;
numFrames1_1 = size(vidFrames1_1, 4);
load cam2_1;
numFrames2_1 = size(vidFrames2_1, 4);
load cam3_1;
numFrames3_1 = size(vidFrames3_1, 4);
load cam1_2;
numFrames1_2 = size(vidFrames1_2, 4);
load cam2_2;
numFrames2_2 = size(vidFrames2_2, 4);
load cam3_2;
numFrames3_2 = size(vidFrames3_2, 4);
load cam1_3;
numFrames1_3 = size(vidFrames1_3, 4);
load cam2_3;
numFrames2_3 = size(vidFrames2_3, 4);
load cam3_3;
numFrames3_3 = size(vidFrames3_3, 4);
load cam1_4;
numFrames1_4 = size(vidFrames1_4, 4);
load cam2_4;
numFrames2_4 = size(vidFrames2_4, 4);
load cam3_4;
numFrames3_4 = size(vidFrames3_4, 4);

%% Test 1: Ideal Case
% Camera 1
x1_1 = [];
y1_1 = [];
for j=1:numFrames1_1
    A = double(rgb2gray(vidFrames1_1(:,:,:,j)));
    A(:, [1:300 400:end]) = 0;
    A([1:200 400:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x1_1(j) = mean(N);
    y1_1(j) = mean(M);
end

[Y, I] = max(y1_1(1:50));
x1_1 = x1_1(30:end);
y1_1 = y1_1(30:end);

% Camera 2
x2_1 = [];
y2_1 = [];
for j=1:numFrames2_1
    A = double(rgb2gray(vidFrames2_1(:,:,:,j)));
    A(:, [1:220 350:end]) = 0;
    A([1:100 350:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x2_1(j) = mean(N);
    y2_1(j) = mean(M);
end

[Y, I] = max(y2_1(1:41));
x2_1 = x2_1(I:end);
y2_1 = y2_1(I:end);

% Camera 3
x3_1 = [];
y3_1 = [];
for j=1:numFrames3_1
    A = double(rgb2gray(vidFrames3_1(:,:,:,j)));
    A(:, [1:200 480:end]) = 0;
    A([1:230 350:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x3_1(j) = mean(N);
    y3_1(j) = mean(M);
end

[Y, I] = max(y3_1(1:41));
x3_1 = x3_1(I:end);
y3_1 = y3_1(I:end);

%% PCA
L = min([length(y1_1), length(y2_1), length(x3_1)]);
X = [x1_1(1:L); y1_1(1:L); x2_1(1:L); y2_1(1:L); x3_1(1:L); y3_1(1:L)];
[m, n] = size(X);
mn = mean(X, 2);
X = X-repmat(mn,1,n);
[U, S, V] = svd(X, 'econ');
V = V*S;

figure(1)
plot(diag(S)./sum(diag(S)), 'mo')
xlabel('Principal Component') 
ylabel('Fractional Energy')
title('Case 1')

figure(2)
plot(V(:,1)), hold on
plot(V(:,2))
plot(V(:,3))
plot(V(:,4))
plot(V(:,5)) 
plot(V(:,6)), hold off
xlabel('Time (frames)')
ylabel('Displacement (pixels)')
title('Case 1')
lgd = legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6');
title(lgd, 'Principal Components')

print('-f1','Energy_1','-dpng')
print('-f2','Case_1','-dpng')

%% Test 2: Noisy Case
% Camera 1
x1_2 = [];
y1_2 = [];
for j=1:numFrames1_2
    A = double(rgb2gray(vidFrames1_2(:,:,:,j)));
    A(:, [1:300 400:end]) = 0;
    A([1:200 400:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x1_2(j) = mean(N);
    y1_2(j) = mean(M);
end

[Y, I] = max(y1_2(1:30));
x1_2 = x1_2(I:end);
y1_2 = y1_2(I:end);

% Camera 2
x2_2 = [];
y2_2 = [];
for j=1:numFrames2_2
    A = double(rgb2gray(vidFrames2_2(:,:,:,j)));
    A(:, [1:200 400:end]) = 0;
    A([1:50 370:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x2_2(j) = mean(N);
    y2_2(j) = mean(M);
end

[Y, I] = max(y2_2(1:30));
x2_2 = x2_2(I:end);
y2_2 = y2_2(I:end);

% Camera 3
x3_2 = [];
y3_2 = [];
for j=1:numFrames3_2
    A = double(rgb2gray(vidFrames3_2(:,:,:,j)));
    A(:, [1:250 480:end]) = 0;
    A([1:180 320:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x3_2(j) = mean(N);
    y3_2(j) = mean(M);
end

[Y, I] = max(y3_2(1:40));
x3_2 = x3_2(I:end);
y3_2 = y3_2(I:end);

%% PCA
L = min([length(y1_2), length(y2_2), length(x3_2)]);
X = [x1_2(1:L); y1_2(1:L); x2_2(1:L); y2_2(1:L); x3_2(1:L); y3_2(1:L)];
[m, n] = size(X);
mn = mean(X, 2);
X = X-repmat(mn,1,n);
[U, S, V] = svd(X, 'econ');
V = V*S;

figure(3)
plot(diag(S)./sum(diag(S)), 'mo')
xlabel('Principal Component')
ylabel('Fractional Energy')
title('Case 2')

figure(4)
plot(V(:,1)), hold on
plot(V(:,2))
plot(V(:,3))
plot(V(:,4))
plot(V(:,5)) 
plot(V(:,6)), hold off
xlabel('Time (frames)')
ylabel('Displacement (pixels)')
title('Case 2')
lgd = legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6');
title(lgd, 'Principal Components')

print('-f3','Energy_2','-dpng')
print('-f4','Case_2','-dpng')

%% Test 3: Horizontal Displacement
% Camera 1
x1_3 = [];
y1_3 = [];
for j=1:numFrames1_3
    A = double(rgb2gray(vidFrames1_3(:,:,:,j)));
    A(:, [1:250 400:end]) = 0;
    A([1:200 400:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x1_3(j) = mean(N);
    y1_3(j) = mean(M);
end

[Y, I] = max(y1_3(1:30));
x1_3 = x1_3(I:end);
y1_3 = y1_3(I:end);

% Camera 2
x2_3 = [];
y2_3 = [];
for j=1:numFrames2_3
    A = double(rgb2gray(vidFrames2_3(:,:,:,j)));
    A(:, [1:220 420:end]) = 0;
    A([1:150 400:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x2_3(j) = mean(N);
    y2_3(j) = mean(M);
end

[Y, I] = max(y2_3(1:50));
x2_3 = x2_3(I:end);
y2_3 = y2_3(I:end);

% Camera 3
x3_3 = [];
y3_3 = [];
for j=1:numFrames3_3
    A = double(rgb2gray(vidFrames3_3(:,:,:,j)));
    A(:, [1:150 480:end]) = 0;
    A([1:180 350:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x3_3(j) = mean(N);
    y3_3(j) = mean(M);
end

[Y, I] = max(y3_3(1:30));
x3_3 = x3_3(I:end);
y3_3 = y3_3(I:end);

%% PCA
L = min([length(y1_3), length(y2_3), length(x3_3)]);
X = [x1_3(1:L); y1_3(1:L); x2_3(1:L); y2_3(1:L); x3_3(1:L); y3_3(1:L)];
[m, n] = size(X);
mn = mean(X, 2);
X = X-repmat(mn,1,n);
[U, S, V] = svd(X, 'econ');
V = V*S;

figure(5)
plot(diag(S)./sum(diag(S)), 'mo')
xlabel('Principal Component')
ylabel('Fractional Energy')
title('Case 3')

figure(6)
plot(V(:,1)), hold on
plot(V(:,2))
plot(V(:,3))
plot(V(:,4))
plot(V(:,5)) 
plot(V(:,6)), hold off
xlabel('Time (frames)')
ylabel('Displacement (pixels)')
title('Case 3')
lgd = legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6');
title(lgd, 'Principal Components')

print('-f5','Energy_3','-dpng')
print('-f6','Case_3','-dpng')

%% Test 4: Horizontal Displacement and Rotation
% Camera 1
x1_4 = [];
y1_4 = [];
for j=1:numFrames1_4
    A = double(rgb2gray(vidFrames1_4(:,:,:,j)));
    A(:, [1:300 470:end]) = 0;
    A([1:200 380:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x1_4(j) = mean(N);
    y1_4(j) = mean(M);
end

[Y, I] = max(y1_4(1:41));
x1_4 = x1_4(I:end);
y1_4 = y1_4(I:end);

% Camera 2
x2_4 = [];
y2_4 = [];
for j=1:numFrames2_4
    A = double(rgb2gray(vidFrames2_4(:,:,:,j)));
    A(:, [1:220 410:end]) = 0;
    A([1:50 400:end], :) = 0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x2_4(j) = mean(N);
    y2_4(j) = mean(M);
end

[Y, I] = max(y2_4(1:50));
x2_4 = x2_4(I:end);
y2_4 = y2_4(I:end);

% Camera 3
x3_4 = [];
y3_4 = [];
for j=1:numFrames3_4
    A = double(rgb2gray(vidFrames3_4(:,:,:,j)));
    A(:, [1:300 510:end]) = 0;
    A([1:150 290:end], :) = 0; 
    A(1:150, :) = 0; A(290:end, :) = 0; A(:, 1:300) = 0; A(:, 510:end)=0;
    [Y, I] = max(A(:));
    [M, N] = find(A >= 11/12 * Y);
    x3_4(j) = mean(N);
    y3_4(j) = mean(M);
end

[Y, I] = max(y3_4(1:50));
x3_4 = x3_4(I:end);
y3_4 = y3_4(I:end);

%% PCA
L = min([length(y1_4), length(y2_4), length(x3_4)]);
X = [x1_4(1:L); y1_4(1:L); x2_4(1:L); y2_4(1:L); x3_4(1:L); y3_4(1:L)];
[m, n] = size(X);
mn = mean(X, 2);
X = X-repmat(mn,1,n);
[U, S, V] = svd(X, 'econ');
V = V*S;

figure(7)
plot(diag(S)./sum(diag(S)), 'mo')
xlabel('Principal Component')
ylabel('Fractional Energy')
title('Case 4')

figure(8)
plot(V(:,1)), hold on
plot(V(:,2))
plot(V(:,3))
plot(V(:,4))
plot(V(:,5))
plot(V(:,6)), hold off
xlabel('Time (frames)')
ylabel('Displacement (pixels)')
title('Case 4')
lgd = legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6');
title(lgd, 'Principal Components')

print('-f7','Energy_4','-dpng')
print('-f8','Case_4','-dpng')