%% MNIST Fashion Classifier with Fully-Connected Neural Network
clear all; close all; clc
load('fashion_mnist.mat')
figure(1)
for k = 1:9
    subplot(3,3,k)
    imshow(reshape(X_train(k,:,:),[28 28]));
end
X_train = im2double(X_train);
X_test = im2double(X_test);
X_train = reshape(X_train,[60000 28 28 1]);
X_train = permute(X_train,[2 3 4 1]);
X_test = reshape(X_test,[10000 28 28 1]);
X_test = permute(X_test,[2 3 4 1]);
X_valid = X_train(:,:,:,1:5000);
X_train = X_train(:,:,:,5001:end);
y_valid = categorical(y_train(1:5000))';
y_train = categorical(y_train(5001:end))';
y_test = categorical(y_test)';
layers = [imageInputLayer([28 28 1])
        fullyConnectedLayer(500)
        tanhLayer
        fullyConnectedLayer(300)
        tanhLayer
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization', 1e-3, ...
    'ValidationData',{X_valid,y_valid}, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(X_train,y_train,layers,options);

%% Training Confusion
figure(2)
y_pred = classify(net,X_train);
plotconfusion(y_train,y_pred)

%% Test Confusion
figure(3)
y_pred = classify(net,X_test);
plotconfusion(y_test,y_pred)
print('-f1','hw51','-dpng')
print('-f2','hw52','-dpng')
print('-f3','hw53','-dpng')

%% MNIST Fashion Classifier with Convolutional Neural Network
clear; close all; clc
load('fashion_mnist.mat')
X_train = im2double(X_train);
X_test = im2double(X_test);
X_train = reshape(X_train,[60000 28 28 1]);
X_train = permute(X_train,[2 3 4 1]);
X_test = reshape(X_test,[10000 28 28 1]);
X_test = permute(X_test,[2 3 4 1]);
X_valid = X_train(:,:,:,1:5000);
X_train = X_train(:,:,:,5001:end);
y_valid = categorical(y_train(1:5000))';
y_train = categorical(y_train(5001:end))';
y_test = categorical(y_test)';
layers = [
    imageInputLayer([28 28 1],"Name","imageinput")
    convolution2dLayer([5 5],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],128,"Name","conv_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],256,"Name","conv_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc_1")
    reluLayer("Name","relu_6")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
options = trainingOptions('adam', ...
    'MaxEpochs',1,...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',1e-3, ...
    'ValidationData',{X_valid,y_valid}, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(X_train,y_train,layers,options);
%% Training Confusion
figure(4)
y_pred = classify(net,X_train);
plotconfusion(y_train,y_pred)
%% Test Confusion
figure(5)
y_pred = classify(net,X_test);
plotconfusion(y_test,y_pred)
print('-f4','hw54','-dpng')
print('-f5','hw55','-dpng')