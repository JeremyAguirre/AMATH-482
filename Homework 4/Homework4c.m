%% AMATH 482: Assignment #4 - Test 3
path = 'c:\Users\jerem\OneDrive\Desktop\AMATH 482\Music3';
file = dir(path);
data = [];
for i = 3:length(file)
    filename = file(i).name;
    [song, Fs] = audioread(strcat(path, '/', filename));
    song = song(:,1) + song(:,2);
    song = resample(song, 20000, Fs);
    Fs = 20000;
    for j = 1:5:125
        test = song(Fs*j:Fs*(j+5), 1);
        spec_test = abs(spectrogram(test));
        spec_test = reshape(spec_test, 1, 8*32769);
        data = [data spec_test'];
    end
end

%% Plotting Dominant Spectrogram Modes
[u, s, v] = svd(data - mean(data(:)), 'econ');
plot(diag(s) ./ sum(diag(s)), 'ko');
plot3(v(1:50, 3), v(1:50, 4), v(1:50, 5), 'ko'), hold on
plot3(v(51:100, 3), v(51:100, 4), v(51:100, 5), 'ro') 
plot3(v(101:150, 3), v(101:150, 4), v(101:150, 5), 'go')
xlabel('Mode 3')
ylabel('Mode 4')
zlabel('Mode 5')

%% k-Nearest Neighbors
result_knn = [];
for i = 1:1000
    q1 = randperm(50);
    q2 = randperm(50);
    q3 = randperm(50);
    xptv = v(1:50, 3:5);
    xbach = v(51:100, 3:5);
    xjc = v(101:150, 3:5);
    xtrain = [xptv(q1(1:30), :) xbach(q2(1:30), :) xjc(q3(1:30),:)];
    xtest = [xptv(q1(31:end), :) xbach(q2(31:end), :) xjc(q3(31:end),:)];
    ctrain = [ones(30,1) 2*ones(30,1) 3*ones(30,1)];
    ctest = [ones(20,1) 2*ones(20,1) 3*ones(20,1)];
    
    index = knnsearch(xtrain, xtest, 'A', 3);
    pre = [];
    true = 0;
    for j = 1:length(index)
        ind = index(j,:);
        x0 = [ctrain(ind(1)) ctrain(ind(2)) ctrain(ind(3))];
        x0 = mode(x0);
        pre = [pre x0];
        if pre == ctest(j,1)
            true = true + 1;
        end
    end
    acc = true/length(index);
    subplot(3,1,1)
    bar(pre);
    title('kNN')
end
result_knn = mean(acc);

%% Naive Bayes
result_nb = [];
for i = 1:1000
    q1 = randperm(50);
    q2 = randperm(50);
    q3 = randperm(50);
    xptv = v(1:50, 3:5);
    xbach = v(51:100, 3:5);
    xjc = v(101:150, 3:5);
    xtrain = [xptv(q1(1:30), :) xbach(q2(1:30), :) xjc(q3(1:30),:)];
    xtest = [xptv(q1(31:end), :) xbach(q2(31:end), :) xjc(q3(31:end),:)];
    ctrain = [ones(30,1) 2*ones(30,1) 3*ones(30,1)];
    ctest = [ones(20,1) 2*ones(20,1) 3*ones(20,1)];
    
    nb = fitcnb(xtrain,ctrain);
    pre = nb.predict(xtest);
    true = 0;
    trials = size(xtest,1);
    for j = 1:trials
        if pre(j,1) == ctest(j,1)
            true = true + 1;
        end
    end
    acc = true/trials;
    subplot(3,1,2)
    bar(pre)
    title('Naive Bayes')
end
result_nb = mean(acc);

%% Linear Discriminant Analysis
result_lda = [];
for i = 1:1000
    q1 = randperm(50);
    q2 = randperm(50);
    q3 = randperm(50);
    xptv = v(1:50, 3:5);
    xbach = v(51:100, 3:5);
    xjc = v(101:150, 3:5);
    xtrain = [xptv(q1(1:30), :) xbach(q2(1:30), :) xjc(q3(1:30),:)];
    xtest = [xptv(q1(31:end), :) xbach(q2(31:end), :) xjc(q3(31:end),:)];
    ctrain = [ones(30,1) 2*ones(30,1) 3*ones(30,1)];
    ctest = [ones(20,1) 2*ones(20,1) 3*ones(20,1)];
    
    pre = classify(xtest,xtrain,ctrain);
    true = 0;
    trials = size(xtest,1);
    for j = 1:trials
        if pre(j,1) == ctest(j,1)
            true = true + 1;
        end
    end
    acc = true/trials;
    subplot(3,1,3)
    bar(pre)
    title('LDA')
end
result_lda = mean(acc);