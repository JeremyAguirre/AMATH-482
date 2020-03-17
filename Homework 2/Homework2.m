%% AMATH 482: Assignment #2
clear all; close all; clc;
load handel 
v = y(1:length(y)-1)'; 
L = 9;
n = length(v);
t = (1:length(v))/Fs;
k = (2*pi/(L)) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);
vt = fft(v);

figure(1)
subplot(2,1,1)
plot(t,v);
xlabel('Time [sec]'); 
ylabel('Amplitude'); 
title('Signal of Interest, v(n)');

subplot(2,1,2)
plot(ks,abs(fftshift(vt))/max(abs(vt)))
xlabel('Frequency [\omega]'); 
ylabel('FFT(v)'); 

%p8 = audioplayer(S,Fs); 
%playblocking(p8);
print('-f1','Signal_of_Interest','-dpng')
%% Gaussian Window with Width of 0.2
a = 0.2;
tstep1 = 0.1;
vgt_spec = []; 
b1 = 0:tstep1:L;
for i=1:length(b1) 
    g = exp(-a*(t-b1(i)).^2); 
    vg = g.*v; 
    vgt = fft(vg); 
    vgt_spec = [vgt_spec; abs(fftshift(vgt))]; 
end

%% Mexican Hat Wavelet with Width of 0.2
a = 0.2;
tstep2 = 0.1;
vmt_spec = [];
b2 = 0:tstep2:L;
for j = 1:length(b2)
    m = 2/(sqrt(3*a)*(pi)^(1/4))*(1-((t - b2(j))/a).^2)...
        .* exp(-(t - b2(j)).^2 / (2*a^2));
    vm = m.*v; vgt = fft(vm);
    vmt_spec = [vmt_spec; abs(fftshift(vgt))];
end

%% Shannon Window with Width of 0.2
window_width = 0.2;
tstep3 = 0.1;
vst_spec = [];
b3 = 0:tstep3:L;

for k=1:length(b3)
    s = (abs(t - b3(k)) < window_width);
    vs = s.*v;
    vst = fft(vs);
    vst_spec = [vst_spec; abs(fftshift(vst))];
end

%% Spectrograms for Width of 0.2
figure (2)
subplot(3,1,1);
pcolor(b1,ks,vgt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Gaussian Window (Width = 0.2)')
colormap(hot)

subplot(3,1,2)
pcolor(b2,ks,vmt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Mexican Hat Wavelet (Width = 0.2)');
colormap(hot)

subplot(3,1,3);
pcolor(b3,ks,vst_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Shannon Window (Width = 0.2)');
colormap(hot)

print('-f2','Width_02','-dpng')
%% Gaussian Window with Width of 2
a = 2;
tstep1 = 0.1;
vgt_spec = []; 
b1 = 0:tstep1:L;
for i=1:length(b1) 
    g = exp(-a*(t-b1(i)).^2); 
    vg = g.*v; 
    vgt = fft(vg); 
    vgt_spec = [vgt_spec; abs(fftshift(vgt))]; 
end

%% Mexican Hat Wavelet with Width of 2
a = 2;
tstep2 = 0.1;
vmt_spec = [];
b2 = 0:tstep2:L;
for j = 1:length(b2)
    m = 2/(sqrt(3*a)*(pi)^(1/4))*(1-((t - b2(j))/a).^2)...
        .* exp(-(t - b2(j)).^2 / (2*a^2));
    vm = m.*v; vgt = fft(vm);
    vmt_spec = [vmt_spec; abs(fftshift(vgt))];
end

%% Shannon Window with Width of 2
window_width = 2;
tstep3 = 0.1;
vst_spec = [];
b3 = 0:tstep3:L;

for k=1:length(b3)
    s = (abs(t - b3(k)) < window_width);
    vs = s.*v;
    vst = fft(vs);
    vst_spec = [vst_spec; abs(fftshift(vst))];
end

%% Spectrograms for Width of 2
figure (3)
subplot(3,1,1);
pcolor(b1,ks,vgt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Gaussian Window (Width = 2)')
colormap(hot)

subplot(3,1,2)
pcolor(b2,ks,vmt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Mexican Hat Wavelet (Width = 2)');
colormap(hot)

subplot(3,1,3);
pcolor(b3,ks,vst_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Shannon Window (Width = 2)');
colormap(hot)

print('-f3','Width_2','-dpng')
%% Gaussian Window with Width of 20
a = 20;
tstep1 = 0.1;
vgt_spec = []; 
b1 = 0:tstep1:L;
for i=1:length(b1) 
    g = exp(-a*(t-b1(i)).^2); 
    vg = g.*v; 
    vgt = fft(vg); 
    vgt_spec = [vgt_spec; abs(fftshift(vgt))]; 
end

%% Mexican Hat Wavelet with Width of 20
a = 20;
tstep2 = 0.1;
vmt_spec = [];
b2 = 0:tstep2:L;
for j = 1:length(b2)
    m = 2/(sqrt(3*a)*(pi)^(1/4))*(1-((t - b2(j))/a).^2)...
        .* exp(-(t - b2(j)).^2 / (2*a^2));
    vm = m.*v; vgt = fft(vm);
    vmt_spec = [vmt_spec; abs(fftshift(vgt))];
end

%% Shannon Window with Width of 20
window_width = 20;
tstep3 = 0.1;
vst_spec = [];
b3 = 0:tstep3:L;

for k=1:length(b3)
    s = (abs(t - b3(k)) < window_width);
    vs = s.*v;
    vst = fft(vs);
    vst_spec = [vst_spec; abs(fftshift(vst))];
end

%% Spectrograms for Width of 20
figure (4)
subplot(3,1,1);
pcolor(b1,ks,vgt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Gaussian Window (Width = 20)')
colormap(hot)

subplot(3,1,2)
pcolor(b2,ks,vmt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Mexican Hat Wavelet (Width = 20)');
colormap(hot)

subplot(3,1,3);
pcolor(b3,ks,vst_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Shannon Window (Width = 20)');
colormap(hot)

print('-f4','Width_20','-dpng')
%% Gaussian Window: Oversampling
a = 2;
tstep1 = 0.05;
vgt_spec = []; 
b1 = 0:tstep1:L;
for i=1:length(b1) 
    g = exp(-a*(t-b1(i)).^2); 
    vg = g.*v; 
    vgt = fft(vg); 
    vgt_spec = [vgt_spec; abs(fftshift(vgt))]; 
end

%% Mexican Hat Wavelet: Oversampling
a = 2;
tstep2 = 0.05;
vmt_spec = [];
b2 = 0:tstep2:L;
for j = 1:length(b2)
    m = 2/(sqrt(3*a)*(pi)^(1/4))*(1-((t - b2(j))/a).^2)...
        .* exp(-(t - b2(j)).^2 / (2*a^2));
    vm = m.*v; vgt = fft(vm);
    vmt_spec = [vmt_spec; abs(fftshift(vgt))];
end

%% Shannon Window: Oversampling
window_width = 2;
tstep3 = 0.05;
vst_spec = [];
b3 = 0:tstep3:L;

for k=1:length(b3)
    s = (abs(t - b3(k)) < window_width);
    vs = s.*v;
    vst = fft(vs);
    vst_spec = [vst_spec; abs(fftshift(vst))];
end

%% Spectrograms (Oversampling)
figure (5)
subplot(3,1,1);
pcolor(b1,ks,vgt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Gaussian Window (Oversampling)')
colormap(hot)

subplot(3,1,2)
pcolor(b2,ks,vmt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Mexican Hat Wavelet (Oversampling)');
colormap(hot)

subplot(3,1,3);
pcolor(b3,ks,vst_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Shannon Window (Oversampling)');
colormap(hot)

print('-f5','Oversampling','-dpng')
%% Gaussian Window: Undersampling
a = 2;
tstep1 = 1;
vgt_spec = []; 
b1 = 0:tstep1:L;
for i=1:length(b1) 
    g = exp(-a*(t-b1(i)).^2); 
    vg = g.*v; 
    vgt = fft(vg); 
    vgt_spec = [vgt_spec; abs(fftshift(vgt))]; 
end

%% Mexican Hat Wavelet: Undersampling
a = 2;
tstep2 = 1;
vmt_spec = [];
b2 = 0:tstep2:L;
for j = 1:length(b2)
    m = 2/(sqrt(3*a)*(pi)^(1/4))*(1-((t - b2(j))/a).^2)...
        .* exp(-(t - b2(j)).^2 / (2*a^2));
    vm = m.*v; vgt = fft(vm);
    vmt_spec = [vmt_spec; abs(fftshift(vgt))];
end

%% Shannon Window: Undersampling
window_width = 2;
tstep3 = 1;
vst_spec = [];
b3 = 0:tstep3:L;

for k=1:length(b3)
    s = (abs(t - b3(k)) < window_width);
    vs = s.*v;
    vst = fft(vs);
    vst_spec = [vst_spec; abs(fftshift(vst))];
end

%% Spectrograms (Undersampling)
figure (6)
subplot(3,1,1);
pcolor(b1,ks,vgt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Gaussian Window (Undersampling)')
colormap(hot)

subplot(3,1,2)
pcolor(b2,ks,vmt_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Mexican Hat Wavelet (Undersampling)');
colormap(hot)

subplot(3,1,3);
pcolor(b3,ks,vst_spec.'), shading interp
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
title('Shannon Window (Undersampling)');
colormap(hot)

print('-f6','Undersampling','-dpng')

%% Part 2: Piano
clear all, close all, clc
[y,Fs] = audioread('music1.wav');
v = y';
L = length(v)/Fs;
n = length(v);
tr_piano = (1:length(v))/Fs;
k = (2*pi/(L)) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);
vt = fft(v);

figure(7)
subplot(2,1,1)
plot(tr_piano,v);
xlabel('Time [sec]'); 
ylabel('Amplitude'); 
title('Mary had a little lamb (piano)');

subplot(2,1,2)
plot(ks,abs(fftshift(vt))/max(abs(vt)))
xlabel('Frequency [\omega]'); 
ylabel('FFT(v)'); 
print('-f7','Piano','-dpng')

p8 = audioplayer(y,Fs); playblocking(p8);

a = 50;
tspan = 0.25;
Piano_Score = [];
vgt_spec = [];
b = 0:tspan:L;
for i=1:length(b)
    g=exp(-a*(tr_piano-b(i)).^2);
    vg=g.*v; 
    vgt=fft(vg);
    [M, I] = max(abs(vgt));
    Piano_Score = [Piano_Score; abs(k(I))/(2*pi)];
    vgt_spec=[vgt_spec; abs(fftshift(vgt)) / max(abs(vgt))];
end

figure(8)
plot(b,Piano_Score)
title('Piano: Music Score');
xlabel('Time [sec]');
ylabel('Frequency [Hz]');
print('-f8','Piano_Score','-dpng')

figure(9)
pcolor(b, ks, abs(vgt_spec).'), shading interp
ylim([1000 2500])
title('Piano Spectrogram')
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
colormap(hot)
print('-f9','Piano_Spec','-dpng')

%% Part 2: Recorder
clear all, close all, clc
[y,Fs] = audioread('music2.wav'); 
v = y';
L = length(v)/Fs;
n = length(v);
tr_recorder = (1:length(v))/Fs;
k = (2*pi/(L)) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);
vt = fft(v);

figure(10) 
subplot(2,1,1)
plot(tr_recorder,v); 
xlabel('Time [sec]'); 
ylabel('Amplitude'); 
title('Mary had a little lamb (recorder)');

subplot(2,1,2)
plot(ks,abs(fftshift(vt))/max(abs(vt)))
xlabel('Frequency [\omega]'); 
ylabel('FFT(v)'); 
print('-f10','Recorder','-dpng')

p8 = audioplayer(y,Fs); playblocking(p8);

a = 35;
tspan = 0.1;
Recorder_Score = [];
vgt_spec=[];
b=0:tspan:L;
for j=1:length(b)
    g=exp(-a*(tr_recorder-b(j)).^2);
    vg=g.*v; 
    vgt=fft(vg);
    [M, I] = max(abs(vgt));
    Recorder_Score = [Recorder_Score; abs(k(I)/(2*pi))];
    vgt_spec=[vgt_spec; abs(fftshift(vgt)) / max(abs(vgt))];
end

figure(11)
plot(b, Recorder_Score);
xlim([0 14])
title('Recorder: Music Score');
xlabel('Time [sec]');
ylabel('Frequency [Hz]');
print('-f11','Recorder_Score','-dpng')

figure(12)
pcolor(b, ks, abs(vgt_spec).'), shading interp
ylim([4000 8000])
title('Recorder Spectrogram')
xlabel('Time [sec]');
ylabel('Frequency [\omega]');
colormap(hot)
print('-f12','Recorder_Spec','-dpng')