import numpy as np
import sklearn as sk
import scipy as sp
import pylab

from sklearn import preprocessing
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange, signal

def plotSpectrum(y,Fs, color):
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[1:n/2] # one side frequency range (sliced the frq in half)

    Y = fft(y)/n # fft computing and normalization
    Y = Y[1:n/2]

    plot(frq,abs(Y),color) # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

def pitchDetection(y,t,title):
    FFT = abs(sp.fft(y))
    freqs = sp.fftpack.fftfreq(t.size, t[1]-t[0])

    #replace negative values
    FFT[abs(FFT) < 0.000001]= 0 # some zeros
    FFT = np.ma.masked_equal(FFT,0)
    maxPos = FFT.argmax(axis=0) #finds position in FFT with highest value = most energy
    fundamentalFreq = abs(freqs.real[maxPos]) #uses maxPos to find the corresponding fundamental freq
    print('Fundamental Frequency Sine Wave ',title,'= ', fundamentalFreq, ' Hz (Beats per Second)')


def pitchDetectionData(y,t,title):
    yfft = fft(y)

    FFT = abs(sp.fft(y))
    freqs = sp.fftpack.fftfreq(len(t), 1)

    #replace negative values
    FFT[abs(FFT) < 0.000001]= 0 # some zeros
    FFT = np.ma.masked_equal(FFT,0)

    maxPos = FFT.argmax(axis=0) #finds position in FFT with highest value = most energy
    fundamentalFreq = abs(freqs.real[maxPos]) #uses maxPos to find the corresponding fundamental freq
    print('Fundamental Frequency ',title,'= ', fundamentalFreq*60, ' Breaths per Minute (Avg Adult is 12-18)')

###START-PROOF#####################################################################
Fs = 100.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = arange(0,1,Ts) # time vector

ff = 30;   # frequency of the test signal
y = sin(2*pi*ff*t)
y = y + .6*sin(4*pi*ff*t) + .4*sin(8*pi*ff*t) + .2*sin(12*pi*ff*t) #summing sines

win = signal.hann(6) #LP Filter using hanning window
yFilt = signal.convolve(y, win, mode='same') / sum(win)

#compute FFT
yfft = fft(y)
FFT = abs(sp.fft(y))
freqs = sp.fftpack.fftfreq(t.size, t[1]-t[0])
#replace negative values
FFT[abs(FFT) < 0.000001] = 0 # some zeros
FFT = np.ma.masked_equal(FFT,0)


pitchDetection(y,t,'Normal')
pitchDetection(yFilt,t,'Filtered')

pylab.subplot(212)
pylab.plot(abs(freqs.real), sp.log10(FFT), 'x')

subplot(2,1,1)
plot(t,y, 'g')
plot(t,yFilt, 'r')
title('Complex Waveform - 30 Hz')
xlabel('Time')
ylabel('Amplitude')
subplot(2,1,2)
plotSpectrum(y,Fs, 'g')
plotSpectrum(yFilt,Fs, 'r')
show()

###START-DATA##################################################################
data = np.genfromtxt(('_DataRaw/Breathing.csv'), delimiter=',', skip_header=1)

scaler = preprocessing.MinMaxScaler([-1,1]) #fits better with pitch detection

mnL = 750
mxL = 950
mnL = 1000
mxL = 1400
mnL = 1900
mxL = 2300
mnL = 1900
mxL = 2000
mnL = 50
mxL = 250
time = data[mnL:mxL,18]
#accelX = scaler.fit_transform(data[mnL:mxL, 0])
#gyroY = scaler.fit_transform(data[mnL:mxL, 10])
accelX = data[mnL:mxL, 0]
gyroY = data[mnL:mxL, 10]

win = signal.hann(20)
accelXFilt = signal.convolve(accelX, win, mode='same') / sum(win)
gyroYFilt = signal.convolve(gyroY, win, mode='same') / sum(win)

time = np.array(time).reshape(len(time), 1)
accelX = np.array(accelX).reshape(len(accelX), 1)
gyroY = np.array(gyroY).reshape(len(gyroY), 1)
accelXFilt = np.array(accelXFilt).reshape(len(accelXFilt), 1)
gyroYFilt = np.array(gyroYFilt).reshape(len(gyroYFilt), 1)

pitchDetectionData(gyroY,time,'Gyroscope Y')
pitchDetectionData(accelX,time,'Acceleration X')
pitchDetectionData(gyroYFilt,time,'Gyroscope Y - Filtered ')
pitchDetectionData(accelXFilt,time,'Acceleration X - Filtered ')

title('Acceleration X ')
ylabel('Filtered (blue) & Normal (green)')
xlabel('Time')
plot(time, accelX, 'y')
plot(time, accelXFilt, 'b')
show()

title('Gyroscope Y')
plot(time, gyroY, 'y')
plot(time, gyroYFilt, 'b')
show()
