#!/usr/bin/env python
import numpy as n
import matplotlib.pyplot as plt

def kp(code1,code2):
    l1=len(code1)
    l2=len(code2)
    
    kpcode=n.zeros(l1*l2,dtype=n.complex64)
    for i in range(l1):
        kpcode[n.arange(l2)+l2*i]=code2*code1[i]
    return(kpcode)
    
def simulate_meas(code,v,noise_std=0.1):
    l=len(v)
    noise=(n.random.randn(l)+n.random.randn(l)*1j)*noise_std
    m=n.fft.ifft(n.fft.fft(code,l)*n.fft.fft(v))+noise
    return(m)

def matched_filter(code,m):
    scale=n.sum(n.abs(code))
    l=len(m)
    v_mf = n.fft.ifft(n.fft.fft(m)*n.conj(n.fft.fft(code,l)))/scale
    return(v_mf)

def inverse_filter(code,m):
    l=len(m)
    v_if = n.fft.ifft(n.fft.fft(m)/n.fft.fft(code,l))
    return(v_if)

v=n.zeros(1024,dtype=n.complex64)
v[100]=1.0
v[510]=10.0+2.0j
v[515]=-10.0-2.0j
v[650:700]=1.0+1.0j

plt.subplot(211)
plt.plot(v.real)
plt.plot(v.imag)
plt.title("V")
plt.title("V")

barker13=n.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1],dtype=n.complex64)

m=simulate_meas(barker13,v,noise_std=0.01)
plt.subplot(212)
plt.plot(m.real)
plt.plot(m.imag)
plt.title("m")
plt.show()



v_mf = matched_filter(barker13,m)
plt.subplot(211)
plt.plot(v_mf.real)
plt.plot(v_mf.imag)
plt.title("MF")


v_if = inverse_filter(barker13,m)
plt.subplot(212)
plt.plot(v_if.real)
plt.plot(v_if.imag)
plt.title("IF")
plt.show()

b169=kp(barker13,barker13)
m2 = simulate_meas(b169,v,noise_std=1.0)
m1 = simulate_meas(barker13,v,noise_std=1.0)
v_mf2 = matched_filter(b169,m2)
v_if2 = inverse_filter(b169,m2)

v_mf1 = matched_filter(barker13,m1)
v_if1 = inverse_filter(barker13,m1)

# compare 169 lenght code with 13 length code.
plt.subplot(221)
plt.plot(v_mf1.real)
plt.plot(v_mf1.imag)
plt.title("MF Barker 13")
plt.subplot(223)
plt.plot(v_if1.real)
plt.plot(v_if1.imag)
plt.title("IF Barker 13")
plt.subplot(222)
plt.plot(v_mf2.real)
plt.plot(v_mf2.imag)
plt.title("MF Barker 169")
plt.subplot(224)
plt.plot(v_if2.real)
plt.plot(v_if2.imag)
plt.title("IF Barker 169")

plt.show()

