import math
import numpy as np

## Global variables
um = 1e-6;
cm = 1e-2;
mm = 1e-3; 
kg = 1.66053907e-27; # converts AMU to kg
AMU = 6.022e26
K = 8.9875e9;
CHARGE = 1.6e-19;
Z = 1; # ion charge number
e = 1.62176634e-19 # elementary charge
m = 171*kg; # mass (Yb 171 ion)
z0 = -11.8*um; # location of ground plane (next to M3)
z_image_M4 = 2*z0; # z position of image DC voltages from M4

## electrode geometry
lcs = 2069.6*2*um; # central slot length
wcs = 60*um; # central slot width
L1 = 70*um; # length of electrodes Q1-38
w1 = 20*um; # estimated width of electrode Q1-38
yc_odd = wcs/2+w1/2; # y-coordinate of electrode center (odd Q1-38)
yc_even = -1*yc_odd; # y-coordinate of electrode center (even Q1-38)
xc_max = 630*um; # maximum x-coordinate of electrode center
w2 = 110*um; # estimated width Q39,40
wrf = 110*um; # estimated width rf electrode
gap = 5*um; # estimated gap 

## DC electrode coordinates
electrode_center_x = np.linspace(-1*xc_max,xc_max,int(2*xc_max/L1+1)); # x-coordinate of electrode center, Q1-Q38 (odd/even) in order
xy1k = []; # corner 1 (bottom left) of electrode
xy2k = []; # corner 2 (top right) of electrode
# Q1-38 odd
for p in range (0,19): # Q1-38 odd
    xy1k.append((electrode_center_x[p]-L1/2,yc_odd-w1/2))
    xy2k.append((electrode_center_x[p]+L1/2,yc_odd+w1/2)) 
# Q39
(x1_39,y1_39) = (-1*lcs/2,wcs/2+w1+gap+wrf+2*gap); # updated
(x2_39,y2_39) = (lcs/2,wcs/2+w1+gap+wrf+2*gap+w2)
xy1k.append((x1_39,y1_39))
xy2k.append((x2_39,y2_39)) 
# Q1-38 even
for q in range (0,19): # Q1-38 even
    xy1k.append((electrode_center_x[q]-L1/2,yc_even-w1/2)) 
    xy2k.append((electrode_center_x[q]+L1/2,yc_even+w1/2)) 
# Q40
(x1_40,y1_40) = (-1*lcs/2,-1*(wcs/2+w1+gap+wrf+2*gap+w2))
(x2_40,y2_40) = (lcs/2,-1*(wcs/2+w1+gap+wrf+2*gap)); # updated
xy1k.append((x1_40,y1_40)) 
xy2k.append((x2_40,y2_40)) 

## RF electrode coordinates
# parameters used for tuning ion height
gap_rf = 59.4*um; 
wrf_new = 135.3*um; 
(x11,y11) = (-1*lcs/2,-1*(gap_rf/2+wrf_new)); # corner 1 (bottom left) of electrode 1
(x21,y21) = (lcs/2,-1*gap_rf/2); # corner 2 (top right) of electrode 1
(x12,y12) = (-1*lcs/2,gap_rf/2); # corner 1 (bottom left) of electrode 2
(x22,y22) = (lcs/2,gap_rf/2+wrf_new); # corner 2 (top right) of electrode 2

## RF electrode voltage
scale_vrf = 0.6651; # scaling for RF potential to tune radial trap frequencies
vrf_original = 250; # original rf voltage amplitude
vrf = scale_vrf * vrf_original; # modified rf voltage amplitude
omega = 2*math.pi*36*10**6; # rf voltage angular frequency

## DC electrode voltages (sorted Q odd then Q even)
scale_vdc = 0.7578; # scaling for RF potential to tune axial trap frequency
vk = scale_vdc * np.array([1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, 9.209430601182174, 0.39616128244237486,
    -0.12035087347568768, 0.39616128244237486, 9.209430601182174,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, -0.21367342138725715, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    9.209430601182174, 0.39616128244237486, -0.12035087347568768,
    0.39616128244237486, 9.209430601182174, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    1.4486543682787263, 1.4486543682787263, 1.4486543682787263,
    -0.21367342138725715])