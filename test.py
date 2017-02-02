import numpy as np

totalDesk = 22810+22748+22934+23041+22629+22789+22823+22797+22725+22488
totalUnSynDesk = 9+8+8+9+6+10+9+6+9+11
totalAccount = 155358+154956+156024+156124+154938+153859+154475+154575+155275+154221
totalUnSynAccount = 8+8+8+7+6+10+8+5+9+11

print "Total Desk:",totalDesk
print "Total UnSyn Desk:",totalUnSynDesk
print "UnSyn Desk Ratio:",totalUnSynDesk/float(totalDesk)*100,"%"
print "Total Player:",totalAccount
print "Total UnSyn Player:",totalUnSynAccount
print "UnSyn Player Ratio:",totalUnSynAccount/float(totalAccount)*100,"%"


a= [1,2,3,4,5,6]
b= [6,5,4,3,2,1,8]
print zip(a,b)
print map(lambda (x,y):x*y,zip(a,b))
print reduce(lambda m,n:m+n ,map(lambda (x,y):x*y,zip(a,b)),1.0)

def f(x):
    return 1 if x>0 else 0

print range(9)

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
print sigmoid(0.000005)

print __file__