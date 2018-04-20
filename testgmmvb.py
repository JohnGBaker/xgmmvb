import numpy as np
import numpy.linalg
#import kdtree
import gmmvb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
import sys
import argparse
import os

verbose=False
do_gmmvb=True
do_xgmmvb=False
do_dump=True
trials=1

#np.random.seed(918237491)#take the mystery out

#command line arguments
parser = argparse.ArgumentParser(description="Perform Gaussian Mixture Model Variational Bayesian clustering analysis on data")
parser.add_argument('--data',help="File with the source data")
parser.add_argument('-k',help="Specify number of components [for test data] (def 5)",default="5",type=float)
parser.add_argument('-n',help="Specify approx number of sample points",default="500",type=float)
parser.add_argument('--skip',help="Specify text list of columns to skip (comma separated, count from 1")
parser.add_argument('--down',help="Integer factor by which to downsample the data.")
args=parser.parse_args()

mdim=2

Npts=int(args.n)
kcent=int(args.k)
parnames=None
outname="test.pdf"

#generate points
if(args.data==None):
    #make test data with some number of centers
    if(kcent==0):#uniform
        points=np.random.rand(Npts,mdim).tolist()
    else:
        #set sigma so that area of 2-sigma ranges for k centers
        #is fill fraction f of the unit area range
        #ignoring the issue of overlap with the edge
        #k*pi*twosigma^2-f*(1-(1-twosigma)^2)/4 = f
        ffill=0.5
        buf=2;
        sigma=0.5/math.sqrt(kcent*math.pi/ffill)*np.random.rand(kcent)
        print ("making data with",kcent,"clusters")
        print ("sigma=",sigma)
        datacenters=(((np.random.rand(kcent,mdim).T*(1-2*buf*sigma)+buf*sigma)).T).tolist()
        points=[]
        covs=[np.identity(mdim)*sigma[j]**2 for j in range(kcent)]
        ccov0=list(zip(datacenters,covs))
        for c,sig in zip(datacenters,sigma):
            print ("c=",c)
            print ("sig=",sig)
            for i in range(int(Npts/kcent*2*np.random.rand())):
                p=np.random.normal(c,sig)
               # while(not ( (p>0).all() and (p<1).all())):
                #    p=np.random.normal(c,sig)
                if( ( (p>0).all() and (p<1).all())):
                    points.append(p.tolist())
        Npts=len(points)
        #print (np.array(points))
        print ("generated Npts=",len(points))
else:
    outname=args.data.replace(".dat","_xgmmvb.pdf")
    with open(args.data) as f:
        points=np.loadtxt((x.replace(b':',b' ') for x in f))
    print ("Raw data line:",points[0])
    parnames="eval,lpost,llike,accrat,prop,dist,phi,inc,lam,beta,psi,one".split(",")

    if(not args.skip==None):
        skipcols=[int(x)-1 for x in args.skip.split(",")]
        print ("Skipping cols:",skipcols)
        cols=range(points.shape[1])
        cols=list(set(cols)-set(skipcols))
        print ("Keeping cols:",cols)
        points=points[:,cols]
        parnames=[parnames[i] for i in cols]
        print ("Kept data line:",points[0])
    if(not args.down==None):
        points=points[::int(args.down)]
    Npts,mdim=points.shape
    #next we rescale the points to a unit hypercube
    mins=np.amin(points,axis=0)
    maxs=np.amax(points,axis=0)
    eps=0.001
    scales=(maxs-mins)*(1+2*eps)
    print ("scales=",scales)
    points=(points-mins)/scales+eps
    print ("Scaled data line:",points[0])
    points=points.tolist()
    
    
com=np.zeros(mdim)
for p in points:com+=p
print ("com=",com)
com=(np.array(com)/len(points)).tolist()
print ("com=",com)

print ("Npts,dim,k=",Npts,mdim,kcent)

niter=0
time0=1
best_model=None
best_lpost=None
best_Fval=None
pp = PdfPages(outname)
    
for i in range(trials):
    if(do_gmmvb):
        start=time.time()
        niter=0
        #model=gmmvb.compute_gmmvb(points,2,pp)
        model=gmmvb.compute_gmmvb(points,kcent)
        #model=gmm.compute_gmmvb(points,2,"screen")
        dtime=time.time()-start
        print ("Npts=",Npts)
        print ("cov0=",ccov0[0][1])
        print ("ldc0=",np.linalg.slogdet(ccov0[0][1])[1])
        print ("weights/centers:",zip(model.Ncomp,model.rho))
        print ("time=",dtime,"time/time0=",dtime/time0)
        print ("----------")
        time0=dtime
        Fval=model.F
        if(best_model==None or Fval>best_Fval):
            best_Fval=Fval
            best_model=model
            
    if(do_xgmmvb):
        start=time.time()
        niter=0
        #model=gmmvb.compute_xgmmvb(points,pp)
        model=gmmvb.compute_xgmmvb(points)
        dtime=time.time()-start
        print ("Found",model.kappa," components")
        print ("weights/centers:",zip(model.Ncomp,model.rho))
        print ("time=",dtime,"time/time0=",dtime/time0)
        print ("----------")
        time0=dtime
        Fval=model.F
        if(best_model==None or Fval>best_Fval):
            best_Fval=Fval
            best_model=model

#print "sigmas="
#for s in best_model.sigma:
#    print np.matrix(s)
    
#print "inverse sigmas="
#for s in best_model.sigmainv:
#    print np.matrix(s)
    
if(do_dump ):
    #pp = PdfPages(outname)
    print ("best model F=",best_model.F)
    clusters=best_model.draw_clusters()
    #print("ccov0=",ccov0)
    for ix in range(mdim-1):
        for iy in range(ix+1,mdim):
            best_model.plot(clusters,ix,iy,parnames,pp,truths=ccov0)
pp.close()
