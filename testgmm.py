import numpy as np
#import kdtree
import gmm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
import sys
import argparse

verbose=False
do_gmm=False
do_xgmm=True
do_dump=True
trials=1

np.random.seed(918237491)#take the mystery out

#command line arguments
parser = argparse.ArgumentParser(description="Perform Gaussian Mixture Model clustering analysis on data")
parser.add_argument('--data',help="File with the source data")
parser.add_argument('-k',help="Specify number of components [for test data] (def 5)",default="5",type=float)
parser.add_argument('--skip',help="Specify text list of columns to skip (comma separated, count from 1")
parser.add_argument('--down',help="Integer factor by which to downsample the data.")
args=parser.parse_args()

mdim=2

#Npts=int(sys.argv[1])
Npts=12500
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
        sigma=0.5/math.sqrt(kcent*math.pi/ffill)
        print "making data with",kcent,"clusters"
        print "sigma=",sigma
        datacenters=np.random.rand(kcent,mdim).tolist()
        points=[]
        for c in datacenters:
            for i in range(int(Npts/kcent)):
                p=np.random.normal(c,sigma)
                while(not ( (p>0).all() and (p<1).all())):
                    p=np.random.normal(c,sigma)
                points.append(p.tolist())
        Npts=len(points)
        print np.array(points)
        print "generated Npts=",len(points)
else:
    outname=args.data.replace(".dat","_cluster.pdf")
    with open(args.data) as f:
        points=np.loadtxt((x.replace(b':',b' ') for x in f))
    print "Raw data line:",points[0]
    parnames="eval,lpost,llike,accrat,prop,dist,phi,inc,lam,beta,psi,one".split(",")

    if(not args.skip==None):
        skipcols=[int(x)-1 for x in args.skip.split(",")]
        print "Skipping cols:",skipcols
        cols=range(points.shape[1])
        cols=list(set(cols)-set(skipcols))
        print "Keeping cols:",cols
        points=points[:,cols]
        parnames=[parnames[i] for i in cols]
        print "Kept data line:",points[0]
    if(not args.down==None):
        points=points[::int(args.down)]
    Npts,mdim=points.shape
    #next we rescale the points to a unit hypercube
    mins=np.amin(points,axis=0)
    maxs=np.amax(points,axis=0)
    eps=0.001
    scales=(maxs-mins)*(1+2*eps)
    print "scales=",scales
    points=(points-mins)/scales+eps
    print "Scaled data line:",points[0]
    points=points.tolist()
    
    
com=np.zeros(mdim)
for p in points:com+=p
print "com=",com
com=(np.array(com)/len(points)).tolist()
print "com=",com

print "Npts,dim,k=",Npts,mdim,kcent

niter=0
time0=1
best_model=None
best_lpost=None
best_BICevid=None
pp = PdfPages(outname)
    
for i in range(trials):
    if(do_gmm):
        start=time.time()
        niter=0
        #model=gmm.compute_gmm(points,2,0,pp)
        model=gmm.compute_gmm(points,2,mdim-1,pp)
        #model=gmm.compute_gmm(points,2,0,"screen")
        dtime=time.time()-start
        print "weights/centers:",zip(model.phi,model.mu)
        print "time=",dtime,"time/time0=",dtime/time0
        print "----------"
        time0=dtime
        lpost=model.lpost
        if(best_model==None or lpost>best_lpost):
            best_lpost=lpost
            best_model=model
            
    if(do_xgmm):
        start=time.time()
        niter=0
        #model=gmm.compute_xgmm(points,0,pp)
        model=gmm.compute_xgmm(points,mdim-1)
        dtime=time.time()-start
        print "Found",model.k," components"
        print "weights/centers:",zip(model.phi,model.mu)
        print "time=",dtime,"time/time0=",dtime/time0
        print "----------"
        time0=dtime
        BICevid=model.BICevid
        if(best_model==None or BICevid>best_BICevid):
            best_BICevid=BICevid
            best_model=model

print "sigmas="
for s in best_model.sigma:
    print np.matrix(s)
    
print "inverse sigmas="
for s in best_model.sigmainv:
    print np.matrix(s)
    
if(do_dump ):
    #pp = PdfPages(outname)
    print "best model lpost=",best_model.lpost
    clusters=best_model.draw_clusters(points)
    for ix in range(mdim-1):
        for iy in range(ix+1,mdim):
            best_model.plot(clusters,ix,iy,parnames,pp)
pp.close()
