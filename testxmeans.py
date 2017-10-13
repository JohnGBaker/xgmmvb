import numpy as np
#import kdtree
import xmeans as kdtree
import matplotlib.pyplot as plt
import math
import time
import sys
import argparse
from matplotlib.backends.backend_pdf import PdfPages

verbose=False
test=False
do_kmeans=False
trials=4

np.random.seed(918237491)#take the mystery out

#command line arguments
parser = argparse.ArgumentParser(description="Perform clustering analysis on data")
parser.add_argument('--data',help="File with the source data")
parser.add_argument('-k',help="Specify number of centers [for test data] (def 5)",default="5",type=float)
parser.add_argument('--skip',help="Specify text list of columns to skip (comma separated, count from 1")
parser.add_argument('--down',help="Integer factor by which to downsample the data.")
args=parser.parse_args()

#classic kmeans
def classic_update_cluster_centers(points,centers):
    com_x_n=np.zeros_like(centers)
    n_for_cent=np.zeros(len(centers))
    distortion=0
    for p in points:
        mind2=float('inf')
        for i in range(len(centers)):
            d2=kdtree.dist2(centers[i],p)
            if(d2<mind2):
                mind2=d2
                ic=i
        com_x_n[ic]+=np.array(p)
        n_for_cent[ic]+=1
        distortion+=mind2
    return com_x_n, n_for_cent,distortion

def classic_split_clusters(points,centers):
    clusters=[ [] for i in  range(len(centers))]
    for p in points:
        mind2=float('inf')
        for i in range(len(centers)):
            d2=kdtree.dist2(centers[i],p)
            if(d2<mind2):
                mind2=d2
                ic=i
        clusters[ic].append(p)
    return clusters

def plot_state(points,centers):
    p=np.array(points)
    c=np.array(centers)
    plt.scatter(p[:,0],p[:,1],c="r",marker="x")
    plt.scatter(c[:,0],c[:,1],c="b",marker="o")
    plt.show()
    
#topxkcdcolors="purple,green,blue,pink,brown,red,light blue,teal,orange,light green,magenta,yellow,sky blue,grey,lime green,light purple,violet,dark green,turquoise,lavender,dark blue,tan,cyan,aqua,forest green,mauve,dark purple,bright green,maroon,olive,salmon,beige,royal blue,navy blue,lilac,black,hot pink,light brown,pale green,peach,olive green,dark pink,periwinkle,sea green,lime,indigo,mustard,light pink".split(',')
fakexkcdcolors="purple,green,blue,pink,brown,red,lightblue,teal,orange,lightgreen,magenta,yellow,skyblue,grey,limegreen,violet,darkgreen,turquoise,darkblue,tan,cyan,forestgreen,maroon,olive,salmon,royalblue,navy,black,hotpink,palegreen,olive,seagreen,lime,indigo,lightpink".split(',')
if(False):
    for i in range(len(fakexkcdcolors)):
        col=fakexkcdcolors[i]
        xs=np.arange(10)
        ys=np.zeros(10)+i
        #print "col=",col
        plt.scatter(xs,ys,color=col,marker="x")
    plt.show()

def plot_clusters(points,centers,ix=0,iy=1,parnames=None,backend=None):
    clusters=classic_split_clusters(points,centers)
    for i in range(len(centers)):
        #print "cluster",i,"has",len(clusters[i]),"points"
        p=np.array(clusters[i])
        c=np.array(centers[i])
        col=fakexkcdcolors[i%len(fakexkcdcolors)]
        #print "color='"+col+"'"
        plt.scatter(p[::4,ix],p[::4,iy],color=col,marker=".")
        plt.scatter([c[ix]],[c[iy]],color=col,marker="*")
        if(parnames==None):
            parnames=["p"+str(i) for i in range(len(points[0]))]
        plt.xlabel(parnames[ix])
        plt.ylabel(parnames[iy])
    if(backend==None):
        plt.show()
    else:
        backend.savefig()
        plt.clf()
mdim=3
kdtree.mdim=mdim

#Npts=int(sys.argv[1])
Npts=6500
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
    kdtree.mdim=mdim
    #next we rescale the points to a unit hypercube
    mins=np.amin(points,axis=0)
    maxs=np.amax(points,axis=0)
    eps=0.001
    scales=(maxs-mins)*(1+2*eps)
    print "scales=",scales
    points=(points-mins)/scales+eps
    print "Scaled data line:",points[0]
    points=points.tolist()
    
    
var=0
com=np.zeros(mdim)
for p in points:com+=p
print "com=",com
com=(np.array(com)/len(points)).tolist()
print "com=",com
for p in points:
    var+=kdtree.dist2(p,com)

print "Npts,dim,k=",Npts,mdim,kcent
print "var=",var
centers=kdtree.draw_k_centers(points,kcent)
initcenters=centers[:]          
print "initcenters:",initcenters
if(verbose):plot_state(points,centers)
if(verbose):plot_clusters(points,centers)

start=time.time()
niter=0
time0=1
if(test and False):
    for count in range(250):
        niter=count+1
        com_x_n,n_for_cent,distortion=classic_update_cluster_centers(points,centers)
        olds=centers
        centers=[com_x_n[i,:]/n_for_cent[i] for i in range(kcent)]
        if(verbose):print "counts/centers:",zip(n_for_cent,centers)
        if(verbose):print "distortion=",distortion
        if((np.array(olds)==np.array(centers)).all()):break
    
    #plot_state(points,centers)
    print "niter, dcount, tcount=",niter,kdtree.dcount,kdtree.tcount
    print "limit0=",Npts*len(centers)*niter
    time0=time.time()-start
    print "time=",time0
    print "counts/centers:",zip(n_for_cent,centers)
    if(verbose):plot_clusters(points,centers)

#now do with kdtree structure
mincorner=np.zeros(mdim)
maxcorner=np.ones(mdim)

for i in range(trials):
    kdtree.ncount=0
    kdtree.dcount=0
    kdtree.tcount=0
    kdtree.NPmax=128*(4)**(i-trials/2)
    print "NPmax=",kdtree.NPmax
    tree=kdtree.node(mincorner,maxcorner)
    centers=initcenters[:]
    #print "centers:",centers
    
    start=time.time()
    tree.insert(points)
    print "var, treevar=",var,tree.var_x_n
    #tree.print_nodes()
    print "inittime=",time.time()-start
    if(do_kmeans):
        start=time.time()
        niter=0
        if(test):
            for count in range(250):
                niter=count+1
                #print "-------------"
                com_x_n,ns,distortion=tree.update_cluster_centers(centers,range(kcent))
                #tree.print_nodes()
                olds=centers
                centers=[com_x_n[i,:]/ns[i] for i in range(kcent)]
                if(verbose):print "counts/centers:",zip(ns,centers)
                if(verbose):print "distortion=",distortion
                if((np.array(olds)==np.array(centers)).all()):break
            #plot_state(points,centers)
            print "niter, dcount, tcount, ncount=",niter,kdtree.dcount,kdtree.tcount,kdtree.ncount
            print "limit=",(3*int(Npts/kdtree.NPmax+1)+Npts)*len(centers)*niter
        else:
             centers,ns,distortion,sigmas=kdtree.get_k_centers(tree,kcent,points)       
        dtime=time.time()-start
        print "time=",dtime,"time/time0=",dtime/time0
        print "counts/centers:",zip(ns,centers)
        print "----------"

    #kdtree.NPmax=64
    print "NPmax=",kdtree.NPmax
    start=time.time()
    xcenters,xns,xdistortion,xsigmas=kdtree.get_x_centers(tree,points)
    dtime=time.time()-start
    print "Found",len(xcenters),"centers"
    print "dist=",xdistortion
    print "time=",dtime,"time/time0=",dtime/time0
    
pp = PdfPages(outname)
for ix in range(mdim-1):
    for iy in range(ix+1,mdim):
        plot_clusters(points,centers,ix,iy,parnames,pp)
for ix in range(mdim-1):
    for iy in range(ix+1,mdim):
        plot_clusters(points,xcenters,ix,iy,parnames,pp)
pp.close()
