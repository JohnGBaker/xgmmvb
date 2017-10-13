import numpy as np
import math
mdim=2
NPmax=25
NPmin=1
verbose_tree=False
verbose_update=False
test_results=False
dcount=0
tcount=0
ncount=0

def dist2(p1,p2):
    global dcount
    dcount+=1
    dp=np.array(p1)-np.array(p2)
    return np.dot(dp,dp)

def closest_point_in_region(p,hmin,hmax):
    #return closest point to p in hyperrectangle represented by lower and
    #upper corners hmin and hmax
    x=np.zeros(np.array(p).size)
    for i in range(mdim):
        x[i]=p[i]
        if(hmin[i]>x[i]):
            x[i]=hmin[i]
        elif(hmax[i]<=x[i]):
            x[i]=hmax[i]
    return x
        
def p1_dominates_p2_over_region(p1,p2,hmin,hmax):
    #return true if every (possible) point in region is closer to p1 than p2
    #print "testing dominance of ",p1,">",p2," in:",hmin,hmax
    pc=np.zeros(mdim)
    for i in range(mdim):
        if(p2[i]>p1[i]):
            pc[i]=hmax[i]
        else:
            pc[i]=hmin[i]
    #print "  pc=",pc
    #print "  dist(pc,p1)=",math.sqrt(dist2(p1,pc))
    #print "  dist(pc,p2)=",math.sqrt(dist2(p2,pc))
    #pc is now extreme point in h along direction from p1 to p2
    #if this point is still closer to p1, then all points in h are closer to p1
    return (dist2(p1,pc)<dist2(p2,pc))

class node:
    #kd-tree vars: sdim,sval,left,right
    #domain: bbmin,bbmax: corners of bounding box
    #points: points contained by leaf node.
    #kmeans vars: com_x_n
    
    def __init__(self,bbmin,bbmax,axis=0):
        self.iaxis=axis
        self.bbmin=bbmin.copy()
        self.bbmax=bbmax.copy()
        self.cutval=None
        self.NPtotal=0
        self.points=[]
        self.leaf=True
        self.left=None
        self.right=None
        self.next_marker=int(0)
        self.com_x_n=np.zeros(mdim)
        self.var_x_n=0
        
    def print_nodes(self,indent=""):
        if(self.leaf):
            print indent+"leaf node: bbmin=",self.bbmin," bbmax=",self.bbmax," N=",self.NPtotal
            print indent+" com=",self.com_x_n/self.NPtotal
            print indent+" var=",self.var_x_n
            print indent+" points = ("+str(self.NPtotal)+")",np.array(self.points)[0:2]
        else:
            print indent+"branch node: bbmin=",self.bbmin," bbmax=",self.bbmax," N=",self.NPtotal
            print indent+" com=",self.com_x_n/self.NPtotal
            print indent+" var=",self.var_x_n
            print indent+"left:"
            self.left.print_nodes(indent+"- ")
            print indent+"right:"
            self.right.print_nodes(indent+"- ")
            
            
    def reset_marker(self):
        if(self.leaf):
            self.next_marker=0

    def __iter__(self):
        return self

    def next(self):
        if self.leaf:
            if self.next_marker >= self.NPtotal:
                raise StopIteration
            else:
                self.next_marker+=1
                return self.points[self.next_marker]
        else: #not leaf
            if self.next_marker >=2:
                raise StopIteration
            elif(self.next_marker==0):
                #pointing left
                try:
                    return self.left.__next()
                except (StopIteration):
                    #done with left
                    self.next_marker=1
                    #continue forward
            if self.next_marker==1:
                #pointing right
                try:
                    return self.right.__next()
                except (StopIteration):
                    #done with left
                    self.next_marker=2
                    raise StopIteration
            print "node.__next__: Should not reach here. next_marker=",self.next_marker 

    def insert(self,points):
        #add a points to the kd-tree
        if(self.leaf):
            self.points=self.points+points
            self.NPtotal=len(self.points)
            if(verbose_tree):print "insert: added ",len(points)," points for total of ",self.NPtotal
            for point in points:
                for i in range(mdim):
                    if(point[i]<self.bbmin[i] or point[i]>self.bbmax[i]):
                        print "ERROR point ",point
                        print "not in bbox:",self.bbmin,self.bbmax
                self.com_x_n+=np.array(point)
            com=self.com_x_n/self.NPtotal
            self.var_x_n=0
            for point in points:
                self.var_x_n+=dist2(point,com)
                
            if(self.NPtotal>NPmax):
                if(verbose_tree):print "Need to divide"
                #divide to for new leafs
                #we control for a few possible pathologies
                #we don't want to avoid empty nodes, so we check for this
                #if one-side would be empty we narrow the bounding box
                #then try rotating the dimension, repeating
                #until satisfied, or until the box is too small
                #note that we do a geometrical bisection cut rather than a
                #median cut to avoid splitting clusters in the middle 
                bbmin=self.bbmin.copy()
                bbmax=self.bbmax.copy()
                iaxis=(self.iaxis-1)%mdim
                icount=0;
                while(True and icount<mdim*5):
                    icount+=1
                    #first rotate dim
                    iaxis=(iaxis+1)%mdim
                    #try to bisect
                    cutval=(bbmin[iaxis]+bbmax[iaxis])/2.0
                    if(verbose_tree):print "trying to divide on axis[",iaxis,"] = ",cutval
                    leftpoints=[]
                    rightpoints=[]
                    #divide set of points
                    for point in self.points:
                        if(point[iaxis]<cutval):leftpoints.append(point)
                        else:rightpoints.append(point)
                    #check left/right set sizes
                    ok=True
                    if(len(leftpoints)<NPmin):
                        ok=False
                        bbmin[iaxis]=cutval
                    elif(len(rightpoints)<NPmin):
                        ok=False
                        bbmax[iaxis]=cutval
                    if(ok):
                        #everything is ready for splitting
                        self.iaxis=iaxis
                        self.cutval=cutval
                        self.divide(leftpoints,rightpoints)
                        break
        #Above we either divided or not, now update up-flowing info    
        if(not self.leaf):
            self.NPtotal=self.left.NPtotal+self.right.NPtotal
            self.com_x_n=self.left.com_x_n+self.right.com_x_n
            self.var_x_n=self.left.var_x_n+self.right.var_x_n
            dcom2=dist2(self.left.com_x_n/self.left.NPtotal,self.right.com_x_n/self.right.NPtotal)
            self.var_x_n+=dcom2*self.left.NPtotal*self.right.NPtotal/self.NPtotal
 
            
    def divide(self,leftpoints=None,rightpoints=None):
        #divide leaf node
        if(not self.leaf):
            print "Cannot divide non-leaf node"
        child_iaxis=(self.iaxis+1)%mdim
        if(verbose_tree):print "dividing node: bbmin,bbmax=",self.bbmin,self.bbmax
        bbmin=np.array(self.bbmin)
        bbmax=np.array(self.bbmax)
        bbmax[self.iaxis]=self.cutval
        if(verbose_tree):print "   left: bbmin,bbmax=",bbmin,bbmax
        self.left=node(bbmin,bbmax,axis=child_iaxis)
        bbmax[self.iaxis]=self.bbmax[self.iaxis]
        bbmin[self.iaxis]=self.cutval
        if(verbose_tree):print "  right: bbmin,bbmax=",bbmin,bbmax
        self.right=node(bbmin,bbmax,axis=child_iaxis)
        if(leftpoints==None):
            for point in self.points:
                if(point[self.iaxis]<self.cutval):leftpoints.append(point)
                else:rightpoints.append(point)            
        if(verbose_tree):print "  inserting on left:"
        self.left.insert(leftpoints)
        if(verbose_tree):print "  inserting on right:"
        self.right.insert(rightpoints)
        self.leaf=False
        self.points=[]

    def update_cluster_centers(self,centers,icents):
        #passed in a set of centers:
        #and a set of possibly relevant center indices icents
        #updates the centers to center-assigned centriods
        #results are returned in arrays com_x_n and n_for_cent
        #with com[i]*n_for_cent[i]=com_x_n[i] corresonding to  center[i]

        #if len(icents)>1, we first look into trimming down icents
        global tcount
        if(verbose_update):print "node BBox=",self.bbmin,self.bbmax
        if(verbose_update):print "icents=",icents
        if(verbose_update):print "centers=",centers
        dcount00=dcount
        potential=len(icents)*self.NPtotal -  (len(icents)-1)*2
        if(potential>len(icents)*3 and len(icents)>1):
            dist2s=[dist2(centers[ix],closest_point_in_region(centers[ix],self.bbmin,self.bbmax)) for ix in icents]
            mindist2=float('inf')
            imin=[]
            for i in range(len(dist2s)):
                if(dist2s[i]<mindist2):
                    mindist2=dist2s[i]
                    imin=[icents[i]]
                elif(dist2s[i]==mindist2):
                    imin.append(icents[i])
            #print "imin=",imin
            #List imin contains the list of centers which are closest to the node's
            #bounding box region, there may be a tie; in particular several nodes
            #may be inside the region (dist=0).
            #If len(imin)==1, then it is possible that this center may "own" the
            #region, meaning that everywhere within the region may be closest to
            #the same single center.
            #If len(imin)>1 then obviously none of them may own the region, but they
            #may collectively dominate over other centers, meaning that one of these
            #centers is guaranteed to be closer than the dominated center.
            #To accelerate kmeans updates that center can then be excluded as a
            #candidate for this region and can thus be pruned from further
            #consideration.
            #If len(imin) is close total number of centers, then it is a waste of
            #time to investigate whether some may be excludable, but it will
            #certainly help if some centers can be excluded at leaf level.
            if(self.NPtotal>1 and ( self.leaf or len(imin)<=int(len(icents)/4) )):
                #make set iaway of non-minimum-dist centers, in icents but not imin
                dcount0=dcount
                iaway = np.setdiff1d(icents,imin)
                iexclude=[]
                for i1 in iaway:
                    c1=centers[i1]
                    for i2 in imin:
                        c2=centers[i2]
                        if(p1_dominates_p2_over_region(c2,c1,self.bbmin,self.bbmax)):
                            #print "center",i2,"dominates",i1
                            iexclude.append(i1)
                            break
                #print "iexclude=",iexclude
                if(len(iexclude)>0):
                    #realize the exclusion
                    icents=np.setdiff1d(icents,iexclude)
                #cost=(dcount-dcount0)
                #if(len(icents)>1):cost+=self.NPtotal*len(icents)
                #print "checked excludes: found ",iexclude,"dcount=",dcount-dcount0,"icents=",icents,"NPtotal=",self.NPtotal," profit=",self.NPtotal*(len(icents)+len(iexclude))-cost

        #print" overhead=",dcount-dcount00,"potential=",potential
        global ncount
        ncount+=1
        #now compute the new centriods:
        #print len(icents)
        com_x_n=np.zeros_like(centers)
        n_for_cent=np.zeros(len(centers),dtype=np.int)
        distortion=0
        
        if(len(icents)==1):#if just one center then we are done already
            #print "trivial: n,com_x_n=",self.NPtotal,self.com_x_n
            tcount+=self.NPtotal*len(centers)
            com_x_n[icents[0]]=self.com_x_n
            n_for_cent[icents[0]]=self.NPtotal
            distortion=self.var_x_n+self.NPtotal*dist2(self.com_x_n/self.NPtotal,centers[icents[0]])
        elif(self.leaf):
            #print "brute centers=",centers
            #self.print_nodes(":::")
            #print " icents=",icents
            cps=[closest_point_in_region(centers[ix],self.bbmin,self.bbmax) for ix in icents]
            #print " closest points:",np.array(cps)
            #print " dists=",[math.sqrt(dist2(centers[icents[ix]],cps[ix])) for ix in range(len(icents))]
            #compute centriods area constributions by brute effort
            tcount+=self.NPtotal*(len(centers)-len(icents))
            for p in self.points:
                mind2=float('inf')
                for i in icents:
                    d2=dist2(centers[i],p)
                    if(d2<mind2):
                        mind2=d2
                        ic=i
                #print "p",p," -> ic",ic
                com_x_n[ic]+=p
                n_for_cent[ic]+=1
                distortion+=mind2
                #print " n_for_cent=",n_for_cent
        else: #not a leaf node
            com_x_n_left,n_for_cent_left,distortion_left=self.left.update_cluster_centers(centers,icents)
            com_x_n_right,n_for_cent_right,distortion_right=self.right.update_cluster_centers(centers,icents)
            com_x_n=com_x_n_left+com_x_n_right
            n_for_cent=n_for_cent_left+n_for_cent_right
            distortion=distortion_left+distortion_right
            if(verbose_update):print " ->by centers(left) n,com_x_n=",n_for_cent_left,com_x_n_left
            if(verbose_update):print " ->by centers(right) n,com_x_n=",n_for_cent_right,com_x_n_right
        if(verbose_update):print "node: n,com_x_n=",self.NPtotal,self.com_x_n
        if(verbose_update):print "centers=",centers
        if(verbose_update):print " ->active centers:",icents
        if(verbose_update):print " ->by centers n,com_x_n=",n_for_cent,com_x_n
        if(test_results):
            #check leaf points:
            if(self.leaf):
                for p in self.points:
                    mind2=float('inf')
                    for i in range(len(centers)):
                        d2=dist2(centers[i],p)
                        if(d2<mind2):
                            mind2=d2
                            ic=i
                    if(not ic in icents):
                        print "point ",p
                        print "  closest to center ",ic," not in ",icents
                        print "for node:"
                        self.print_nodes("***")
        
        return com_x_n,n_for_cent,distortion

    def elem(self,ielem):#dereference element in tree-based index order
        if(ielem>=self.NPtotal):
            raise RangeError("Requested element-point in kdtree node exceeds number of points held by node")
        if(self.leaf):
            return self.points[ielem]
        else:
            nleft=self.left.NPtotal
            if(i<nleft):
                return self.left.elem(i)
            else:
                return self.right.elem(i-nleft)

    def draw(self):#draw a point from tree bounding box weighted by point distribution
        if(self.leaf):
            return list(np.random.uniform(self.bbmin,self.bbmax))
        else:
            x=np.random.uniform(self.bbmin[self.iaxis],self.bbmin[self.iaxis])
            if(x<self.cutval):
                return self.left.draw()
            else:
                return self.right.draw()

def draw_k_centers(points,kcent):
    centers=[]
    k=0
    while(k<kcent): #this loop is needed to choose as set of *unique* centers
        centers+=[points[i] for i in np.random.choice(len(points),kcent-k)]
        dups=[]
        for j in range(k,kcent):
            if(centers[j] in centers[:j] or centers[j] in centers[j+1:]):
                dups.append(j)
        #print "dups=",dups
        centers=[centers[j] for j in range(kcent) if j not in dups]
        k=len(centers)
        #print "k=",k
    return centers

def compute_k_centers(tree,kcent,points=None,centers=None):
    if(centers==None):
        if(points==None):
            centers=[]
            for i in range(kcent):
                centers.append(tree.draw())
                print "drew:",np.array(centers[-1])
        else:
            centers=draw_k_centers(points,kcent)
            
    for count in range(250):
        niter=count+1
        com_x_n,n_for_cent,distortion=tree.update_cluster_centers(centers,range(kcent))
        olds=centers
        centers=[com_x_n[i,:]/n_for_cent[i] for i in range(kcent)]
        #if(verbose):print "counts/centers:",zip(n_for_cent,centers)
        if((np.array(olds)==np.array(centers)).all()):break
    return centers,n_for_cent,distortion

def get_k_centers(tree,kcent,points=None):
    N_no_improve=0
    best_dist=float('inf')
    best_cs=[]
    best_ns=[]
    count=0
    while(N_no_improve<6):
        count+=1
        centers,ns,distortion=compute_k_centers(tree,kcent,points)
        print "trial",count," dist=",distortion,"  (best=",best_dist,")"
        #print "    counts/centers:",zip(ns,centers)
        print
        if(distortion<best_dist):
            best_dist=distortion
            best_cs=centers
            best_ns=ns
            N_no_improve=0
        else: N_no_improve+=1
    return centers,ns,distortion
    
