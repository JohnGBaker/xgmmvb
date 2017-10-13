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
useGlobalBIC=True

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

    def update_cluster_centers(self,centers,icents,child_centers=None,child_icents=None):
        #passed in a set of centers:
        #and a set of possibly relevant center indices icents
        #updates the centers to center-assigned centriods
        #results are returned in arrays com_x_n and n_for_cent
        #with com[i]*n_for_cent[i]=com_x_n[i] corresonding to  center[i]
        #
        #If child_centers is passed, then the centers are not updated,
        #and the child centers (a list two associated with each parent center)
        #are updated instead
        
        #if len(icents)>1, we first look into trimming down icents
        global tcount
        if(verbose_update):print "node BBox=",self.bbmin,self.bbmax
        if(verbose_update):print "icents=",icents
        if(verbose_update):print "centers=",centers
        if(verbose_update):print "child_centers=",child_centers
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
        n_for_cent=np.zeros(len(centers),dtype=np.int)
        com_x_n=np.zeros_like(centers)
        dist2sum=np.zeros(len(centers))
        if(not child_centers==None):
            #In this case we will return the update for the children
            #with two children per (parent) center
            #dist2sum will hold squared-distance sum for each child
            n_for_cent=np.zeros((len(centers),2),dtype=np.int)
            com_x_n=np.zeros((len(centers),2,mdim),dtype=np.float)
            dist2sum=np.zeros((len(centers),2),dtype=np.float)
        if(len(icents)==1 and child_centers==None):
            #updating (parent) centers
            #if just one center then we are done already
            #print "trivial: n,com_x_n=",self.NPtotal,self.com_x_n
            tcount+=self.NPtotal*len(centers)
            com_x_n[icents[0]]=self.com_x_n
            n_for_cent[icents[0]]=self.NPtotal
            dist2sum[icents[0]]=self.var_x_n+self.NPtotal*dist2(self.com_x_n/self.NPtotal,centers[icents[0]])
        elif(self.leaf):
            if(child_centers==None):
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
                    ic=-1
                    for i in icents:
                        d2=dist2(centers[i],p)
                        if(d2<=mind2):
                            mind2=d2
                            ic=i
                    #print "p",p," -> ic",ic
                    #if(ic<0):print "icents=",icents
                    com_x_n[ic]+=p
                    n_for_cent[ic]+=1
                    dist2sum[ic]+=mind2
                    #print " n_for_cent=",n_for_cent
            else: #leaf node, updating just child centers
                #to be relevant for child computation center must
                #be in icents and child_icents
                if(child_icents==None):
                    iactive=icents
                else:
                    iactive=list(set(icents).intersection(child_icents))
                if(len(iactive)>0): #there is something in here of relevance
                    if(len(icents)>1):
                        #More than one parent candidate requires to
                        #individually assign parents by brute force
                        ipars=[]
                        for p in self.points:
                            mind2=float('inf')
                            for i in icents:
                                d2=dist2(centers[i],p)
                                if(d2<mind2):
                                    mind2=d2
                                    ic=i
                            ipars.append(ic)
                    else: #just one parent
                        ipars=[icents[0]]*self.NPtotal
                    for ipt in range(len(self.points)):
                        p=self.points[ipt]
                        ipar=ipars[ipt]
                        if(ipar in iactive):
                            mind2=float('inf')
                            for i in range(2):
                                d2=dist2(child_centers[ipar][i],p)
                                if(d2<mind2):
                                    mind2=d2
                                    ic=i
                            #print "comxn,ipar,ic=",com_x_n,ipar,ic," --> ",com_x_n[ipar,ic]
                            com_x_n[ipar,ic]+=p
                            n_for_cent[ipar,ic]+=1
                            dist2sum[ipar,ic]+=mind2

        else: #not a leaf node, continue down
            com_x_n_left,n_for_cent_left,dist2sum_left=self.left.update_cluster_centers(centers,icents,child_centers)
            com_x_n_right,n_for_cent_right,dist2sum_right=self.right.update_cluster_centers(centers,icents,child_centers)
            com_x_n=com_x_n_left+com_x_n_right
            n_for_cent=n_for_cent_left+n_for_cent_right
            dist2sum=dist2sum_left+dist2sum_right
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
        
        return com_x_n,n_for_cent,dist2sum

    def improve_structure(self,parent_centers,sigmas):
        #first initialize the child_centers using the distribution info
        kp=len(parent_centers)
        children=[]
        chsigmas=np.zeros((kp,2))
        chns=np.zeros((kp,2),dtype=np.int)
        for i in range(kp):
            sigma=sigmas[i]
            c=parent_centers[i]
            off=np.random.normal(c,sigma)
            child0=[]
            child1=[]
            for j in range(len(c)):
                child0+=[c[j]-off[j]]
                child1+=[c[j]+off[j]]
            children+=[[child0,child1]]
        #initialize icents
        chicents=range(kp)
        #loop for child-center kmeans:
        while(len(chicents)>0):
            #print "chicents=",chicents
            icents=range(kp)
            #  run update_cluster_centers(self,centers,icents,child_centers):
            chcom_xn,chn,chdist2sum=self.update_cluster_centers(parent_centers,icents,children,chicents)
            #print "chn=",chn
            #  test whether child-kmeans have converged
            dones=[]
            for i in range(len(chicents)):
                ic=chicents[i]
                olds=[x for x  in children[ic]]
                children[ic][0]=chcom_xn[ic,0,:]/chn[ic,0]
                children[ic][1]=chcom_xn[ic,1,:]/chn[ic,1]
                #print "olds[0]=",olds[0]
                #print "chil[0]=",children[ic][0]
                #print "olds[1]=",olds[1]
                #print "chil[1]=",children[ic][1]
                if((np.array(olds)==np.array(children[ic])).all()):
                    dones.append(ic)
            #save results for those that are done
            for i in dones:
                chsigmas[i,0]=math.sqrt(chdist2sum[i,0]/(chn[i,0]-1))
                chsigmas[i,1]=math.sqrt(chdist2sum[i,1]/(chn[i,1]-1))
                chns[i]=chn[i]
            #print "  dones",dones
            #print "  icents=",icents
            #print "  dist2sum=",chdist2sum
            #  update icents to include only those which have not converged
            chicents=[ic for ic in chicents if not ic in dones]
        #print "children=",children
        #print "  sigmas=",chsigmas
        #print "  ns=",chns
            
        #run BIC test
        BICtest=test_BICs(sigmas,chsigmas,chns)
        #print "BIC test results:",BICtest
        newcenters=[]
        for i in range(kp):
            #print "i->",BICtest[i]
            if(BICtest[i]):#replace parent with children
                #print "attaching:",children[i]
                newcenters+=children[i]
            else:
                #print "attaching:",parent_centers[i]
                newcenters.append(parent_centers[i])
        #print "new centers",newcenters
        #(possibly repeat the above multiple times?)
        #return improved list of centers...
        return newcenters
        
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
        com_x_n,n_for_cent,dist2sum=tree.update_cluster_centers(centers,range(kcent))
        olds=[x for x in centers]
        for i in range(kcent):
            if n_for_cent[i]==0:
                com_x_n[i]=tree.draw()
                n_for_cent[i]=1;
                print "Fixing empty cluster"
        centers=[com_x_n[i,:]/n_for_cent[i] for i in range(kcent)]
        #if(verbose):print "counts/centers:",zip(n_for_cent,centers)
        if((np.array(olds)==np.array(centers)).all()):break
    sigmas=[math.sqrt(dist2sum[j]/(n_for_cent[j]-1)) for j in range(len(n_for_cent))]
    #print "returning:", centers,n_for_cent,distortion,sigmas
    return centers,n_for_cent,dist2sum,sigmas

def get_k_centers(tree,kcent,points=None):
    N_no_improve=0
    best_dist=float('inf')
    best_cs=[]
    best_ns=[]
    best_sigmas=[]
    count=0
    while(N_no_improve<6):
        count+=1
        #centers,ns,distortions=compute_k_centers(tree,kcent,points)
        centers,ns,dist2sum,sigmas=compute_k_centers(tree,kcent,points)
        distortion=sum(dist2sum)
        print "trial",count," dist=",distortion,"  (best=",best_dist,")"
        #print "    counts/centers:",zip(ns,centers)
        print
        if(distortion<best_dist):
            best_dist=distortion
            best_cs=centers
            best_ns=ns
            best_sigmas=sigmas
            #best_sigmas=[math.sqrt(dist2sum[j]/(ns[j]-1)) for j in range(len(ns))] 
            N_no_improve=0
        else: N_no_improve+=1
    return centers,ns,distortion,best_sigmas
    
def get_x_centers(tree,points=None):
    kcent=2
    centers=None
    while(True):
        #print "*******\n  k =",kcent
        newcenters,ns,dist2sum,sigmas=compute_k_centers(tree,kcent,points,centers)
        distortion=sum(dist2sum)
        #print "ns=",ns
        #print "centers=",centers
        newcenters=tree.improve_structure(newcenters,sigmas)
        if((not centers==None) and (len(newcenters)==len(centers)) and (np.array(newcenters)==np.array(centers)).all()):break
        centers=newcenters
        kcent=len(centers)
        print "k=",kcent," dist=",distortion
    return centers,ns,distortion,sigmas

def test_BICs(sigmas_p,sigmas_c,ns_c):
    #In Pelleg-Moore, the BIC computation is a bit sketchy
    #This is clarified somewhat in the analysis here:
    # https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf
    #which provides a factor mdim correction in one term, but also gets
    #a sign wrong on one term...
    #Leaving out model independent terms, I get:
    #
    # BIC = sum(Rn*ln(Rn), n=1..K) + M*K/2 - R*M/2*ln(Q) - (M+1)*K/2*ln(R)
    #
    #where Q = (R-1)*sigma^2, R is the number of points, and M is dim.
    #        = sum( Qn, n=1..K)
    #with
    #      Qn = (Rn-1)*sigma_n^2  = dist2sum[n]
    #
    #Now, for node n, compare the "p" parent and a "c" child model
    #with the cluster n  split into two clusters np and nm. 
    #
    # BICc - BICp = Rnp*ln(Rnp) + Rnm*ln(Rnm) - Rn*ln(Rn) + M/2
    #               - R*M/2*ln(Q-Qn+Qnp+Qnm) + R*M/2*ln(Q)
    #               - (M+1)/2*ln(R)
    #             = Rnp*ln(Rnp/Rn) + Rnm*ln(Rnm/Rn) + M/2
    #               - R*M/2*ln(1+(Qnp+Qnm-Qn)/Q)
    #               - (M+1)/2*ln(R)
    #To get something like what is implemented in the literature, the BIC is computed only localy with Rn===R, Qn===Q
    #             = Rnp*ln(Rnp/Rn) + Rnm*ln(Rnm/Rn) + M/2
    #               - Rn*M/2*ln((Qnp+Qnm)/Qn)
    #               - (M+1)/2*ln(Rn)
    #We note that the difference between these is:
    #    local-global = M/2*(Rn*ln((Qnp+Qnm)/Qn)-R*ln(1+(Qnp+Qnm-Qn)/Q)) + (M+1)/2*ln(Rn/R)
    #                 = ...?
    test_results=[]
    Nmin=10
    K=len(ns_c)
    Q=0
    Qs=[]
    for i in range(K):
        dn=ns_c[i,0]+ns_c[i,1]
        dQ=sigmas_p[i]**2*(dn-1)
        Q+=dQ
        Qs.append(dQ)
    Ntot=sum(sum(ns_c))
    lnNtot=math.log(Ntot)
    for i in range(K):
        #parent model BIC:
        N=ns_c[i,0]+ns_c[i,1]
        Vn=Qs[i]*(N-1)
        Qn=Qs[i]
        N1=ns_c[i,0]
        N2=ns_c[i,1]
        if(N1<Nmin or N2<Nmin or Qn==0):
            #print "dBIC   ----"
            test_results.append(False)
        else:
            V1=sigmas_c[i,0]**2
            V2=sigmas_c[i,1]**2
            Q1=(N1-1)*sigmas_c[i,0]**2
            Q2=(N2-1)*sigmas_c[i,1]**2
            #print "N,Nn,N1,N2:",Ntot,N,N1,N2
            #print "Q,Qn,Q1,Q2:",Q,Qn,Q1,Q2," --> ",(Q1+Q2-Qn)/Qn,(Q1+Q2)/Qn,math.log((Q1+Q2)/Qn)
            #print N1*math.log(N1*1.0/N)
            #print  math.log(N2*1.0/N)
            #print  Ntot*mdim/2*math.log(1+(Q1+Q2-Qn)/Q)
            #print (mdim+1)/2*lnNtot
            if(useGlobalBIC):
                #global version
                dBIC = N1*math.log(N1*1.0/N) + N2*math.log(N2*1.0/N) - mdim/2.0 - mdim/2.0*Ntot*math.log(1.0+(Q1+Q2-Qn)/Q) - (mdim+1)/2.0*math.log(Ntot)
                #print "dN,Dv,dp=", N1*math.log(N1*1.0/N) + N2*math.log(N2*1.0/N), - mdim/2.0 - mdim/2.0*Ntot*math.log(1.0+(Q1+Q2-Qn)/Qn), - (mdim+1)/2.0*math.log(Ntot)
            else:
                #local version
                dBIC = N1*math.log(N1*1.0/N) + N2*math.log(N2*1.0/N) - mdim/2.0 - mdim/2.0*N*math.log((Q1+Q2)/Qn) - (mdim+1)/2.0*math.log(N)
                #print "dN,Dv,dp=", N1*math.log(N1*1.0/N) + N2*math.log(N2*1.0/N), - mdim/2.0 - mdim/2.0*N*math.log((Q1+Q2)/Qn), - (mdim+1)/2.0*math.log(N)
            #print "dBIC ",dBIC
            test_results.append(dBIC>0)
    return test_results

