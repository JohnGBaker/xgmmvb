#This file implements a Gaussian Mixture Model via Expectation-Maximation for clustering
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import copy
import cProfile as profile
import time

lndetsigmamax=None
epsTiny=1e-30
displayCounter=0
displayEvery=4
#class for a Gaussian mixture model
#Basic variables include:
#  k      number of model components
#  dim    space dimension
#  x      point data
#         [alternatively might be better to have a light-weight version where points are not copied in]
#  w      pointwise component weight s
#  phi    total component weights
#  mu     component centroids
#  sigma  component covariances

class gmm:
    def __init__(self,muinit,kappa=0,sigmainit=None,phiinit=None):
        #muinit is the initial guess for the centroids, as numpy array
        #kappa is: prior(detsigma)=detsigma**(-kappa/2)
        #So-called Jeffreys independent prior amounts to kappa=dim-1
        self.kappa=kappa
        self.k=muinit.shape[0]
        self.dim=muinit.shape[1]
        print "Constructed Gaussian mixture with ",self.k," components over space of dimension ",self.dim, "kappa=",kappa 
        #initalize as equal-weighted unit-Normals
        if phiinit is None:
            self.phi=np.zeros(self.k)+1.0/self.k
        else:
            self.phi=np.copy(phiinit)
        self.sigma=np.zeros((self.k,self.dim,self.dim))
        self.sigmainv=np.zeros((self.k,self.dim,self.dim))
        self.lndetsigma=np.zeros(self.k)
        for j in range(self.k):
            if(sigmainit is None):
                q=1.0
                self.sigma[j,:,:]=np.identity(self.dim)/q
                self.sigmainv[j,:,:]=np.identity(self.dim)*q
            else:
                self.sigma[j]=sigmainit[j]
                self.sigmainv[j]=np.linalg.pinv(self.sigma[j])
            s,lndet=np.linalg.slogdet(self.sigma[j])
            self.lndetsigma[j]=lndet
        self.mu=np.copy(muinit)
        self.EMtol=0.01 
        
    def show(self):
        print "Gaussian mixture model: ",self.k,"components"
        print "phi,mu="
        for i in range(self.k):print " ",self.phi[i],self.mu[i]
        print "logdetsigma=",self.lndetsigma

        
    def expectationStep(self,points,oldw=None,oldalpha=None,alpha0=None,activeComponents=None,sloppyWeights=False):
        #If oldw and oldalpha are provided,
        #  then use the previous results to estimate the need to update.
        #If activeComponents is provided,
        #  then update weights only for compts in this list.
        #  alpha0 is then used to compute the contrib to alpha from inactives
        #note:
        #here we assume that we are doing EM on the MAP
        #the j-component likelihood is:
        #llike[j]=sum([ -((p[i]-c.mu).T*c.invsigma*(p[i]-c.mu)+ln(2pi)+c.logdetsigma)/2.0*Z[i,mu] for i in npoints ])
        #and the component log prior is -kappa/2*c.logdetsigma
        #start = time.time()
        n=len(points)
        w = np.zeros((n,self.k))
        alpha = np.zeros(n)
        if(sloppyWeights):
            assert(oldw is not None)
            assert(oldalpha is not None)
        if(activeComponents is not None):
            assert(oldw is not None)
            assert(alpha0 is not None)
            inactiveComponents=set(range(self.k))-set(activeComponents)
            np.copyto(w,oldw)
            np.copyto(alpha,oldalpha)
            partiallyActive=True
        else:
            inactiveComponents=[]
            activeComponents=range(self.k)
            partiallyActive=False
        #Compute the weights
        nskip=0
        for i in range(n):
            #print time.time()-start
            #start=time.time()
            compute_w=True
            if(sloppyWeights):
                #if these are both provided then we try to economize effort
                #by skipping points with likely unchanging weights
                tol=self.EMtol/n
                ow=oldw[i,:]
                maxval=0
                for j in activeComponents:
                    if(ow[j]>=maxval):
                        jmax=j
                        maxval=ow[j]                        
                uncert=(1.0-maxval)/tol
                if(uncert<0.1):uncert=0.1
                #print "uncert=",uncert
                compute_w=uncert-np.random.rand()>0
                #ie compute if p = sum-of-non-max-compts/tol > 1,
                # otherwise randomly with probability p
                if(not compute_w):
                    w[i,:]=ow
                    alpha[i]=oldalpha[i]
                    nskip+=1
            if(compute_w):
                #start1=time.time()
                #print " ",start1-start
                if(partiallyActive):
                    #First just copy over inactive component weights
                    activeWeight=sum([w[i,j] for j in activeComponents])
                else:activeWeight=1.0
                #if(partiallyActive):print i,activeWeight
                #get collective weight for active components
                start2=time.time()
                #print " ",start2-start1
                if(activeWeight>epsTiny):
                    #now get relative weights for nontrivially active points 
                    for j in activeComponents:
                        #startj=time.time()
                        dp=points[i]-self.mu[j]
                        w[i,j]=self.phi[j]*math.exp(-(self.sigmainv[j,:,:].dot(dp).T.dot(dp) + self.lndetsigma[j])/2.0)+epsTiny*epsTiny
                        #print "   ",time.time()-startj
                    #now normalize
                    #start3=time.time()
                    #print " ",start3-start2
                    activealpha=sum([w[i,j] for j in activeComponents])
                    #if(partiallyActive):print i,"active weight/alpha:",activeWeight,activealpha
                    for j in activeComponents:
                        w[i,j]*=activeWeight/activealpha
                        #print "w[",i,j,"]->",w[i,j]
                    #start4=time.time()
                    #print " ",start4-start3
                    alpha[i]=activealpha
                    if(activeWeight<1.0):
                        alpha[i]+=alpha0[i]*(1-activeWeight)
                    #start5=time.time()
                    #print " ",start5-start4
            
                #else:
                    #print w[i,:]
                    #print points[i]
                    #print self.mu
                    #print self.phi
                    #if( 4 in activeComponents and partiallyActive and activeWeight>1e-3):print i,activeWeight,alpha[i]
            #check w normalization
            wsum=np.sum(w[i,:])
            if((wsum-1.0)**2>1e-10):
                print "WARNING wsum=",wsum
        #print "alpha=",alpha
        #alphaj=[sum(w[:,j]*alpha[:]) for j in range(self.k)] 
        #print "alphaj=",alphaj
        
        self.lpost=self.logpost(alpha)
        if (nskip>0):print "nskip=",nskip
        return w,alpha
    
    def maximizationStep(self,points,w,lndetsigmamax=None,activeComponents=None):
        #maximize E[lpost] over z
        #Assume prior indep of z
        #Assume llike(x|z)=sum(llike(xi|zi),i)
        #-> E[lpost] = sum(sum(llike(xi|j)E[zi=j],j),i) + prior
        #E[zi=j]=w[i,j] from Expectation step
        #
        #Aside from Z, posterior is function of mu, sigma, phi
        #(for each j)
        #Now maximize E[lpost] over {mu,sigma,phi}
        #
        if(activeComponents is None):activeComponents=range(self.k)
        for j in activeComponents:
            self.phi[j]=np.mean(w[:,j])
        #Fix normalization:
        phisum=np.sum(self.phi)
        if(False):
            self.phi*=1.0/phisum
        elif(False):
            phiact=sum(self.phi[i] for i in activeComponents)
            for j in activeComponents:
                #phiact'+(phisum-phiact)=1
                #phiact'/phiact=1+(1-phisum)/phiact
                self.phi[j]*=(1.0-phisum)/phiact+1.0
            
        n=len(points)
        for j in activeComponents:
            mu=np.zeros(self.dim)
            for i in range(n):
                mu+=w[i,j]*points[i,:]
            mu=mu/(n*self.phi[j]) 
            self.mu[j]=mu
            sigma=np.zeros((self.dim,self.dim))
            for i in range(n):
                x=points[i]-mu
                sigma+=np.outer(x,x)*w[i,j]
            #If kappa==0 the rest of this will be nearly trivial
            #but since the kappa-prior would indicate
            #that we divide the sum by n*phi-kappa
            #which, is problematic for small n*phi.
            #In the limit of convergence, this is the same as first
            #adding kappa*sigma then dividing by (n*phi)
            #Either of these may perhaps not converge if the expected number
            #of points in the component n*phi is too small.
            #The former version may yield nonsensical non positive-definite
            #results though so we take the latter.  
            #Convergence may be encouraged by setting some lndetsigmamax
            if(lndetsigmamax is not None):
                off=self.kappa*math.exp(0.5/self.dim*(self.lndetsigma-lndetsigmamax))
                #Note: With this, if sigma and n*phi are small then:
                #lndetsigma-lndetsigmamax
                #     -> self.lndetsigma-dim*0.5/dim*(self.lndetsigma-lndetsigmamax)  -lndetsigmamax
                #     -> 0.5(self.lndetsigma-lndetsigmamax)
                #so expect convergence toward lndetsigma=lndetsigmamax
                #also note that this is always *smaller* than with off=0
            else: off=0
            sigma=sigma + self.kappa*self.sigma[j]
            sigma=sigma/(n*self.phi[j]+off)
            sign,lndet=np.linalg.slogdet(sigma)
            if(sign==1):
                #here we slow the rate at which lndet sigma can change
                self.sigma[j]=sigma
                hedge=0.0
                softlndetsigma=lndet+hedge*(self.lndetsigma[j]-lndet)
                self.sigma[j]=self.sigma[j]*math.exp((softlndetsigma-lndet)/self.dim)
                self.lndetsigma[j]=softlndetsigma
                self.sigmainv[j,:,:]=np.linalg.pinv(self.sigma[j])
            else:
                 # in pathological case we do not update sigma
                print "Warning: Encountered pathological sigma for compt ",j
                print " sigma=",sigma
                print "npts=",len(points)
    def run_EM_MAP(self,x,activeComponents=None,backend=None,sloppyWeights=False,MaxSteps=800):
        lpost=-float('inf')
        #print "running EM"
        w,alpha=self.expectationStep(x)
        alpha0=[a for a in alpha]
        noskip=False
        print "active =",activeComponents
        every=1
        for count in range(MaxSteps):
            #print "n=",count,activeComponents
            #print "calling maxStep"
            self.maximizationStep(x,w,activeComponents=activeComponents)
            #print "calling expStep"
            w,alpha=self.expectationStep(x,w,alpha,alpha0,activeComponents,sloppyWeights and not noskip)
            lpostold=lpost
            lpost=self.lpost
            #print "phis=",self.phi
            #print "lndetsigma=",self.lndetsigma
            if(False and activeComponents is not None):
                for j in activeComponents: print " mu[",j,"]=",self.mu[j]
            #print count,"lpost=",lpost,"phi=",self.phi
            if(count%every==0):print count,"lpost=",lpost
            if(count/every>3):every*=2
            #print "mu1=",self.mu[1]
            if(backend is not None):
                global displayCounter, displayEvery
                if(displayCounter%displayEvery==0):
                    print "plotting ",displayCounter/displayEvery
                    clusters=self.draw_clusters(x)
                    self.plot(clusters,0,1,None,backend)
                displayCounter+=1
            if(lpost-lpostold<self.EMtol):
                if(noskip ):break
                else:noskip=True
            else:noskip=False
        self.evaluate(x)
        #print "alpha=",alpha
        print "best lpost=",lpost

    def evaluate(self,x):
        #set lpost and BICevid
        #print "Evaluating model with "+str(self.k)+" components"
        w,alpha=self.expectationStep(x)
        self.BICevid=-self.BIC(alpha)/2.0

    def update(self,x):
        #set lpost and BICevid
        #print "Evaluating model with "+str(self.k)+" components"
        w,alpha=self.expectationStep(x)
        self.maximizationStep(x,w)
        
    def improveStructure3(self,points,sloppyWeights=False,backend=None,nSamp=1500):
        #Like V2,  continues through, considering splits for each of
        #the whole (at call time) list of components
        #Other differences from V1 are:
        #    tolOverlap=0.03  Significantly overlapping components are
        #                     co-optimised during the split EMs
        #        tolRelax=10  Provides a relaxed tolerance for split EMs
        #                     though full tol used if it seems that the BIC
        #                     diff is close (less than tolRelax^2)
        #Differences from V2:
        #   nSamp=500    Target number of samples for the component being
        #                considered for split. Points with w[i,j]<1/Nsamp^2
        #                are left out of split EM.  If more than Nsamp remain
        #                these are downsampled
        splits=[False]*self.k
        replacements=[]
        w,alpha=self.expectationStep(points)
        updatelist=np.random.permutation(self.k)
        model=self
        model.evaluate(points)
        EMtol=self.EMtol
        tolRelax=10
        tolOverlap=0.03
        wCut=1.0/nSamp/nSamp
        jcount=0
        for j in updatelist:
            jcount+=1
            print "\nTrying split "+str(jcount)+"/"+str(len(updatelist))
            #We first compute the overlap of this component with others
            #Use full points w for this
            overs=[np.sum(w[:,j]*w[:,k]) for k in range(model.k)]
            overs=np.array(overs)/sum(overs)
            active=[i for i in range(len(overs)) if overs[i]>tolOverlap]
            print j,": overlaps =",overs
            print "-> active =",active
            #Next we downselect to just the relevant points
            jpoints=self.samplePoints(points,nSamp=nSamp,target=j,actives=active,w=w)
            #Work with a parallel test model (on the reduced set of points)
            testmodel=gmm(model.mu,model.kappa,model.sigma,model.phi)
            wj,alphaj=testmodel.expectationStep(jpoints)
            #  split center into two
            #  choose farthest of several from the component distribution and reflect
            x0s=np.random.multivariate_normal(model.mu[j],model.sigma[j],3)
            dists=[np.linalg.norm(x0-model.mu[j]) for x0 in x0s]
            maxdist=max(dists)
            x0=x0s[dists.index(maxdist)]
            x1=-x0+2.0*model.mu[j] #thus x0-mu + x1-mu = 0
            #newmu=np.ma.concatenate((model.mu[:j],[x0,x1],model.mu[j+1:]))
            newmu=np.concatenate((model.mu,[x1]))
            newmu[j,:]=x0;
            #form new phi
            newphi=np.concatenate((model.phi,[0.5*model.phi[j]]))
            newphi[j]*=0.5;
            #copy covariance to new cluster
            #newsigma=np.ma.concatenate((model.sigma[:j],[model.sigma[j],model.sigma[j]],model.sigma[j+1:]))
            newsigma=np.concatenate((model.sigma,[model.sigma[j]]))
            #print j,": mu=",newmu,"\n    sigma=",newsigma
            actives=active+[model.k]
            newmodel=gmm(newmu,model.kappa,newsigma,newphi)
            #tol-1 cycle
            newmodel.EMtol=EMtol*tolRelax
            newmodel.run_EM_MAP(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights)
            #testmodel.show()
            testmodel.EMtol=EMtol*tolRelax
            testmodel.run_EM_MAP(jpoints,activeComponents=active,sloppyWeights=sloppyWeights)
            #  compare new BIC to original clustering BIC [wrt all points]
            #tol-2 cycle
            if(newmodel.BICevid>testmodel.BICevid and testmodel.BICevid-newmodel.BICevid<EMtol*tolRelax**2):#This is a close call continue EM with orig EMtol
                newmodel.EMtol=EMtol
                testmodel.EMtol=EMtol
            #Run a second time for "trimming" [resets relative weights]
            newmodel.run_EM_MAP(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights)
            testmodel.run_EM_MAP(jpoints,activeComponents=active,sloppyWeights=sloppyWeights)

            #testmodel.update(points)  
            #testmodel.evaluate(points)  
            #must update phi,sigma,mu for all pts to get correct results
            newmodel.update(points)  
            newmodel.evaluate(points)
            print "actives=",actives
            print "BICs orig/new",model.BICevid,newmodel.BICevid
            print "lposts orig/new",model.lpost,newmodel.lpost
            #print "model:\n",model.show()
            #print "newmodel:\n",newmodel.show()
            if(newmodel.BICevid>model.BICevid):
                print "Found improved model:"
                #newmodel.show()
                newmodel.EMtol=EMtol
                model=newmodel
                #we accept the newmodel as model
                #we don't run full EM, but we do need to update w and we
                #(do?) update the model params
                w,alpha=model.expectationStep(points)#update w
                #must reeval model above if doing this...
                model.maximizationStep(points,w)
                model.evaluate(points)
                print "evaluated model: BIC=",model.BICevid,"lpost=",model.lpost
                if(True):
                    phisum=np.sum(self.phi)
                    print " [before] =",np.array2string(model.phi,formatter={'float_kind':lambda x: "%.5f" % x})
                    print "lndetsigma=",model.lndetsigma
                    print "model.mu =\n",np.array2string(model.mu,formatter={'float_kind':lambda x: "%.3f" % x})
                print "model.phi=",np.array2string(model.phi,formatter={'float_kind':lambda x: "%.5f" % x})
                if(backend is not None):#do it all over for display purposes
                    newmodel=gmm(newmu,model.kappa,newsigma,newphi)
                    newmodel.EMtol=EMtol*tolRelax
                    newmodel.run_EM_MAP(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)
                    if(newmodel.BICevid>model.BICevid and model.BICevid-newmodel.BICevid<EMtol*tolRelax**2):newmodel.EMtol=EMtol
                    newmodel.run_EM_MAP(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)
                
        print "returning with BIC=",model.BICevid
        return model
    
    def samplePoints(self,points,nSamp=1500,target=None,actives=None,w=None):
        #     nSamp   target # of samples for indicated compt
        #    target   if provided indicates component for which to scale sample
        #             otherwise smallest, or smallest among actives
        #   actives   if provided only include points relevant for any of
        #             the indicated components (by index)
        #         w   if provided, assumed to have precomputed weights
        if(actives is None):
            if target is None:
                actives=range(self.k)
            else:
                actives=[target]
        if(w is None):
            w,alpha=self.expectationStep(points)
        wCut=1.0/nSamp**2
        #Select all points relevant for active components
        idxs=[i for i in range(len(points)) if sum([w[i,k] for k in actives])>wCut]
        if target is None:
            #num of points relevant for smallest component
            nt=min(sum( w[i,j]>wCut for i in range(len(points))) for j in range(self.k))
        else:
            nt=sum( w[i,target]>wCut for i in range(len(points)))

        #ie want nSamp pts relev for comp j
        ntarg=int(nSamp*len(idxs)/(nt+1.0/nSamp))
        print "Selecting min(", len(idxs),",",ntarg, ") points for split."
        if len(idxs)>ntarg :
            idxs=np.random.choice(idxs,ntarg,replace=False)
        spoints=np.array([points[i] for i in idxs])
        return spoints

    def logpost(self,alpha):
        llike=0
        n=len(alpha)
        for i in range(n):
            llike+=np.log(alpha[i])
        lprior=0
        if(self.kappa != 0):
            for j in range(self.k):
                lprior-=self.kappa/2.0*self.lndetsigma[j]
        return llike - lprior

    def BIC(self,alpha):
        n=len(alpha)
        logpost=self.logpost(alpha)
        #dof are k*d mu compts,k*d*(d+1)/2 sigma compts, k wts
        dof=self.k*(self.dim+self.dim*(self.dim+1)/2+1)
        return -0.5*2*logpost+dof*math.log(n)
        #the BIC doesn't make sense except for n>>dof
        #here we additionally penalize cases were this isn't true
        #return -logpost+dof*math.log(n+dof*dof/n)
        #return -logpost+dof*math.log(n+dof)


    def draw_clusters(self,points):
        p=np.array(points)
        clusters=[ [] for i in  range(self.k)]
        w,alpha=self.expectationStep(p)
        n=len(p)
        for i in range(n):
            x=np.random.rand()
            run=0
            #print "wi=",w[i]
            for j in range(self.k):
                run+=w[i,j]
                #print "j,x,run",j,x,run
                if(x<run):
                    ic=j
                    break
            #print "ic=",ic
            clusters[ic].append(p[i])
        return clusters

    def ellipse(self,ic,ix,iy,siglev=2,N_thetas=60):
        cov=self.sigma[ic,:,:]
        dtheta=2.0*math.pi/(N_thetas-1)
        thetas=np.arange(0,(2.0*math.pi+dtheta),dtheta)
        ang=-math.pi/4.
        root=cov[ix,iy]/math.sqrt(cov[ix,ix]*cov[iy,iy])
        if(root>1):root=1
        if(root<-1):root=-1
        acoeff=math.sqrt(1-root)
        bcoeff=math.sqrt(1+root)
        xcoeff=math.sqrt(cov[iy,iy])
        ycoeff=math.sqrt(cov[ix,ix])
        #print("a2,b2",acoeff*acoeff,bcoeff*bcoeff)
        #print("a,b,ang, xcoeff,ycoeff, root=",acoeff,bcoeff,ang,xcoeff,ycoeff,root)
        #in the next line we convert the credibility limit
        #to a "sigma" limit for a 2-d normal
        #this becomes a scale-factor for the error ellipse
        #1-exp(x^2/(-2))=y
        #-2*log(1-y)=x^2
        #lev_fac = math.sqrt( -2 * math.log( 1 - siglev ) )
        lev_fac=siglev
        #print ("scales for quantile level = ",siglev," -> ",lev_fac,": (",xcoeff*lev_fac,",",ycoeff*lev_fac,")")
        elxs=[self.mu[ic,iy]+lev_fac*xcoeff*(acoeff*math.cos(th)*math.cos(ang)-bcoeff*math.sin(th)*math.sin(ang)) for th in thetas] 
        elys=[self.mu[ic,ix]+lev_fac*ycoeff*(acoeff*math.cos(th)*math.sin(ang)+bcoeff*math.sin(th)*math.cos(ang)) for th in thetas] 
        return elys,elxs
        
    def plot(self,clusters,ix=0,iy=1,parnames=None,backend=None):
        #n=sum([len(c) for c in clusters])
        #ndown=n/1000
        ndown=1
        for j in range(self.k):
            p=np.array(clusters[j])
            #print p.shape
            #print p
            if(len(p)<1):continue
            c=np.array(self.mu[j])
            col=fakexkcdcolors[j%len(fakexkcdcolors)]
            #print p.shape
            plt.scatter(p[::ndown,ix],p[::ndown,iy],color=col,marker=".")
            plt.scatter([c[ix]],[c[iy]],color=col,marker="*")
            elxs,elys=self.ellipse(j,ix,iy,2.0)
            plt.plot(elxs,elys,color=col)
            plt.xlim([0,1]) 
            plt.ylim([0,1]) 
            if(parnames is None):
                parnames=["p"+str(ip) for ip in range(self.dim)]
            plt.xlabel(parnames[ix])
            plt.ylabel(parnames[iy])
        if(backend is None or backend=="screen"):
            plt.show()
        else:
            backend.savefig()
            #print "savedfig"
            plt.clf()

fakexkcdcolors="purple,green,blue,pink,brown,red,lightblue,teal,orange,lightgreen,magenta,yellow,skyblue,grey,limegreen,violet,darkgreen,turquoise,darkblue,tan,cyan,forestgreen,maroon,olive,salmon,royalblue,navy,black,hotpink,palegreen,olive,seagreen,lime,indigo,lightpink".split(',')

def compute_gmm(points,k,kappa=0,backend=None,sloppyWeights=False):
    muinit=np.array([points[i] for i in np.random.choice(len(points),k)])
    model=gmm(muinit,kappa)
    x=np.array(points)
    #profile.runctx("model.run_EM_MAP(x,backend=backend)",globals(),locals())
    model.run_EM_MAP(x,backend=backend,sloppyWeights=sloppyWeights)
    return model

def compute_xgmm(points,kappa=0,backend=None,sloppyWeights=False):
    k=2
    nsamp=5000
    muinit=np.array([points[i] for i in np.random.choice(len(points),k)])
    model=gmm(muinit,kappa)
    x=np.array(points)
    #sx=x
    sx=np.array(model.samplePoints(x,nsamp))
    #profile.runctx("model.run_EM_MAP(x,backend=backend)",globals(),locals())
    model.run_EM_MAP(sx,backend=backend,sloppyWeights=sloppyWeights)
    oldBICevid=model.BICevid
    count=0
    while(True):
        count+=1
        print
        print "Beginning xgmm cycle",count
        t0=time.time()
        sx=np.array(model.samplePoints(x,nsamp))
        model.evaluate(sx)
        oldBICevid=model.BICevid
        oldk=model.k
        model=model.improveStructure3(x,backend=backend,nSamp=500)
        #model=model.improveStructure3(sx,backend=backend)
        t1=time.time()
        print "improve model time =",t1-t0
        print
        print "============"
        print model.show()
        #sx=model.samplePoints(x)
        model.run_EM_MAP(np.array(sx),backend=backend)
        model.evaluate(sx)
        print "BICevid/old=",model.BICevid,oldBICevid
        if(model.k==oldk):break
        print "full model EM time =",time.time()-t1
        print
        
    return model

