#This file implements a variational Bayesian Gaussian Mixture Model for clustering
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import copy
import cProfile as profile
import time
import scipy
import scipy.special
import copy
import time

emptyCut=20.0
EMtolfac=0.01
UpdateTol=0.0001
gTol=0.001
need_testmodel=False
do_stashing=False #First test of this worked mostly but small numerical differences and only 3% time saving.  Maybe do more experiments
useSkipFac=False

displayCounter=0
displayEvery=4
#class for a Gaussian mixture model
#Basic variables include:
#  k      number of model components  [\kappa]
#  dim    space dimension
#  x      sample point data
#  ndata  point data N
# Z params:
#  w      pointwise component weights [\hat\gamma]
# theta params:
#  V
#  nu
#  rho
#  beta
#  lamb
# eta params:
#  eta1-5 
#  Ncomp   component effective count 
##options
doProjection=False

log2pi=math.log(2*math.pi)
logpi=math.log(math.pi)

def checkF(logdetcov,logdetcov0,D,N,nu0,beta0):
    #This is what I get for the objective function in the case kappa=1
    #Note the result is independent of Y!
    #We understand cov to be the expected value (nu*V)^-1
    #so logdet(V) = -logdet(cov)-D log(nu)
    nu=nu0+N
    #print("nu,lnu,ldc=",nu,math.log(nu),logdetcov)
    t1=0.5*nu*(-logdetcov-D*math.log(nu))
    t2=-0.5*nu0*(-logdetcov0-D*math.log(nu0))
    t3=-0.5*N*D*math.logpi
    t4=-0.5*D*math.log(1+N/beta0)
    t5=lnGammaDhalf(nu,D)-lnGammaDhalf(nu0,D)
    #print("checkFparts:",t1,t2,t3,t4,t5)
    #print("D/2*NlnN=",D*N/2.0*(math.log(N)))
    return t1+t2+t3+t4+t5
    
def sampleMultivarStudents(nu,mu,sigma):
    #Multivariate Student's t distribution constructed from
    #chi-squared rescaled from from multivariate normal (as in wikipedia)
    u=np.random.chisquare(nu)
    y=np.random.multivariate_normal(np.zeros_like(mu),sigma)
    fac=math.sqrt(nu/u)
    x=mu+y*fac
    return x

def sampleComponentPosteriorPredictive(Vinv,nu,rho,beta):
    #Realize single component part of Eq 32 \ref{eq:posteriorPredictive} 
    D=len(rho)
    sigma=(1+1/beta)/(1+nu-D)*Vinv
    v=nu+1-D
    return sampleMultivarStudents(v,rho,sigma)

def digammaDhalf(x,D):
    val=0
    for i in range(D):
        val+=scipy.special.digamma((x-i)/2.0)
    return val

def lnGammaDhalf(x,D):
    assert(isinstance(D, int))
    val=logpi*(D*(D-1)/2.0)#had /4.0 here before, but that seemed wrong...
    for i in range(D):
        val+=scipy.special.gammaln((x-i)/2.0)
    return val

def get_uninformative_eta0(kappa,D=None,Y=None):
    #Here we set parameters for and uniformative prior
    #We leave the option to cheat a little by generally scaling things by gross data (Y) properties
    if(Y is not None):
        #We have data: Y should be np 2D array of points x components
        D=len(Y[0,:])
        nu0=D+2.0
        #print "Y=",Y[:,0] 
        #min0=np.amin(Y[:,0])
        #print "min0=",min0
        mins = np.array([ np.amin(Y[:,i]) for i in range(D) ])
        maxs = np.array([ np.amax(Y[:,i]) for i in range(D) ])
        #For the precision matrix, V, want V*nu0 = diag(1/(max-min))
        wid=0.5*np.ones(D)#*(maxs-mins)
        #For means set middle points
        rho0=0.5*np.ones(D)#*(maxs+mins)
        #For the covariance scaling parameter set something smallish
        #beta0=0.1
        #beta0=nu0
        beta0=0.01
        Vinv0=np.diag(wid*wid*nu0*beta0/4.0)
    elif(D is not None):
        fail
        nu0=D*1.0
        Vinv0=np.diag(np.ones(D))
        rho0=np.zeros(D)
        beta0=1e-8
    else:
        raise ValueError("Must set either D or Y")
    #set uniform Dirichlet distributions
    lamb0=2.0
    
    eta=[None]*5
    eta[0]=Vinv0+beta0*np.outer(rho0,rho0)
    eta[1]=nu0-D*1.0
    eta[2]=beta0*rho0
    eta[3]=beta0
    eta[4]=lamb0

    return eta

def compute_derived_theta(kappa,D,eta):
    #This operates over all components in vector form
    #note: eta = [ V^-1 + beta*rho*rho , nu - D , beta*rho , beta , lamb ]
    #could make this more efficient by only operating on active components
    #for i in range(5):print("eta[",i,"]=",eta[i])
    nu    = eta[1] + D
    beta  = eta[3].copy() 
    lamb  = eta[4].copy() 
    rho   = (eta[2].T/beta).T
    Vinv  = np.array([ eta[0][j] - beta[j]*np.outer(rho[j],rho[j]) for j in range(kappa )])
    Vnu   = np.array([ nu[j] * np.linalg.pinv( Vinv[j] ) for j in range(kappa )])
    logdetV  = np.array([ -np.linalg.slogdet( Vinv[j] )[1] for j in range(kappa) ])
    return nu,beta,lamb,rho,Vinv,Vnu,logdetV
    
def compute_A_NW(D,nu,beta,logdetV):
    return nu/2.0*(logdetV+D*math.log(2.0))+lnGammaDhalf(nu,D)-D/2.0*math.log(beta)+D/2.0*log2pi

def compute_A_D(lamb):
    kappa=len(lamb)
    barlamb=sum(lamb)
    result=sum([scipy.special.gammaln(lamb[k]) for k in range(kappa)])
    result-= scipy.special.gammaln(barlamb)
    return result

 
def compute_Aeta(D,nu,beta,lamb,logdetV):
    kappa=len(lamb)
    Acomp=[None]*kappa
    for j in range(kappa):
        Acomp[j]=compute_A_NW(D,nu[j],beta[j],logdetV[j])
    A_Dval=compute_A_D(lamb)
    return sum(Acomp)+A_Dval,Acomp,A_Dval

def ellipse(rho,cov,ic,ix,iy,siglev=2,N_thetas=60):
        #print "ic,||cov||,|cov|",ic,np.linalg.norm(cov),np.linalg.det(cov)
        rcov=(cov[np.ix_([ix,iy],[ix,iy])])
        #print "   ||rcov||,|rcov|,rcov",np.linalg.norm(rcov),np.linalg.det(rcov),rcov
        #cov=np.diag(np.ones(self.D)*0.01)
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
        elxs=[rho[iy]+lev_fac*xcoeff*(acoeff*math.cos(th)*math.cos(ang)-bcoeff*math.sin(th)*math.sin(ang)) for th in thetas] 
        elys=[rho[ix]+lev_fac*ycoeff*(acoeff*math.cos(th)*math.sin(ang)+bcoeff*math.sin(th)*math.cos(ang)) for th in thetas]
        return elys,elxs
        
class gmmvb:
    def __init__(self,Y,kappa,kappa_penalty=0.1,logistic=False):
        self.have_stash=False
        self.needFalt=False
        self.Y=np.array(Y)
        self.N=len(Y)
        self.D=len(Y[0])
        self.logistic=logistic
        self.set_map()
        self.Y2=np.array([np.outer(self.Y[i,:],self.Y[i,:]) for i in range(self.N)])
        self.kappa=kappa
        self.kappa_penalty=kappa_penalty
        self.eta0=get_uninformative_eta0(self.kappa,self.D,self.Y)
        self.Ncomp=[None]*self.kappa
        if(self.needFalt):self.gammabar = np.zeros((self.N,self.kappa))
        self.resetActive(range(self.kappa))
        #save prior theta stuff
        self.nu0,self.beta0,self.lamb0,self.rho0,self.Vinv0,self.Vnu0,self.logdetV0=compute_derived_theta(self.kappa,self.D,[np.array([self.eta0[i]]*self.kappa) for i in range(5)])
        self.A0,self.A_comp0,self.A_Dval0=compute_Aeta(self.D,self.nu0,self.beta0,self.lamb0,self.logdetV0)
        print ("Initializing with prior:")
        for k in self.activeComponents:print ("rho",k,self.rho0[k],self.Ncomp[k],self.logdetV0[k],self.logdetV0[k]+math.log((1+1.0/self.beta0[k])/max([1,self.nu0[k]-self.D-1]))*self.D,self.beta0[k],"\n Sigma-eigvals",np.linalg.eigvalsh(self.Vinv0[k]*((1+1.0/self.beta0[k])/max([1,self.nu0[k]-self.D-1]))))
        self.eta=[np.array([self.eta0[i]]*self.kappa) for i in range(5)] #initialize with prior        
        self.updated_eta=True
        self.nu,self.beta,self.lamb,self.rho,self.Vinv,self.Vnu,self.logdetV=compute_derived_theta(self.kappa,self.D,self.eta)
        self.updated_theta=True
        self.Aeta,self.A_comp,self.A_Dval=compute_Aeta(self.D,self.nu,self.beta,self.lamb,self.logdetV)
        
        #randomly initialize rhos from data point
        self.rho=np.array([Y[i] for i in np.random.choice(self.N,self.kappa)])
        self.hatgamma=np.zeros((self.N,self.kappa))
        self.have_hatgamma=False
        self.request_stash=False
        self.F=None
        self.Falt=None
        self.like=None
        self.EMtol=self.eta0[3]*EMtolfac
        self.skips={}
        self.show()
        
    def copy(self):
        #return copy.deepcopy(self)
        return self.copyA()
    
    def copyA(self):
        other=copy.copy(self)
        other.Ncomp=copy.deepcopy(self.Ncomp)
        if(self.needFalt):other.gammabar = self.gammabar.copy
        other.nu0,other.beta0,other.lamb0,other.rho0,other.Vinv0,other.Vnu0,other.logdetV0=compute_derived_theta(other.kappa,other.D,[np.array([other.eta0[i]]*other.kappa) for i in range(5)])
        other.A0,other.A_comp0,other.A_Dval0=compute_Aeta(other.D,other.nu0,other.beta0,other.lamb0,other.logdetV0)
        other.eta=copy.deepcopy(self.eta)
        other.nu,other.beta,other.lamb,other.rho,other.Vinv,other.Vnu,other.logdetV=compute_derived_theta(other.kappa,other.D,other.eta)
        other.Aeta,other.A_comp,other.A_Dval=compute_Aeta(other.D,other.nu,other.beta,other.lamb,other.logdetV)
        
        #randomly initialize rhos from data point
        other.rho=self.rho.copy()
        other.hatgamma=self.hatgamma
        return other
    
    def compute_derived_theta(self):
        #This operates over all components in vector form
        #note: eta = [ V^-1 + beta*rho*rho , nu - D , beta*rho , beta , lamb ]
        #could make this more efficient by only operating on active components
        assert self.updated_eta
        self.nu,self.beta,self.lamb,self.rho,self.Vinv,self.Vnu,self.logdetV=compute_derived_theta(self.kappa,self.D,self.eta)
        self.updated_theta=True
        self.updated_Aeta=False
        
    def show(self,lead="  "):
        print (lead,"Variational Bayesian Gaussian mixture model: ",self.kappa,"components")
        #print ("eta0:",self.eta[0])
        #print ("eta1:",self.eta[1])
        #print ("eta2:",self.eta[2])
        #print ("eta3:",self.eta[3])
        #print ("eta4:",self.eta[4])
        print (lead,"th: nu,beta,lamb=",self.nu,self.beta,self.lamb,"\n",lead," rho=",self.rho,"\n",lead," logdetVnu=",self.logdetV+self.D*np.log(self.nu))
        #print (lead,"Ncomp=",self.Ncomp)
        print (lead,"partitions=",np.sum(self.hatgamma,axis=0))
        #print ("min/max:sum(hatgamma[i,:])=",min(np.sum(self.hatgamma,axis=1)),max(np.sum(self.hatgamma,axis=1)))
        
    def update_Aeta(self):
        self.Aeta,self.A_comp,self.A_Dval=compute_Aeta(self.D,self.nu,self.beta,self.lamb,self.logdetV)
        self.updated_Aeta=True
        
    def computeF(self):
        #This computation of the objective function assumes that the model
        #parameters have just been updated (maximization step)
        #needvars:hatgamma,Aeta,A0
        #Eq 29 \ref{eq:computeF}
        assert self.updated_theta
        assert self.updated_Aeta
        if(self.have_stash):N=self.stash_N
        else:N=self.N
        D=self.D
        val0  = -N*D/2.0*log2pi
        glogg= sum([self.hatgamma[i,j]*math.log(1e-300+self.hatgamma[i,j]) for j in range(self.kappa) for i in range(self.N)])
        if(self.have_stash):glogg+=self.stash_glogg
        if(False):
            for i in range(self.N):
                gvec=[self.hatgamma[i,j]*math.log(1e-300+self.hatgamma[i,j]) for j in range(self.kappa)]
                #print min(gvec)
                if(min(self.hatgamma[i])>1e-4):print ("glog:",min(self.hatgamma[i]),gvec,self.hatgamma[i,:],self.Y[i])
        val = val0-glogg
        val += self.Aeta - self.A0;
        val -= self.kappa_penalty*math.sqrt(self.N)*self.kappa
        return val

    def computeFalt(self):
        #Compute the objective function F using eq 30 from notes
        #Assumes that the partition fractions have just been
        #updated (expectation step)
        #This computation is not going to be particularly fast or optimized
        assert self.updated_gamma
        #print("gbi=",np.sum(self.gammabar,1))
        #print("lgbi=",np.log(np.sum(self.gammabar,1)))
        like = sum(np.log(np.sum(self.gammabar,1)))
        barlamb = sum(self.lamb)
        barlamb0 = sum(self.lamb0)
        Dterm0 = ( (barlamb-barlamb0) * scipy.special.digamma(barlamb)
                   + scipy.special.gammaln(barlamb0) - scipy.special.gammaln(barlamb) )      
        Dterm = Dterm0 + sum([-(self.lamb[j]-self.lamb0[j])*scipy.special.digamma(self.lamb[j]) + scipy.special.gammaln(self.lamb[j]) - scipy.special.gammaln(self.lamb0[j]) for j in range(self.kappa)])
        drho=self.rho-self.rho0
        Vterm = 0.5 * sum([
            self.D*self.nu[j] - np.einsum('ij,ji->', self.Vinv0[j],self.Vnu[j])
            +self.nu0[j]*(self.logdetV[j]-self.logdetV0[j])
            -self.beta0[j]*np.einsum('i,ij,j->',drho[j],self.Vnu[j],drho[j])
            for j in range(self.kappa) ])
        nuterm = - sum([
            0.5*(self.nu[j]-self.nu0[j])*digammaDhalf(self.nu[j],self.D)
            -lnGammaDhalf(self.nu[j],self.D)+lnGammaDhalf(self.nu0[j],self.D)
            for j in range(self.kappa) ])
        #print("beta0,beta:",self.beta0[0],self.beta)
        betaterm = - self.D/2.0 * sum([
            math.log(self.beta[j]/self.beta0[j])
            - 1.0 + self.beta0[j]/self.beta[j]
            for j in range(self.kappa) ])
        #print("altF parts:",like, Dterm, Vterm, nuterm, betaterm)
        penalty = self.kappa_penalty*math.sqrt(self.N)*self.kappa

        return like + (  Dterm + Vterm + nuterm + betaterm ) - penalty
    
    def resetActive(self,newActive):
        self.stashRestore()
        self.activeComponents=newActive
        self.have_g=False;
        
    def expectationStep(self):
        #Here we compute the meta_parameters for q(Z), that is hat_gamma,
        #using [Eqs after (34)] from the notes
        #requires: updated eta and derived theta parameters
        #produces: updated hatgamma[i,j] (for active components)
        

        #If we need to compute g[i] we do that now
        #Note that we are assuming that the maximization hasn't been called
        #since activeComponents was updated
        #print("Entering expectation Step: Active=",self.activeComponents)
        assert self.updated_theta
        if( not self.have_g ):
            assert(not self.have_stash)
            if(len(self.activeComponents)<self.kappa):
                #print("recomputing g")
                #We define this using the *inactive components* since we sometimes delete one of the actives
                self.g=np.array([1.0 - sum([self.hatgamma[i,j] for j in range(self.kappa) if not j in self.activeComponents ]) for i in range(self.N)])
            else:
                self.g=np.ones(self.N)
            self.have_g=True
        elif( self.request_stash ):
            self.stashInactiveData()
        
        #Pre-compute data-independent terms from Eq(24) from notes
        barlamb=sum(self.lamb)
        log_gamma0=scipy.special.digamma(self.lamb)-scipy.special.digamma(barlamb)-self.D*logpi/2
        for j in self.activeComponents:
            log_gamma0[j] += ( self.logdetV[j] + digammaDhalf(self.nu[j],self.D) - self.D/self.beta[j] ) / 2.0
        self.logLike=0
        #Now loop over the data for the rest of the terms and normalization
        for i in range(self.N):
            if(self.g[i]<gTol):continue #Leave insignificant hatgamma values unchanged for insignificantly overlapping data
            #First we compute log(gamma) as needed
            log_gamma=np.zeros(len(self.activeComponents))
            for k in range(len(self.activeComponents)):
                j=self.activeComponents[k]
                log_gamma[k]=log_gamma0[j]
                #Data-dependent terms from Eq(24) from notes
                #dy=yi-rho
                #log_gamma=-dy.Vnu.dy/2+log_gamma_offset
                dy=self.Y[i]-self.rho[j]
                vec=np.dot(self.Vnu[j],dy/2)
                log_gamma[k] += -np.dot(dy,vec)
                #print "lg",log_gamma[k],-np.dot(dy,vec),dy,vec
                #may want to skip this if not computing Falt
                if(self.needFalt):self.gammabar[i,k]=math.exp(log_gamma[k])
                #print("gb[",i,",",k,"]=",self.gammabar[i,k])
            #Now the exponentials of log_gamma
            gamma=np.zeros(len(self.activeComponents))
            log_gamma_baseline=max(log_gamma)
            for k in range(len(self.activeComponents)):
                #Subtract off a reference baseline to avoid floating point issues
                gamma[k]=math.exp(log_gamma[k]-log_gamma_baseline)

            #Update the likelihood

            #Next we normalize
            normfac=self.g[i]/sum(gamma);
            for k in range(len(self.activeComponents)):
                j=self.activeComponents[k]
                self.hatgamma[i,j]=gamma[k]*normfac
                #print "i,j,hatgamma,gamma,normfac,g",i,j,self.hatgamma[i,j],gamma[k],normfac,self.g[i]
        if(not self.have_g):self.have_g=True
        self.updated_eta=False
        self.updated_theta=False
        self.updated_gamma=True
            
    def componentOverlap(self,j1,j2):
        assert(not self.have_stash)
        #print("computing overlap:")
        #for i in range(self.N):
        #    print(" ",i,self.hatgamma[i,j1],self.hatgamma[i,j2])
        return np.sum(self.hatgamma[:,j1]*self.hatgamma[:,j2])

    def maximizationStep(self):
        if(not self.have_g):raise ValueError("Should call expectationStep before maximizationStep after changing activeComponents")
        #Realizes \ref{eq:meta-update} = Eq. 28 of notes
        #requires: updated hatgamma[i,j]
        #produces: updated eta/theta (for active components)
        
        #if(self.activeComponents is None):activeComponents=range(self.k)

        #first compute needed component-weighted sums
        yNcomp=[None]*self.kappa
        yyNcomp=[None]*self.kappa
        for j in self.activeComponents:
            self.Ncomp[j]=np.sum(self.hatgamma[:,j])
            yNcomp[j]=np.sum((self.hatgamma[:,j]*self.Y.T).T,0)
            yyNcomp[j]=np.sum((self.hatgamma[:,j]*self.Y2.T).T,0)
            if(self.have_stash):
                self.Ncomp[j]+=self.stash_Ncomp[j]
                yNcomp[j]+=self.stash_yNcomp[j]
                yyNcomp[j]+=self.stash_yyNcomp[j]
        #print("Ncomp=",self.Ncomp)
        #print("yNcomp=",yNcomp)
        #print("yyNcomp=",yyNcomp)
        
        for j in self.activeComponents:
            #print "j,Ncomp,yNcomp,eigvals(yyNcomp-brr)",j,self.Ncomp[j],yNcomp[j],np.linalg.eigvalsh(yyNcomp[j]-np.outer(yNcomp[j],yNcomp[j])/self.Ncomp[j])
            self.eta[0][j]=self.eta0[0]+yyNcomp[j]
            self.eta[1][j]=self.eta0[1]+self.Ncomp[j]
            self.eta[2][j]=self.eta0[2]+yNcomp[j]
            self.eta[3][j]=self.eta0[3]+self.Ncomp[j]
            self.eta[4][j]=self.eta0[4]+self.Ncomp[j]
            #print "eta1=",self.eta[1][j]
        self.updated_eta=True
        self.updated_gamma=False
        self.compute_derived_theta()
        #for j in self.activeComponents:
            #sMean=yNcomp[j]/self.Ncomp[j]
            #sVar=yyNcomp[j]/self.Ncomp[j]-np.outer(sMean,sMean)
            #print j,"Vinv/nu,sVar:\n",(self.Vinv[j]/self.nu[j]),"\n",(sVar)
            #print (j,"eigvals:Vinv/nu,sVar:\n",np.linalg.eigvalsh(self.Vinv[j]/self.nu[j]),"\n",np.linalg.eigvalsh(sVar))
            #print ("rho-sMean:",self.rho[j]-sMean)

    def project_partitions(self, dhgamma, dhgammaP1):
        assert(not self.have_stash)
        self.update_Aeta()
        Fval=self.computeF()
        #Here we take a step away from strict EM and try to accelerated
        #the process by projecting toward the conclusion of process.
        #The idea is that the EM process produces a sequence of dN vectors
        #And we can try to extrapolate that sequence.  For the first version
        #of this we suppose that the trend is linear, independently for each
        #component, concluding at dhgamma(j,t,i) = 0 for  some t0(j,i).  The result is
        # ratio = dhgamma[-1]/dhgamma[0].
        # t0 = 1 / ( ratio - 1 ).
        #This clearly doesn't make sense if ratio(i,j)<1.  We then suppose that
        #we leap ahead some fraction (alpha) of that time.  To do so we integrate
        #to compute the total step.  We get:
        # deltahgamma = alpha*(1-alpha/2)*ddhgamma[0]/(ratio-1)
        #Finally we have to assure that sum(deltahgamma(i))=0. To realize this we just work with
        #a common mean of the  ratio for all components then deltahgamma 
        #will be a direct rescaling of dhgamma[0].
        #print("Trying projection")
        self.savehgamma=self.hatgamma
        alpha=0.25
        fac=alpha*(1-alpha/2)
        ratios=dhgamma/dhgammaP1
        for i in range(self.N):
            scaling=0
            if( all( r > 0 for r in ratios[i]) ):
                scaling=np.mean(ratios[i])*fac
            if(scaling>0):
                newhg=self.hatgamma[i]+dhgamma[i]*scaling
                if( all( hg > 0 for hg in newhg )):
                    self.hatgamma[i]=newhg
        self.maximizationStep()
        self.update_Aeta()
        testFval=self.computeF()
        print("Fval,test:",Fval,testFval)
        if(not testFval>Fval):
            #abort
            print("Failed")
            self.hatgamma=self.savehgamma
            self.maximizationStep()
            return False
        print("Succeeded")

        return True
                    
    def run_EM(self,backend=None,MaxSteps=800):
        assert(self.updated_theta)
        Fval=-float('inf')
        #self.expectationStep()
        
        print ("Running EM active =",self.activeComponents)
        start=time.time()

        every=100
        Ncomp_old=np.array(self.Ncomp)
        dhgamma=None
        if(doProjection):oldhgamma=self.hatgamma.copy()
        for count in range(MaxSteps):
            self.expectationStep()
            dhgammaP1=dhgamma
            if(doProjection and oldhgamma is not None ):dhgamma=self.hatgamma-oldhgamma
            if(self.needFalt):altFval=self.computeFalt()
            if(doProjection):
                if(dhgammaP1 is not None):
                    self.maximizationStep()
                    if self.project_partitions(dhgamma,dhgammaP1):
                        oldhgamma=None
                        dhgamma=None
                    else:
                        oldhgamma=self.hatgamma.copy()
                else:
                    self.maximizationStep()
                    oldhgamma=self.hatgamma.copy()
            else:
                self.maximizationStep()
            Ncomp=np.array(self.Ncomp)
            if(count>0):
                dNcomp = Ncomp-Ncomp_old
                dNcomp2=np.linalg.norm(dNcomp)**2
            else:
                if(self.have_stash):dNcomp2=self.stash_N**2
                else:dNcomp2=self.N**2
            
            #if self.activeComponents is not None:
            #    jlist=self.activeComponents
            #else:
            #    jlist=range(self.kappa)
            #Fvalold=Fval
            if(False and self.activeComponents is not None):
                for j in self.activeComponents: print (" mu[",j,"]=",self.mu[j])
            if(count%every==0):
                if(count>0):print (count,"dNcomp=",dNcomp2,Ncomp-Ncomp_old)
                #for k in self.activeComponents:print "rho",k,self.rho[k],self.Ncomp[k],self.logdetV[k],self.logdetV[k]+math.log((1+1.0/self.beta[k])/max([1,self.nu[k]-self.D-1]))*self.D,self.beta[k],"\n Sigma-eigvals",np.linalg.eigvalsh(self.Vinv[k]*((1+1.0/self.beta[k])/max([1,self.nu[k]-self.D-1])))
                if(self.needFalt):print (count,"altFval=",altFval)
                self.update_Aeta()
                Fval=self.computeF()
                print (count,"Fval=",Fval)
                print (count,"Ncomp=",self.Ncomp)
                #print (count,"lndet=",self.logdetV+self.D*np.log(self.nu))
            Ncomp_old=Ncomp.copy()
            #if(count/every>3):every*=2
            #print "mu1=",self.mu[1]
            if(backend is not None):
                global displayCounter, displayEvery
                if(displayCounter%displayEvery==0):
                    print ("plotting ",displayCounter/displayEvery)
                    clusters=self.draw_clusters()
                    self.plot(clusters,0,1,None,backend)
                displayCounter+=1
            #print ("test:",Fval-Fvalold,"<",self.EMtol)
            #if(Fval-Fvalold<self.EMtol):
            #break
            if(dNcomp2<self.EMtol):
                break
        self.updated_gamma=True
        if(self.needFalt):altFval=self.computeFalt()
        self.update_Aeta()
        Fval=self.computeF()
        self.F=Fval

        if(self.needFalt):print ("best Fval/alt=",Fval,altFval)
        else: print ("best Fval",Fval)
        if(False):
            for j in range(self.kappa):
                #print("inv(Vnu)=",np.linalg.pinv(self.Vnu[j]))
                logdetcov  = -np.linalg.slogdet( self.Vnu[j] )[1]
                logdetcov0  = -np.linalg.slogdet( self.Vnu0[j] )[1]
                print ("checkF(",j,"):",checkF(logdetcov,logdetcov0,self.D,self.Ncomp[j],self.nu0[j],self.beta0[j]))
                print ("Fcomp(",j,"):",self.A_comp[j]-self.A_comp0[j]-self.Ncomp[j]*self.D/2.0*log2pi)
                #print("coeff=",-1-logdetcov+self.D*math.log(2*math.pi))
                #print ("largeN: F->",self.Ncomp[j]/2*(-1-logdetcov+self.D*log2pi)-(self.D/4.)*(self.D+3)*math.log(self.Ncomp[j]))
        dtime=time.time()-start
        print(" count=",count)
        print("               EM time:",dtime)
        
    def stashInactiveData(self):
        #This is an experimental idea to improve efficiency for large N cases
        #If there are many N which are not significant for the activeComponents
        #then we waste some time e.g. in the maximization step in working with
        #them.  Here we identify them and save the relevant info about them.
        assert(not self.have_stash)
        stash_map=[ i for i in range(self.N) if self.g[i]>=gTol ]
        self.stash_map=stash_map
        self.stash_Y=self.Y.copy()
        self.Y=self.Y[stash_map]
        self.stash_Y2=self.Y2.copy()
        self.Y2=self.Y2[stash_map]
        self.stash_N=self.N
        self.N=len(self.Y)
        print("stashing down to N=",self.N)
        self.stash_hatgamma=self.hatgamma.copy()
        self.hatgamma=self.hatgamma[stash_map]
        self.stash_g=self.g.copy()
        self.g=self.g[stash_map]
        self.stash_Ncomp=[None]*self.kappa
        self.stash_yNcomp=[None]*self.kappa
        self.stash_yyNcomp=[None]*self.kappa
        self.stash_glogg=sum([self.stash_hatgamma[i,j]*math.log(1e-300+self.stash_hatgamma[i,j]) for j in range(self.kappa) for i in range(self.stash_N)])-sum([self.hatgamma[i,j]*math.log(1e-300+self.hatgamma[i,j]) for j in range(self.kappa) for i in range(self.N)])
        for j in self.activeComponents:
            #print "shapes: ",self.hatgamma[:,j].shape,self.Y.shape,self.Y2.shape
            self.stash_Ncomp[j]=np.sum(self.stash_hatgamma[:,j])-np.sum(self.hatgamma[:,j])
            #print " y shape ",(self.hatgamma[:,j]*self.Y.T).T.shape
            self.stash_yNcomp[j]=np.sum((self.stash_hatgamma[:,j]*self.stash_Y.T).T,0)-np.sum((self.hatgamma[:,j]*self.Y.T).T,0)
            #print " yy shape ",(self.hatgamma[:,j]*self.Y2.T).T.shape
            self.stash_yyNcomp[j]=np.sum((self.stash_hatgamma[:,j]*self.stash_Y2.T).T,0)-np.sum((self.hatgamma[:,j]*self.Y2.T).T,0)
            #print "->shapes: ",yNcomp[j].shape,yyNcomp[j].shape
        self.have_stash=True
        self.request_stash=False

    def stashRestore(self):
        #This is an experimental idea to improve efficiency for large N cases
        #If there are many N which are not significant for the activeComponents
        #then we waste some time e.g. in the maximization step in working with
        #them.  Here we identify them and save the relevant info about them.
        self.have_g=False
        self.have_hatgamma=False
        if( not self.have_stash):return
        stash_map=self.stash_map
        self.Y=self.stash_Y
        self.Y2=self.stash_Y2
        self.N=len(self.Y)
        #for i in range(len(stash_map)):
        #    self.stash_hatgamma[stash_map[i]]=self.hatgamma[i]
        #    self.stash_g[stash_map[i]]=self.g[i]
        #self.hatgamma=self.stash_hatgamma
        #self.g=self.stash_g
        self.hatgamma=np.zeros((self.N,self.kappa))
        self.have_stash=False
        #self.have_g=False
        #self.have_hatgamma=False
        
    def splitComponent(self,j):
        #This is a crucial element of the improve-structure process
        #A component is split into two new ones
        #We define the split in terms of the derived theta, then
        #update eta by a maximization step...

        #To set up the split versions of component j:
        #  -First draw a few samples from the posterior predictive
        #  -Then choose the one farthest from the mean x0
        #    -This point will replace the mean
        #  -The set a new component mean x1 reflected opposite the original
        #    -This component will be added on the end
        #  -The rest of the theta params are defined as in Eq 34 \ref{eq:split}
        print("split:entry:Ncomp=",self.Ncomp)
        #print("Before split:");self.show()
        assert(self.updated_theta)
        assert(not self.have_stash)
        x0s=[sampleComponentPosteriorPredictive(self.Vinv[j],self.nu[j],self.rho[j],self.beta[j]) for i in range(3)]
        dists=[np.linalg.norm(x0-self.rho[j]) for x0 in x0s]
        maxdist=max(dists)
        x0=x0s[dists.index(maxdist)]
        x1=-x0+2.0*self.rho[j] #thus x0-rho + x1-rho = 0
        nu0 = self.eta0[1] + self.D
        nuval    = 0.5*( self.nu[j] + nu0 )
        betaval  = 0.5*( self.beta[j]  + self.eta0[3] ) 
        lambval  = 0.5*( self.lamb[j]  + self.eta0[4] ) 
        Vinvval  = 2.0/(1.0+nu0/self.nu[j])*self.Vinv[j]
        Vnuval   = nuval * np.linalg.pinv( Vinvval )
        logdetVval  = -np.linalg.slogdet( Vinvval )[1]
        self.rho   = np.concatenate(( self.rho    , [    x1   ]  ))
        self.nu    = np.concatenate(( self.nu     , [  nuval  ]  ))
        self.beta  = np.concatenate(( self.beta   , [ betaval ]  ))
        self.lamb  = np.concatenate(( self.lamb   , [ lambval ]  ))
        self.Vinv  = np.concatenate(( self.Vinv   , [ Vinvval ]  ))
        self.Vnu   = np.concatenate(( self.Vnu    , [ Vnuval  ]  ))
        self.logdetV=np.concatenate(( self.logdetV, [logdetVval] ))
        self.rho[j]    = x0.copy();
        self.nu[j]     = nuval;
        self.beta[j]   = betaval;
        self.lamb[j]   = lambval;
        self.Vinv[j]   = Vinvval.copy();
        self.Vnu[j]    = Vnuval.copy();
        self.logdetV[j]= logdetVval;
        actives=self.activeComponents+[self.kappa]
        self.kappa+=1
        #self.hatgamma=np.zeros((self.N,self.kappa))
        self.A_comp = [None]*self.kappa
        self.activeComponents=actives.copy()
        print("Split:actives=",self.activeComponents)
        #We don't call resetActive because, in this case g[i] specifically should not change because of the split
        #Update prior info
        self.nu0,self.beta0,self.lamb0,self.rho0,self.Vinv0,self.Vnu0,self.logdetV0=compute_derived_theta(self.kappa,self.D,[np.array([self.eta0[i]]*self.kappa) for i in range(5)])
        self.A0,self.A_comp0,self.A_Dval0=compute_Aeta(self.D,self.nu0,self.beta0,self.lamb0,self.logdetV0)
        self.update_Aeta()
        newhatgamma=np.zeros((self.N,self.kappa))
        for k in range(self.kappa-1): newhatgamma[:,k]=self.hatgamma[:,k]
        self.hatgamma=newhatgamma
        self.have_hatgamma=False

        self.Ncomp.append(None)
        #print("eta was:",self.eta)
        for i in range(5):self.eta[i]=np.append(self.eta[i],np.array([self.eta0[i]]),axis=0)
        #print("eta --> ",self.eta)
        self.updated_theta=True
        self.updated_eta=False
        self.updated_gamma=False
        #print("After:");self.show()
        self.expectationStep()
        #print("After(Ex):"); self.show()
        self.maximizationStep()
        #print("After(Mx):"); self.show()
        print("Split:Ncomp=",self.Ncomp)
        if(do_stashing):self.request_stash=True
        
    def deleteComponent(self,j):
        #In practice, sometimes the EM algortihm seems to get stuck on a local optimum with an effectively empty component
        #This routine eliminates a target component from the model. Some of this is trivial, but the
        
        print("delete:entry:Ncomp=",self.Ncomp)
        #print("Before:")
        #self.show()
        assert(self.updated_theta)
        self.rho = np.delete(self.rho,j,0)
        self.nu = np.delete(self.nu,j,0)
        self.beta = np.delete(self.beta,j,0)
        self.Vinv = np.delete(self.Vinv,j,0)
        self.Vnu = np.delete(self.Vnu,j,0)
        self.logdetV = np.delete(self.logdetV,j,0)
        self.lamb = np.delete(self.lamb,j,0)
        actives=[ k if k<j else k-1 for k in self.activeComponents if k!=j ]
        self.kappa-=1
        del self.A_comp[j]
        #Update prior info
        self.nu0,self.beta0,self.lamb0,self.rho0,self.Vinv0,self.Vnu0,self.logdetV0=compute_derived_theta(self.kappa,self.D,[np.array([self.eta0[i]]*self.kappa) for i in range(5)])
        self.A0,self.A_comp0,self.A_Dval0=compute_Aeta(self.D,self.nu0,self.beta0,self.lamb0,self.logdetV0)
        self.update_Aeta()
        self.hatgamma = np.delete(self.hatgamma,j,axis=1)
        self.have_hatgamma=False

        ##
        del self.Ncomp[j]
        if(self.needFalt):self.gammabar = np.zeros((self.N,self.kappa))
        #print("eta was:",self.eta)
        for i in range(5):self.eta[i] = np.delete(self.eta[i],j,axis=0)
        #print("eta --> ",self.eta)
        self.updated_theta=True
        self.updated_eta=False
        self.updated_gamma=False
        #print("After:");self.show()
        self.resetActive(range(self.kappa))  #We need to allow all components to absorb any partition weight
        self.expectationStep()
        #print("After(Ex):"); self.show()
        self.maximizationStep()
        #print("After(Mx):"); self.show()
        #print("delete:Ncomp=",self.Ncomp)
        self.resetActive(actives)  #We need to allow all components to absorb any partition weight
        
#probably don't need this...?
    def update(self):
        #set lpost and BICevid
        #print "Evaluating model with "+str(self.k)+" components"
        self.expectationStep()
        self.maximizationStep()
        
        
    def improveStructure(self,updates,backend=None,trials=1):
        #FIXME trials>1 not implemented
        #FIXME outdated note
        #Like V2,  continues through, considering splits for each of
        #the whole (at call time) list of components
        #Other differences from V1 are:
        #    tolOverlap=0.03  Significantly overlapping components are
        #                     co-optimised during the split EMs
        #        tolRelax=10  Provides a relaxed tolerance for split EMs
        #                     though full tol used if it seems that the BIC
        #                     diff is close (less than tolRelax^2)
        ##   nSamp=500    Target number of samples for the component being
        ##                considered for split. Points with w[i,j]<1/Nsamp^2
        ##                are left out of split EM.  If more than Nsamp remain
        ##                these are downsampled
        splits=[False]*self.kappa
        replacements=[]
        self.expectationStep()
        self.maximizationStep()
        self.update_Aeta()
        self.F=self.computeF()
        #print("****",self.F)
        updatelist=np.random.permutation(self.kappa)
        model=copy.deepcopy(self)
        #model=self.copy()
        #model.evaluate(points)
        EMtol=self.EMtol
        tolRelax=10
        tolOverlap=0.03
        jcount=0
        last_updates=updates
        updates=[]
        for j in updatelist:
            jcount+=1
            print ("\nTrying split "+str(jcount)+"/"+str(len(updatelist)))
            #We first compute the overlap of this component with others
            #Use full points w for this
            #overs=[np.sum(w[:,j]*w[:,k]) for k in range(model.kappa)]
            overs=[model.componentOverlap(j,k) for k in range(model.kappa)]
            print("overlaps=",overs)
            overs=np.array(overs)/(sum(overs)+1e-10)

            #As an efficiency move, we do not check for splits of components which have insignificant overlap
            #with those updated last time.  We reasonably expect that not much will have changed since the
            #last time that test was done.
            metric=sum([overs[i] for i in last_updates])
            skipfac=1
            if(useSkipFac and model.skips.get(j) is not None):skipfac=model.skips[j]
            if(skipfac<3 and metric<UpdateTol):
                if(model.skips.get(j) is None):model.skips[j]=1
                else:
                    model.skips[j]=model.skips[j]+1
                print("Skipping split test of component ",j," with metric ",metric," skip",model.skips[j])
                continue
            print ("setting skips[",j,"]=0")
            model.skips[j]=0
            print(model.skips)
            
            active=[i for i in range(len(overs)) if overs[i]>tolOverlap]
            print (j,": overlaps =",overs)
            print ("-> active =",active)

            #We work with a parallel test model for apples to apples comparison
            #Alternatively, it might be better to run EM on the model to convergence
            #(over the overlapping components) at this point...
            model.resetActive(active)
            model.run_EM()
            newmodel=copy.deepcopy(model)
            #newmodel=model.copy()
            newmodel.EMtol=EMtol*tolRelax
            if(need_testmodel):
                testmodel=copy.deepcopy(newmodel)
                #testmodel=newmodel.copy()
            newmodel.splitComponent(j)
            newmodel.run_EM()
            if(need_testmodel):
                #testmodel.EMtol=EMtol*tolRelax
                testmodel.run_EM()
                #print ("testmodel=");testmodel.show()
                print ("testmodel.F=",testmodel.F)

            #tol-2 cycle
            #print ("newmodel=");newmodel.show()
            print ("newmodel.F=",newmodel.F)
            if(need_testmodel):testF=testmodel.F
            else:testF=model.F
            if(newmodel.F>testF and testF-newmodel.F<EMtol*tolRelax**2):#This is a close call continue EM with orig EMtol
                newmodel.EMtol=EMtol
                if(need_testmodel):testmodel.EMtol=EMtol
            #Run a second time for "trimming" [resets relative weights]
            newmodel.run_EM()
            if(need_testmodel):testmodel.run_EM()
            print ("Compare models")
            if(need_testmodel):print ("testmodel.F=",testmodel.F)
            print ("newmodel.F=",newmodel.F)
            print ("model.F=",model.F)

            #newmodel.update(points)  
            #newmodel.computeF()
            print ("actives=",newmodel.activeComponents)
            print ("Fvals orig/new",model.F,newmodel.F)
            #print ("lposts orig/new",model.lpost,newmodel.lpost)
            if(need_testmodel):testF=testmodel.F
            else:testF=model.F
            if(newmodel.F>model.F or testF>model.F):
                print ("Found improved model:")
                if(newmodel.F>testF):
                    print("New model is better")
                    model=newmodel
                    updates.append(j)
                    updates.append(model.kappa-1)
                else:
                    print("Test model is better")
                    model=testmodel
                model.EMtol=EMtol
                #we accept the newmodel as model
                #we don't run full EM, but we do need to update w and we
                #(do?) update the model params
                model.resetActive(range(model.kappa))
                model.expectationStep()#update w
                model.maximizationStep()
                print ("evaluated model: F=",model.F)
                if(True):
                    phisum=np.sum(self.Ncomp)
                    print (" [before] =",np.array2string(np.array(model.Ncomp),formatter={'float_kind':lambda x: "%.5f" % x}))
                    print ("lndetV=",model.logdetV)
                    print ("model.rho =\n",np.array2string(np.array(model.rho),formatter={'float_kind':lambda x: "%.3f" % x}))
                print ("model.phi=",np.array2string(np.array(model.Ncomp),formatter={'float_kind':lambda x: "%.5f" % x}))
                if(backend is not None):#do it all over for display purposes
                    print("Need to rework this.....")
                    #newmodel=gmm(newmu,model.kappa,newsigma,newphi)
                    #newmodel.EMtol=EMtol*tolRelax
                    #newmodel.run_EM(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)
                    #if(newmodel.F>model.F and model.F-newmodel.F<EMtol*tolRelax**2):newmodel.EMtol=EMtol
                    #newmodel.run_EM(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)

        print("before deletions: skips=",model.skips)

        #Sometimes the EM optimization process leads to some empty or near-empty components
        #For truly empty components it should always be the case that the fit is more optimal if they are removed
        #We don't assume this though, we check.
        #print("Checking for empty components")
        empties=[ j for j in range(model.kappa) if model.Ncomp[j] < emptyCut ]
        print("\nChecking for deletions: = ",empties)
        print("  empties = ",empties)
        Nempty=len(empties)
        deletions=0
        
        #We follow the same procedure for checking the deletions as we did for testing the splits
        jcount=0
        for j in range(model.kappa):
            done=False
            while(not done):
                if not j in empties:
                    done=True
                    continue
                #We do the loop over empties this way since we may need to relabel some as we go.
                jcount+=1
                print ("\nTrying deletion "+str(jcount)+"/"+str(Nempty)," j=",j, "Ncomp[j]=",model.Ncomp[j])

                #We first compute the overlap of this component with others
                #Since the component is nearly empty, it won't be much.
                overs=[model.componentOverlap(j,k) for k in range(model.kappa)]
                print("overlaps=",overs)
                overs=np.array(overs)/(sum(overs)+1e-10)
                active=[i for i in range(len(overs)) if overs[i]>tolOverlap]
                print (j,": overlaps =",overs)
                print ("-> active =",active)
                
                #We don't bother with running a parallel test model in this case
                newmodel=copy.deepcopy(model)
                #newmodel=model.copy()
                newmodel.resetActive(active)
                newmodel.deleteComponent(j)
                newmodel.EMtol=EMtol
                newmodel.run_EM()
                
                print ("Compare models (deletion)")
                print ("newmodel.F=",newmodel.F)
                print ("model.F=",model.F)
                
                print ("actives=",newmodel.activeComponents)
                print ("Fvals orig/new",model.F,newmodel.F)

                if(newmodel.F>model.F):
                    print ("Deleting non-optimal component with <N>=",model.Ncomp[j])
                    #We accept the newmodel as model
                    model=newmodel
                    #Relabel updates and empties
                    empties=[ k if k<j else k-1 for k in empties if k!=j ]
                    updates=[ k if k<j else k-1 for k in updates if k!=j ]
                    newskips=[ (k,model.skips[k]) if k<j else (k-1,model.skips[k]) for k in model.skips if k!=j ]
                    print("newskips=",newskips)
                    model.skips=dict(newskips)
                    print("newskips=",model.skips)
                    deletions+=1
                    #we don't run full EM, but we do need to update full partitions and parameters
                    model.resetActive(range(model.kappa))
                    model.expectationStep()#update w
                    model.maximizationStep()
                    print ("evaluated model: F=",model.F)
                else: done=True
        print ("after deletions: skips=",model.skips)
        print ("returning with Fval=",model.F)
        return model,updates,deletions

    def samplePoints(self,nSamp=1500,target=None,actives=None):
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
        wCut=1.0/nSamp**2
        #Select all points relevant for active components
        idxs=[i for i in range(self.N) if sum([self.hatgamma[i,k] for k in actives])>wCut]
        if target is None:
            #num of points relevant for smallest component
            nt=min(sum( self.hatgamma[i,j]>wCut for i in range(self.N)) for j in range(self.kappa))
        else:
            nt=sum( self.hatgamma[i,target]>wCut for i in range(self.N))

        #ie want nSamp pts relev for comp j
        ntarg=int(nSamp*len(idxs)/(nt+1.0/nSamp))
        print ("Selecting min(", len(idxs),",",ntarg, ") points for split.")
        if len(idxs)>ntarg :
            idxs=np.random.choice(idxs,ntarg,replace=False)
        spoints=np.array([Y[i] for i in idxs])
        return spoints

    def draw_clusters(self):
        p=np.array(self.Y)
        clusters=[ [] for i in  range(self.kappa)]
        n=len(p)
        for i in range(n):
            x=np.random.rand()
            run=0
            #print "wi=",w[i]
            for j in range(self.kappa):
                run+=self.hatgamma[i,j]
                #print "j,x,run",j,x,run
                if(x<run):
                    ic=j
                    break
            #print "ic=",ic
            clusters[ic].append(p[i])
        return clusters

    def set_map(self):
        points=self.Y
        #rescale the points to a unit hypercube
        mins=np.amin(points,axis=0)
        self.mapmins=mins
        maxs=np.amax(points,axis=0)
        self.mapeps=1.0/self.N
        scales=(maxs-mins)*(1+2*self.mapeps)
        self.mapscales=scales
        print ("map scales=",scales)
        points=(points-mins)/scales+self.mapeps
        #print ("Scaled data line:",points[0])
        #points=points.tolist()
        
        #Optionally now apply (inverse) logistic map to soften edge boundaries
        #This will stretch the points near the edge. Since we have already
        #already forced at least one point to be near the edge, this could have
        #have the effect that the max/min points will be stretched to be large
        #outliers.  We try to prevent that by adjusting the edge farther out if
        #needed.  In particular we ensure that the closest point is no closer
        #than half the edge distance for the nth closest point.
        if not self.logistic: return
        nclose=1+int(0.01*len(points))
        self.mapepslo=np.zeros(self.D)
        self.mapepshi=np.zeros(self.D)
        for ix in range(self.D):
            print('ix minmax:',ix,min(points[:,ix]),max(points[:,ix]))
            vals=np.sort(points[:,ix])
            epslo=max([self.mapeps,vals[nclose]/2.0])
            epshi=max([self.mapeps,(1.0-vals[-(nclose+1)])/2.0])
            self.mapepslo[ix]=epslo
            self.mapepshi[ix]=epshi
            scale=(1-2*self.mapeps)/(1.0-epslo-epshi)
            print('  eps,epslo,epshi,scale:',self.mapeps,epslo,epshi,scale)
            points[:,ix]=epslo+(points[:,ix]-self.mapeps)/scale
            print(' minmax:',min(points[:,ix]),max(points[:,ix]))
        #now the logit map so that min/max points are roughly preserved
        #We specifically use x->x'=(1+ln(1/x-1)/ln(eps))/2
        twologeps=2.0*np.log(self.mapeps)
        for ix in range(self.D):
            points[:,ix]=0.5+np.log(1.0/points[:,ix]-1)/twologeps
            print(' ix,minmax:',ix,min(points[:,ix]),max(points[:,ix]))
        self.Y=points
        
    def unmap(self,xs,nologistic=False):
        #Here we put the data back into original form.
        for ix in range(self.D):
            if self.logistic and not nologistic: 
                #First apply logistic inverse-map x'->x=(1+eps^(2*x-1))^-1
                #inverse of: points[:,ix]=0.5+np.log(1.0/points[:,ix]-1)/twologeps
                #print(' 1.minmax:',min(xs[:,ix]),max(xs[:,ix]))
                xs[:,ix]=(1+self.mapeps**(2*xs[:,ix]-1))**-1
                #print(' 2.minmax:',min(xs[:,ix]),max(xs[:,ix]))


                #Then adjust the window edges back
                epslo=self.mapepslo[ix]
                epshi=self.mapepshi[ix]
                scale=(1-2*self.mapeps)/(1.0-epslo-epshi)
                #inverse of: points[:,ix]=epslo+(points[:,ix]-self.mapeps)/scale
                xs[:,ix]=self.mapeps+(xs[:,ix]-epslo)*scale
                #print(' 3.minmax:',min(xs[:,ix]),max(xs[:,ix]))

            #shift and scale back to input form
            scale=self.mapscales[ix]
            minx=self.mapmins[ix]
            #print('unmap: ix,minx,scale:',ix,minx,scale)
            #inverse of: points=(points-mins)/scales+self.mapeps
            xs[:,ix]=minx+scale*(xs[:,ix]-self.mapeps)
            #print(' 4.minmax:',min(xs[:,ix]),max(xs[:,ix]))
                            
        return xs            
        

    def ellipse(self,ic,ix,iy,siglev=2,N_thetas=60,unmap=False):
        cov=self.Vinv[ic,:,:]*((1+1.0/self.beta[ic])/max([1,self.nu[ic]-self.D-1]))
        elx,ely=ellipse(self.rho[ic],cov,ic,ix,iy,siglev,N_thetas)
        if unmap:
            #The transform mapping works on the full param space, so...
            els=np.zeros((N_thetas,self.D))
            els[:,ix]=elx
            els[:,iy]=ely
            els=self.unmap(els).T
            elx=els[ix];
            ely=els[iy]
        return  elx,ely
        
    def plot(self,clusters,ix=0,iy=1,parnames=None,backend=None,truths=None,unmap=True):
        #n=sum([len(c) for c in clusters])
        #ndown=n/1000
        ndown=1
        for j in range(self.kappa):
            p=np.array(clusters[j])
            #print p.shape
            #print p
            if(len(p)<1):continue
            c=np.array(self.rho[j])
            col=fakexkcdcolors[j%len(fakexkcdcolors)]
            ps=p[::ndown]
            if unmap: ps=self.unmap(ps)
            xs=ps[:,ix]
            ys=ps[:,iy]
            plt.scatter(xs,ys,color=col,marker=".",s=0.1)
            cs=c.reshape(1,-1)
            if unmap:cs=self.unmap(cs)
            cx=cs[:,ix]
            cy=cs[:,iy]
            plt.scatter(cx,cy,color='k',marker="*")
            plt.scatter(cx,cy,color=col,marker=".")
            elx,ely=self.ellipse(j,ix,iy,1.0,unmap=unmap)
            plt.plot(elx,ely,color='k')
            elx,ely=self.ellipse(j,ix,iy,0.98,unmap=unmap)
            plt.plot(elx,ely,color=col)
            #print('transforming min/max')
            #print('xminmax,yminmax:',min(xs),max(xs),min(ys),max(ys))
            minmax=np.array([[0.0]*self.D,[1.0]*self.D])
            minmax=self.unmap(minmax,True)
            #print('minmax',minmax)
            plt.xlim(minmax[:,0]) 
            plt.ylim(minmax[:,1]) 
            if(parnames is None):
                parnames=["p"+str(ip) for ip in range(self.D)]
            plt.xlabel(parnames[ix])
            plt.ylabel(parnames[iy])

        if(truths is not None):
            #print("have truths=",list(truths))
            for rho,cov in truths:
                #print("rho=",rho)
                #print("cov=",cov)
                c=np.array(rho)
                plt.scatter([c[ix]],[c[iy]],color='k',marker="+")
                elxs,elys=ellipse(rho,cov,j,ix,iy,1.0)
                plt.scatter(elxs,elys,color='k',marker=".",s=.1)
                
        if(backend is None or backend=="screen"):
            plt.show()
        else:
            backend.savefig()
            #print "savedfig"
            plt.clf()

fakexkcdcolors="purple,green,blue,pink,brown,red,lightblue,teal,orange,lightgreen,magenta,yellow,skyblue,grey,limegreen,violet,darkgreen,turquoise,darkblue,tan,cyan,forestgreen,maroon,olive,salmon,royalblue,navy,black,hotpink,palegreen,olive,seagreen,lime,indigo,lightpink".split(',')

def compute_gmmvb(points,kappa,backend=None):
    model=gmmvb(points,kappa)
    model.run_EM(backend=backend)
    return model

def compute_xgmmvb(points,backend=None,k_penalty=0.1,logistic=False):
    k=1
    #nsamp=5000
    x=np.array(points)
    model=gmmvb(x,k,k_penalty,logistic=logistic)
    #sx=np.array(model.samplePoints(x,nsamp))
    #profile.runctx("model.run_EM_MAP(x,backend=backend)",globals(),locals())
    #model.run_EM(sx,backend=backend)
    model.run_EM(backend=backend)
    oldFval=model.F
    count=0
    updates=[0]
    while(True):
        count+=1
        print ()
        print ("Beginning xgmmvb cycle",count)
        t0=time.time()
        if(count>1):model.computeF()
        oldFval=model.F
        oldkappa=model.kappa
        model,updates,deletions=model.improveStructure(updates,backend=backend)
        t1=time.time()
        print ("improve structure outcome:")
        print ("kappa=",model.kappa,"  updates=",updates,"deletions=",deletions)
        print ("skips=",model.skips)
        print ("\n   time =",t1-t0)
        print ("\n============\n")
        model.show()
        model.run_EM(backend=backend)
        model.computeF()
        print ("Fval/old=",model.F,oldFval)
        if(len(updates)+deletions==0):break
        print ("full model EM outcome:\n time =",time.time()-t1)
        print ()
        
    return model

