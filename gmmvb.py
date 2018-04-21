#This file implements a Gaussian Mixture Model via Expectation-Maximation for clustering
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

negl_wt=0.001
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
    t3=-0.5*N*D*math.log(math.pi)
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
    for i in range(5):print("eta[",i,"]=",eta[i])
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
    def __init__(self,Y,kappa):
        self.Y=np.array(Y)
        self.N=len(Y)
        self.Y2=np.array([np.outer(self.Y[i,:],self.Y[i,:]) for i in range(self.N)])
        self.kappa=kappa
        self.D=len(Y[0])
        self.eta0=get_uninformative_eta0(self.kappa,self.D,self.Y)
        self.Ncomp=[None]*self.kappa
        self.gammabar = np.zeros((self.N,self.kappa))
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
        self.A,self.A_comp,self.A_Dval=compute_Aeta(self.D,self.nu,self.beta,self.lamb,self.logdetV)

        
        #randomly initialize rhos from data point
        self.rho=np.array([Y[i] for i in np.random.choice(self.N,self.kappa)])
        self.hatgamma=np.zeros((self.N,self.kappa))
        self.F=None
        self.Falt=None
        self.like=None
        self.EMtol=self.eta0[3]*.01
        self.needFalt=False
        
    def compute_derived_theta(self):
        #This operates over all components in vector form
        #note: eta = [ V^-1 + beta*rho*rho , nu - D , beta*rho , beta , lamb ]
        #could make this more efficient by only operating on active components
        assert self.updated_eta
        self.nu    = self.eta[1] + self.D
        self.beta  = self.eta[3].copy() 
        self.lamb  = self.eta[4].copy() 
        self.rho   = (self.eta[2].T/self.beta).T
        self.Vinv  = np.array([ self.eta[0][j] - self.beta[j]*np.outer(self.rho[j],self.rho[j]) for j in range(self.kappa )])
        self.Vnu   = np.array([ self.nu[j] * np.linalg.pinv( self.Vinv[j] ) for j in range(self.kappa )])
        self.logdetV  = np.array([ -np.linalg.slogdet( self.Vinv[j] )[1] for j in range(self.kappa) ])
        #print "shapes:eta0,1,2,3,4,rho,Vinv,Vnu",self.eta[0].shape,self.eta[1].shape,self.eta[2].shape,self.eta[3].shape,self.eta[4].shape,self.rho.shape,self.Vinv.shape,self.Vnu.shape
        #self.show()
        self.updated_theta=True
        self.updated_Aeta=False
        
    def show(self):
        print ("Variational Bayesian Gaussian mixture model: ",self.kappa,"components")
        #print "eta0:",self.eta[0]
        print ("eta1:",self.eta[1])
        print ("eta2:",self.eta[2])
        print ("eta3:",self.eta[3])
        print ("eta4:",self.eta[4])
        print ("th0    ",self.nu[0],self.beta[0],self.lamb[0],self.rho[0],self.logdetV[0])
        print ("th1    ",self.nu[1],self.beta[1],self.lamb[1],self.rho[1],self.logdetV[1])
        print ("th2    ",self.nu[2],self.beta[2],self.lamb[2],self.rho[2],self.logdetV[2])
        print ("th3    ",self.nu[3],self.beta[3],self.lamb[3],self.rho[3],self.logdetV[3])
        print ("th4    ",self.nu[4],self.beta[4],self.lamb[4],self.rho[4],self.logdetV[4])
        
    def A_D(self):
        #need self.lamb
        barlamb=sum(self.lamb)
        result=sum([scipy.special.gammaln(self.lamb[k]) for k in range(self.kappa)])
        result-= scipy.special.gammaln(barlamb)
        return result
    
    def A_NW(self,j):
        #need: derived theta:beta,nu,logdetV,log2pi
        #can evaluate only on active components
        #print ("ANW:",j,self.nu[j]*(self.logdetV[j]/2),self.nu[j]*(self.D*math.log(2.0)/2),lnGammaDhalf(self.nu[j],self.D),-self.D/2.0*math.log(self.beta[j]),self.D/2.0*log2pi)
        return self.nu[j]/2.0*(self.logdetV[j]+self.D*math.log(2.0))+lnGammaDhalf(self.nu[j],self.D)-self.D/2.0*math.log(self.beta[j])+self.D/2.0*log2pi

    def update_Aeta(self):
        #need: updated derived theta
        for j in self.activeComponents:
            self.A_comp[j]=self.A_NW(j)
        self.A_Dval=self.A_D()
        self.Aeta=sum(self.A_comp)+self.A_Dval
        #print ("A_D,A_eta[]",self.A_Dval,self.A_comp)
        #print ("update_Aeta",self.Aeta)
        self.updated_Aeta=True
        
    def computeF(self):
        #This computation of the objective function assumes that the model
        #parameters have just been updated (maximization step)
        #needvars:hatgamma,Aeta,A0
        #Eq 29 \ref{eq:computeF}
        assert self.updated_theta
        assert self.updated_Aeta
        N=self.N
        D=self.D
        val0  = -N*D/2.0*log2pi
        glogg= sum([self.hatgamma[i,j]*math.log(1e-300+self.hatgamma[i,j]) for j in range(self.kappa) for i in range(N)])
        if(False):
            for i in range(N):
                gvec=[self.hatgamma[i,j]*math.log(1e-300+self.hatgamma[i,j]) for j in range(self.kappa)]
                #print min(gvec)
                if(min(self.hatgamma[i])>1e-4):print ("glog:",min(self.hatgamma[i]),gvec,self.hatgamma[i,:],self.Y[i])
        val = val0-glogg
        #print ("Cth.  =",[self.A_comp[k] for k in range(self.kappa)])
        #print ("Cth.. =",[self.A_comp0[k] for k in range(self.kappa)])
        #print ("Cth...-",[self.Ncomp[k]*D*log2pi for k in range(self.kappa)])
        #print ("Cth=",[self.A_comp[k]-self.A_comp0[k]-self.Ncomp[k]*D*log2pi for k in range(self.kappa)])
        #print ("F:",-glogg,val0,"dAeta:",self.Aeta-self.A0,self.Aeta,self.A0,"DA_D:",self.A_Dval-self.A_Dval0,self.A_Dval,self.A_Dval0)
        
        val += self.Aeta - self.A0;
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
        return like + (  Dterm + Vterm + nuterm + betaterm ) 
    
    def resetActive(self,newActive):
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
        assert self.updated_theta
        if(not self.have_g):
            if(len(self.activeComponents)<self.kappa):
                self.g=[sum([self.hatgamma[i,j] for j in self.activeComponents]) for i in range(self.N)]
            else:
                self.g=[1.0]*self.N
            self.have_g=True
        #Pre-compute data-independent terms from Eq(24) from notes
        barlamb=sum(self.lamb)
        log_gamma0=scipy.special.digamma(self.lamb)-scipy.special.digamma(barlamb)-self.D*math.log(math.pi)/2
        for j in self.activeComponents:
            log_gamma0[j] += ( self.logdetV[j] + digammaDhalf(self.nu[j],self.D) - self.D/self.beta[j] ) / 2.0
        self.logLike=0
        #Now loop over the data for the rest of the terms and normalization
        for i in range(self.N):
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
            #print "shapes: ",self.hatgamma[:,j].shape,self.Y.shape,self.Y2.shape
            self.Ncomp[j]=np.sum(self.hatgamma[:,j])
            #print " y shape ",(self.hatgamma[:,j]*self.Y.T).T.shape
            yNcomp[j]=np.sum((self.hatgamma[:,j]*self.Y.T).T,0)
            #print " yy shape ",(self.hatgamma[:,j]*self.Y2.T).T.shape
            yyNcomp[j]=np.sum((self.hatgamma[:,j]*self.Y2.T).T,0)
            #print "->shapes: ",yNcomp[j].shape,yyNcomp[j].shape
            #print j,"eta[0]:",self.eta0[j]
            #print j,"yy:",yyNcomp[j]
            
        for j in self.activeComponents:
            #print "j,Ncomp,yNcomp,eigvals(yyNcomp-brr)",j,self.Ncomp[j],yNcomp[j],np.linalg.eigvalsh(yyNcomp[j]-np.outer(yNcomp[j],yNcomp[j])/self.Ncomp[j])
            self.eta[0][j]=self.eta0[0]+yyNcomp[j]
            self.eta[1][j]=self.eta0[1]+self.Ncomp[j]
            self.eta[2][j]=self.eta0[2]+yNcomp[j]
            self.eta[3][j]=self.eta0[3]+self.Ncomp[j]
            self.eta[4][j]=self.eta0[4]+self.Ncomp[j]
            #print "eta0=",self.eta[0][j]
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
        print("Trying projection")
        print("hatgamma shape=",self.hatgamma.shape)
        self.savehgamma=self.hatgamma
        alpha=0.75
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
        print("hatgamma shape=",self.hatgamma.shape)

        return True
                    
                

    def run_EM(self,backend=None,MaxSteps=800):
        Fval=-float('inf')
        #self.expectationStep()
        print ("active =",self.activeComponents)
        every=1
        Ncomp_old=np.array(self.Ncomp)
        dhgamma=None
        oldhgamma=self.hatgamma.copy()
        for count in range(MaxSteps):
            self.expectationStep()
            dhgammaP1=dhgamma
            if(oldhgamma is not None):dhgamma=self.hatgamma-oldhgamma
            if(self.needFalt):altFval=self.computeFalt()
            if(doProjection and dhgammaP1 is not None):
                self.maximizationStep()
                if self.project_partitions(dhgamma,dhgammaP1):
                    oldhgamma=None
                    dhgamma=None
                else:
                    oldhgamma=self.hatgamma.copy()
            else:
                self.maximizationStep()
                oldhgamma=self.hatgamma.copy()
            Ncomp=np.array(self.Ncomp)
            if(count>0):
                dNcomp = Ncomp-Ncomp_old
                dNcomp2=np.linalg.norm(dNcomp)**2
            else: dNcomp2=self.N**2
            
            #if self.activeComponents is not None:
            #    jlist=self.activeComponents
            #else:
            #    jlist=range(self.kappa)
            Fvalold=Fval
            self.update_Aeta()
            Fval=self.computeF()
            if(False and self.activeComponents is not None):
                for j in self.activeComponents: print (" mu[",j,"]=",self.mu[j])
            if(count%every==0):
                if(count>0):print (count,"dNcomp=",dNcomp2,Ncomp-Ncomp_old)
                #for k in self.activeComponents:print "rho",k,self.rho[k],self.Ncomp[k],self.logdetV[k],self.logdetV[k]+math.log((1+1.0/self.beta[k])/max([1,self.nu[k]-self.D-1]))*self.D,self.beta[k],"\n Sigma-eigvals",np.linalg.eigvalsh(self.Vinv[k]*((1+1.0/self.beta[k])/max([1,self.nu[k]-self.D-1])))
                if(self.needFalt):print (count,"altFval=",altFval)
                print (count,"Fval=",Fval)
                print (count,"Ncomp=",self.Ncomp)
                print (count,"lndet=",self.logdetV)
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
            print ("test:",Fval-Fvalold,"<",self.EMtol)
            #if(Fval-Fvalold<self.EMtol):
            #break
            if(dNcomp2<self.EMtol):
                break
        self.updated_gamma=True
        if(self.needFalt):altFval=self.computeFalt()
        self.F=Fval

        if(self.needFalt):print ("best Fval/alt=",Fval,altFval)
        else: print ("best Fval",Fval)
        for j in range(self.kappa):
            #print("inv(Vnu)=",np.linalg.pinv(self.Vnu[j]))
            logdetcov  = -np.linalg.slogdet( self.Vnu[j] )[1]
            logdetcov0  = -np.linalg.slogdet( self.Vnu0[j] )[1]
            print ("checkF(",j,"):",checkF(logdetcov,logdetcov0,self.D,self.Ncomp[j],self.nu0[j],self.beta0[j]))
            print ("Fcomp(",j,"):",self.A_comp[j]-self.A_comp0[j]-self.Ncomp[j]*self.D/2.0*log2pi)
            #print("coeff=",-1-logdetcov+self.D*math.log(2*math.pi))
            #print ("largeN: F->",self.Ncomp[j]/2*(-1-logdetcov+self.D*log2pi)-(self.D/4.)*(self.D+3)*math.log(self.Ncomp[j]))
                   
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
        self.hatgamma=np.zeros((self.N,self.kappa))
        self.A_comp = [None]*self.kappa
        self.activeComponents=actives.copy()
        #We don't call resetActive because, in this case g[i] specifically should not change because of the split
        self.update_Aeta()
        self.A0=self.Aeta
        self.update_Aeta()
        self.A_comp0=[self.A_comp[k] for k in range(self.kappa)]
        self.A_Dval0=self.A_Dval.copy()
        self.A0=self.Aeta.copy()
        #randomly initialize rhos from data point
        self.rho=np.array([Y[i] for i in np.random.choice(self.N,self.kappa)])
        self.hatgamma=np.zeros((self.N,self.kappa))
        self.F=None
        self.Falt=None
        self.like=None
        
#probably don't need this...?
    def update(self):
        #set lpost and BICevid
        #print "Evaluating model with "+str(self.k)+" components"
        self.expectationStep()
        self.maximizationStep()
        
        
    def improveStructure(self,backend=None):
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
        updatelist=np.random.permutation(self.kappa)
        model=copy.deepcopy(self)
        #model.evaluate(points)
        EMtol=self.EMtol
        tolRelax=10
        tolOverlap=0.03
        jcount=0
        for j in updatelist:
            jcount+=1
            print ("\nTrying split "+str(jcount)+"/"+str(len(updatelist)))
            #We first compute the overlap of this component with others
            #Use full points w for this
            #overs=[np.sum(w[:,j]*w[:,k]) for k in range(model.kappa)]
            overs=[model.componentOverlap(j,k) for k in range(model.kappa)]
            print("overlaps=",overs)
            overs=np.array(overs)/(sum(overs)+1e-10)
            active=[i for i in range(len(overs)) if overs[i]>tolOverlap]
            print (j,": overlaps =",overs)
            print ("-> active =",active)
            #Next we downselect to just the relevant points
            #jpoints=self.samplePoints(points,nSamp=nSamp,target=j,actives=active,w=w)
            #Work with a parallel test model (on the reduced set of points)
            testmodel=copy.deepcopy(self)
            testmodel.resetActive(active)
            testmodel.expectationStep()
            newmodel=copy.deepcopy(testmodel)
            newmodel.splitComponent(j)
            newmodel.EMtol=EMtol*tolRelax
            newmodel.run_EM_MAP()
            testmodel.EMtol=EMtol*tolRelax
            testmodel.run_EM_MAP()
            #  compare new BIC to original clustering BIC [wrt all points]
            #tol-2 cycle
            print ("testmodel=",testmodel.show())
            print ("testmodel.F=",testmodel.F)
            print ("newmodel=",newmodel.show())
            print ("newmodel.F=",newmodel.F)
            if(newmodel.F>testmodel.F and testmodel.F-newmodel.F<EMtol*tolRelax**2):#This is a close call continue EM with orig EMtol
                newmodel.EMtol=EMtol
                testmodel.EMtol=EMtol
            #Run a second time for "trimming" [resets relative weights]
            newmodel.run_EM()
            testmodel.run_EM()

            #newmodel.update(points)  
            newmodel.computeF()
            print ("actives=",newmodel.activeComponents)
            print ("Fvals orig/new",model.F,newmodel.F)
            print ("lposts orig/new",model.lpost,newmodel.lpost)
            if(newmodel.F>model.F):
                print ("Found improved model:")
                #newmodel.show()
                newmodel.EMtol=EMtol
                model=copy.deepcopy(newmodel)
                #we accept the newmodel as model
                #we don't run full EM, but we do need to update w and we
                #(do?) update the model params
                model.expectationStep()#update w
                model.maximizationStep()
                model.computeF()
                print ("evaluated model: F=",model.F)
                if(True):
                    phisum=np.sum(self.phi)
                    print (" [before] =",np.array2string(model.Ncomp,formatter={'float_kind':lambda x: "%.5f" % x}))
                    print ("lndetV=",model.lndetV)
                    print ("model.rho =\n",np.array2string(model.rho,formatter={'float_kind':lambda x: "%.3f" % x}))
                print ("model.phi=",np.array2string(model.Ncomp,formatter={'float_kind':lambda x: "%.5f" % x}))
                if(backend is not None):#do it all over for display purposes
                    #Need to rework this.....
                    newmodel=gmm(newmu,model.kappa,newsigma,newphi)
                    newmodel.EMtol=EMtol*tolRelax
                    newmodel.run_EM(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)
                    if(newmodel.F>model.F and model.F-newmodel.F<EMtol*tolRelax**2):newmodel.EMtol=EMtol
                    newmodel.run_EM(jpoints,activeComponents=actives,sloppyWeights=sloppyWeights,backend=backend)
                
        print ("returning with Fval=",model.F)
        return model

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

    def ellipse(self,ic,ix,iy,siglev=2,N_thetas=60):
        cov=self.Vinv[ic,:,:]*((1+1.0/self.beta[ic])/max([1,self.nu[ic]-self.D-1]))
        return ellipse(self.rho[ic],cov,ic,ix,iy,siglev,N_thetas)
        
    def plot(self,clusters,ix=0,iy=1,parnames=None,backend=None,truths=None):
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
            #print p.shape
            plt.scatter(p[::ndown,ix],p[::ndown,iy],color=col,marker=".",s=0.1)
            plt.scatter([c[ix]],[c[iy]],color='k',marker="*")
            plt.scatter([c[ix]],[c[iy]],color=col,marker=".")
            elxs,elys=self.ellipse(j,ix,iy,1.0)
            plt.plot(elxs,elys,color='k')
            elxs,elys=self.ellipse(j,ix,iy,0.98)
            plt.plot(elxs,elys,color=col)
            plt.xlim([0,1]) 
            plt.ylim([0,1]) 
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

def compute_xgmmvb(points,backend=None):
    k=1
    #nsamp=5000
    x=np.array(points)
    model=gmmvb(x,k)
    #sx=np.array(model.samplePoints(x,nsamp))
    #profile.runctx("model.run_EM_MAP(x,backend=backend)",globals(),locals())
    #model.run_EM(sx,backend=backend)
    #model.run_EM(backend=backend)
    oldFval=model.F
    count=0
    while(True):
        count+=1
        print ()
        print ("Beginning xgmm cycle",count)
        t0=time.time()
        model.computeF()
        oldFval=model.F
        oldkappa=model.kappa
        model=model.improveStructure(backend=backend)
        t1=time.time()
        print ("improve model time =",t1-t0)
        print ()
        print ("============")
        print (model.show())
        model.run_EM(backend=backend)
        model.computeF()
        print ("Fval/old=",model.F,oldFval)
        if(model.kappa==oldkappa):break
        print ("full model EM time =",time.time()-t1)
        print ()
        
    return model

