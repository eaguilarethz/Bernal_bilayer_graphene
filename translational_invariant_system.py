#!/usr/bin/env python
# coding: utf-8
# In[ ]:  #PACKAGES
import sys
import matplotlib.pyplot as plt  
import numpy as np
import multiprocessing.pool
from matplotlib import cm
import time 
#hbar=6.582e-16 #eV*s

#CONSTANTS USED
t0=2.61 #eV
t1=0.361 #eV
t3=0.283 #eV
t4=0.138 #eV
delp=0.015  #eV
a=1     #we use kx*a and Ky*a, so a is already inckuded in the momentum value
v0=t0*np.sqrt(3)*a/2  #eV *nm, hbar cancels when multiplying with pi
v3=t3*np.sqrt(3)*a/2
v4=t4*np.sqrt(3)*a/2
coupled=0.0654e-3 #a*(e 1T d)/(2 hbar),, units for minimally coupled momenta
lambdaI=0.5e-3  #spin orbit coupling in eV
BohrMag=5.78838e-5  #eV/T
screening_lenght= 25  #nm
relative_permitivity= 17.5 #dimentionless
o=None
epsi=1e-12
Auc=np.sqrt(3)*0.246**2/2


#In[] #ARGUMENTS REQUIRED TO INITIALIZE THE CALCULATION
##NEED TO BE PASSED NEXT TO THE FILE NAME WHEN RUNNING THE CODE 

Iterations=int(sys.argv[1])  #100 is preferred
Nx=int(sys.argv[2])
Ny=int(sys.argv[3])   #5 is preferred
Field=float(sys.argv[4])  #0 to 2    #10 is preferred
D=float(sys.argv[5])/1000  #displacement previously used=50e-3  #eV
cutoff=float(sys.argv[6]) #0.1 is preferred value
Electrons=int(sys.argv[7])
Seed=int(sys.argv[9])
Name=str(sys.argv[10])

Area=Nx*Ny*Auc*(2*np.pi)**2*2/((2*cutoff)**2*np.sqrt(3))

#In[] #PART 1. COMPUTING SINGLE PARTICLE EIGENVECTORS AND EIGENVALUES
def Hoffdiag(kx,ky,t,B,theta):
    #kx and ky correspond to kx*a and ky*a, t=+/-1 corresponds to valley,B is given in teslas 
    #theta is given in degrees
    theta=theta*np.pi/180
    H = np.array(np.zeros((4,4)), dtype=complex)
    H[1,0]=v0*(t*(kx+coupled*B*np.sin(theta))+1j*(ky-coupled*B*np.cos(theta)))
    H[2,0]=-v4*(t*kx+1j*ky)
    H[3,0]=-v3*(t*kx-1j*ky) 
    H[2,1]=t1
    H[3,1]=-v4*(t*kx+1j*ky)
    H[3,2]=v0*(t*(kx-coupled*B*np.sin(theta))+1j*(ky+coupled*B*np.cos(theta)))
    return H

def Hdiag_axis(B,s,mu):
    #t=+/-1 corresponds to valley, #mu is givel in eV
    H = np.array(np.zeros((4,4)), dtype=complex)
    H[0,0]=D/2-mu+BohrMag*s*B
    H[1,1]=delp+D/2-mu+BohrMag*s*B
    H[2,2]=delp-D/2-mu+BohrMag*s*B  
    H[3,3]=-D/2-mu+BohrMag*s*B
    return H

def Hamiltonian_single_axis(kx,ky,s,t,B,mu,theta):
    H1=Hdiag_axis(B,s,mu)+Hoffdiag(kx,ky,t,B,theta)+Hoffdiag(kx,ky,t,B,theta).conj().T
    return np.linalg.eigh(H1)

def index2momentum(Index,N_X_or_Y,Cutoff):
    #N_X_or_Y is the total number of mesh points in direction X or direction Y
    spacing=2*Cutoff/(N_X_or_Y-1)
    momentum=(Index-N_X_or_Y/2+1/2)*spacing
    return momentum


def eig_val_vec_mesh_single_axis(Nx,Ny,B,mu,theta):
    #ALWAYS VALID
    #Nx=2*Npoints+1
    Emesh  = np.zeros((Nx,Ny,4), float)
    #indices are:(kx*c,ky*c,band)
    Umesh  = np.zeros((Nx,Ny,4,4), dtype=complex)
    #indices are:(kx*c,ky*c,component,band)
    
    for i in range(Nx):
        kx=index2momentum(i,Nx,cutoff)
        for j in range(Ny):
            ky=index2momentum(j,Ny,cutoff)
            for t in range(2):
                valley=-t*2+1
                for s in range(2):
                    spin=-s*2+1
                    vals,vecs = Hamiltonian_single_axis(kx,ky,spin,valley,B,mu,theta)#maybe exchange i,j so they are x,y
                    Emesh[i,j,s+2*t],Umesh[i,j,:,s+2*t]=vals[1],vecs[:,1]
    #Emesh[i,j,0] is valley 1 , spin 1
    #Emesh[i,j,1] is valley 1 , spin -1
    #Emesh[i,j,2] is valley -1 , spin 1
    #Emesh[i,j,3] is valley -1 , spin -1
    return Emesh,Umesh

#In[] #PART 2. COMPUTING FORM FACTORS AND INTERATION MATRIX
def V(kx1,ky1,kx2,ky2):
    #check units
    deltakx=(kx1-kx2)*2*cutoff/(Nx-1)
    deltaky=(ky1-ky2)*2*cutoff/(Ny-1)

    aq=np.asarray(np.sqrt((deltakx)**2+(deltaky)**2))
    #kx and ky are in reality akx and aky, we multiply by q by 1/a

    default_error_setting=np.geterr()
    np.seterr(invalid="ignore")
    result=np.where(aq<epsi,screening_lenght/0.246,np.tanh(aq*screening_lenght/0.246)/aq)
    np.seterr(**default_error_setting)

    result=9.04749*0.246*result/relative_permitivity
    return result  #eV *nm**2

def V_Hartree(kx1,ky1,kx2,ky2):
    #check units
    deltakx=(kx1-kx2)*2*cutoff/(Nx-1)
    deltaky=(ky1-ky2)*2*cutoff/(Ny-1)
    aq=np.asarray(np.sqrt((deltakx)**2+(deltaky)**2))

    #kx and ky are in reality akx and aky, we multiply q by 1/a

    default_error_setting=np.geterr()
    np.seterr(invalid="ignore")
    result=np.where(aq<epsi,0,np.tanh(aq*screening_lenght/0.246)/aq)
    np.seterr(**default_error_setting)

    result=9.04749*0.246*result/relative_permitivity
    return result 


def initialize_HF(Nx,Ny,B,mu,theta): 
    
    kx=np.arange(Nx)
    ky=np.arange(Ny)

    
    P_ref=1/2*np.einsum("xy,bB->xybB",np.ones((Nx,Ny)),np.eye(4),optimize=True)
    Emesh,Umesh=eig_val_vec_mesh_single_axis(Nx,Ny,B,mu,theta) 
    FormFactor=np.einsum("xyib,XYiB,bB->xyXYB",Umesh.conj(),Umesh,np.eye(4),optimize=True)
    Vint=np.einsum("xyXYB,xyXY->xyXYB",FormFactor,V(kx[:,o,o,o],ky[o,:,o,o],kx[o,o,:,o],ky[o,o,o,:]),optimize=True)
    Vint_Hartree=np.einsum("xyXYB,xyXY->xyXYB",FormFactor,V_Hartree(kx[:,o,o,o],ky[o,:,o,o],kx[o,o,:,o],ky[o,o,o,:]),optimize=True)
    H_single=np.einsum("xyb,bB->xybB",Emesh,np.eye(4),optimize=True)

    return P_ref,H_single,Vint,Vint_Hartree,FormFactor 

#In[] #PART 3. GENERATING RANDOM INITIAL PROJECTOR
def initial_projector(Nx,Ny,seed):
    np.random.seed(seed)
    P=np.random.rand(Nx,Ny,4,4)
    P=(P+np.transpose(P,axes=(0,1,3,2)))/2
    return P


#In[] #PART 4. SELF CONSISTENT HARTREE-FOCK SCHEME
def HF_scheme(number_of_electrons,Max_Iterations,Nx,Ny,proj_in,proj_ref,HS,Vint,Vint_Hartree,FormFactor):
    #we do not specify B since that value is alredy encoded in HS,VH and VF

    KX=np.einsum("k,Kb->kKb",np.arange(Nx),np.ones((Ny,4))).reshape(Nx*Ny*4).astype(int)
    KY=np.einsum("k,Kb->Kkb",np.arange(Ny),np.ones((Nx,4))).reshape(Nx*Ny*4).astype(int)
    BAND=np.einsum("b,kK->kKb",np.arange(4),np.ones((Nx,Ny))).reshape(Nx*Ny*4).astype(int)

    #P_in is the initial projector
    Projector=proj_in

    P_old=100*np.ones((Nx,Ny,4,4))
    Energy_old=10^10
    #spin_valley=np.zeros(4,dtype=complex)
    projector_norms=[]
    iteration_energies=[]
    iteration=0

    while True:

        Fock=np.einsum("zyxu,abzyu,zyabx->abux",Projector-proj_ref,Vint,FormFactor,optimize=True)
        ###########################CHECKING ENERGY DECREASE#################################

        if True:
            E_single=np.einsum("abcf,abcf->",HS,Projector)
            #E_hartree=1/2*np.einsum("abcf,abcf->",Projector-proj_ref,Hartree,optimize=True)
            E_fock=1/2*np.einsum("abcf,abcf->",Projector-proj_ref,Fock,optimize=True)
            Energy=E_single-E_fock/Area
            print("Energy after iteration {} is {}".format(iteration,Energy))

        ##########################SAVING ENERGY AND PROJECTOR NORM##########################
            
        projector_norms.append(np.sum(np.abs(Projector)))
        iteration_energies.append(Energy)
        #############COMPUTING DIFFERENCE IN ENERGY###########################
        diff_E=Energy_old-Energy
        #diff_P=np.abs(np.sum(np.abs(P_old))-np.sum(np.abs(Projector)))
        diff_P=np.abs(np.sum(np.abs(P_old)-np.abs(Projector)))
        #print("diff_E={},diff_P={}".format(diff_E,diff_P))
        diff_E=diff_E.real
        #diff_P=np.sum(np.abs(P_old-Projector))
        if iteration>3 and diff_E<0:
            projector_norms.append(np.sum(np.abs(P_old)))
            iteration_energies.append(Energy_old)
            Projector=P_old
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            print("diff_E<0")
            break
        if diff_E<1e-15 and diff_P<1e-15:
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            print("convergence reached")
            break
        if iteration>Max_Iterations:
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            print("brocken {} th iteration".format(Max_Iterations))
            break
        ########################CONSTUCTING NEW PROJECTOR##############################################

        HF_hamiltonian=HS-Fock/Area

        evals,evecs=np.linalg.eigh(HF_hamiltonian)
        P_new=np.zeros((Nx,Ny,4,4),dtype=complex)
        evals_list=evals.reshape(Nx*Ny*4)
        sorting_indices=np.argsort(evals_list)

        for j in range(number_of_electrons):
            kx=KX[sorting_indices[j]]
            ky=KY[sorting_indices[j]]
            band=BAND[sorting_indices[j]]

            single_projector=np.einsum("i,j->ij",evecs[kx,ky,:,band].conj(),evecs[kx,ky,:,band])
            P_new[kx,ky,:,:]+=single_projector

                
        #P_old=np.copy(Projector)
        P_old=np.copy(Projector)
        Energy_old=np.copy(Energy)
        Projector=P_new
        #spin_valley=np.diag(Projector)
        iteration+=1
    return evals,evecs,projector_norms,iteration_energies,Projector


"""  #EXAMPLE OF VALUES TO RUN THE CODE DIRECTLY ON THE CODE EDITOR
Iterations=100 #100 is preferred
Nx=20
Ny=21  #5 is preferred
Field=0  #0 to 2    #10 is preferred
D=50/1000  #displacement previously used=50e-3  #eV
cutoff=0.1
Electrons=100
Seed=3
Name="example"

Area=Nx*Ny*Auc*(2*np.pi)**2*2/((2*cutoff)**2*np.sqrt(3))
"""
#In[]  #RUNNING THE CALCULATIONS FOR AN SPECIFIC ELECTRON AND SEED
#t = time.time()
P_ref0,H_single0,Vint0,VHartree0,FormFactor0=initialize_HF(Nx,Ny,Field,0,0)
P_in0=initial_projector(Nx,Ny,Seed)

evals0,evecs0,projector_norms0,iteration_energies0,Projector0=HF_scheme(Electrons,Iterations,Nx,Ny,P_in0,P_ref0,H_single0,Vint0,VHartree0,FormFactor0)
np.save("mesh_{}x{}_momentum_diagonal_{}_electrons_{}th_seed_{}displacement_{}magnetic_field_{}.npy".format(Nx,Ny,Electrons,Seed,D,Field,Name),[evals0,1,evecs0,projector_norms0,iteration_energies0,Projector0])
#print("Total time of slow HF schem after {} iterations is {} s".format(Iterations,np.round_(time.time() - t, 3)))

# %%
