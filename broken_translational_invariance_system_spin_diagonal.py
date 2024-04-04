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
a=1#0.246  #nm #we need to use know kx*a and Ky*a as units
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


Iterations=int(sys.argv[1])  
Nx=int(sys.argv[2])
Ny=int(sys.argv[3])   
Field=float(sys.argv[4])  #Teslas
cutoff=float(sys.argv[5]) 
D=float(sys.argv[5])      # meV
Electrons=int(sys.argv[6])
Seed=int(sys.argv[7])
Name=str(sys.argv[8])

D=D*1e-3 #Going from meV to eV

Area=Nx*Ny*Auc*(2*np.pi)**2*2/((2*cutoff)**2*np.sqrt(3))

#In[] #PART 1. COMPUTING SINGLE PARTICLE EIGENVECTORS AND EIGENVALUES

def Hoffdiag(kx,ky,t,B,theta):
    #kx and ky correspond to kx*a and ky*a, t=+/-1 corresponds to valley,B is given in teslas 
    #theta is given in degrees
    theta=theta*np.pi/180
    H = np.array(np.zeros((4,4)), dtype=complex)
    H[1,0]=v0*(t*(kx+coupled*B*np.sin(theta))+1j*(ky-coupled*B*np.cos(theta)))
    H[2,0]=-v4*(t*kx+1j*ky)
    H[3,0]=-v3*(t*kx-1j*ky)  #v3 *pi^dagger
    H[2,1]=t1
    H[3,1]=-v4*(t*kx+1j*ky)
    H[3,2]=v0*(t*(kx-coupled*B*np.sin(theta))+1j*(ky+coupled*B*np.cos(theta)))
    return H

def Hdiag_axis(B,s,mu):
    #t=+/-1 corresponds to valley, #mu is givel in eV
    H = np.array(np.zeros((4,4)), dtype=complex)
    H[0,0]=D/2-mu+BohrMag*s*B
    H[1,1]=delp+D/2-mu+BohrMag*s*B
    H[2,2]=delp-D/2-mu+BohrMag*s*B  #v3 *pi^dagger
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
                    Emesh[i,j,2*s+t],Umesh[i,j,:,2*s+t]=vals[1],vecs[:,1]
    #Emesh[i,j,0] is spin  1 , valley 1
    #Emesh[i,j,1] is spin  1 , valley -1
    #Emesh[i,j,2] is spin -1 , valley 1
    #Emesh[i,j,3] is spin -1 , valley -1
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

    #kx and ky are in reality akx and aky, we multiply by q by 1/a

    default_error_setting=np.geterr()
    np.seterr(invalid="ignore")
    result=np.where(aq<epsi,0,np.tanh(aq*screening_lenght/0.246)/aq)
    np.seterr(**default_error_setting)

    result=9.04749*0.246*result/relative_permitivity
    return result 


def deltatensor(Nx,Ny):
    #boolean data type to reduCe space in memory
    delta=np.zeros((Nx,Ny,Nx,Ny,Nx,Ny,Nx,Ny), dtype=np.bool_)#kx1,ky1,kx2,ky2,kx3,ky3,kx4,ky4
    for kx1 in range(Nx):
        for ky1 in range(Ny):
            for kx2 in range(Nx):
                for ky2 in range(Ny):
                    for kx3 in range(Nx):
                        for ky3 in range(Ny):
                            kx4conserved=kx1+kx2-kx3
                            ky4conserved=ky1+ky2-ky3
                            for kx4 in range(Nx):
                                for ky4 in range(Ny):
                                    if kx4==kx4conserved and ky4==ky4conserved:
                                        delta[kx1,ky1,kx2,ky2,kx3,ky3,kx4,ky4]=1                               
    return delta

def initialize_HF(Nx,Ny,B,mu,theta): 
    
    kx=np.arange(Nx)
    ky=np.arange(Ny)

    P_ref_single_spin=1/2*np.einsum("xX,yY,bB->xybXYB",np.eye(Nx),np.eye(Ny),np.eye(2),optimize=True)
    P_ref=np.zeros((Nx,Ny,2,Nx,Ny,2,2),dtype=complex)
    P_ref[:,:,:,:,:,:,0]=P_ref_single_spin
    P_ref[:,:,:,:,:,:,1]=P_ref_single_spin

    Emesh,Umesh=eig_val_vec_mesh_single_axis(Nx,Ny,B,mu,theta) 
    FormFactor_before=np.einsum("xyib,XYiB,bB->xyXYB",Umesh.conj(),Umesh,np.eye(4),optimize=True)
    FormFactor=np.zeros((Nx,Ny,Nx,Ny,2,2),dtype=complex)
    FormFactor[:,:,:,:,:,0]=FormFactor_before[:,:,:,:,:2]
    FormFactor[:,:,:,:,:,1]=FormFactor_before[:,:,:,:,2:]

    Vint=np.einsum("xyXYBs,xyXY->xyXYBs",FormFactor,V(kx[:,o,o,o],ky[o,:,o,o],kx[o,o,:,o],ky[o,o,o,:]),optimize=True)
    Vint_Hartree=np.einsum("xyXYBs,xyXY,Bb->xybXYBs",FormFactor,V(kx[:,o,o,o],ky[o,:,o,o],kx[o,o,:,o],ky[o,o,o,:]),np.eye(2),optimize=True)

    H_single_before=np.einsum("xyb,xX,yY,bB->xybXYB",Emesh,np.eye(Nx),np.eye(Ny),np.eye(4),optimize=True)
    H_single=np.zeros((Nx,Ny,2,Nx,Ny,2,2),dtype=complex)
    H_single[:,:,:,:,:,:,0]=H_single_before[:,:,:2,:,:,:2]
    H_single[:,:,:,:,:,:,1]=H_single_before[:,:,2:,:,:,2:]
    return P_ref,H_single,Vint,Vint_Hartree,FormFactor

#In[] #PART 3. GENERATING RANDOM INITIAL PROJECTOR
def projector_fixed_seed(Nx,Ny,seed):
    #random seed to initialize projector
    np.random.seed(seed)
    Pup=np.random.rand(Nx*Ny*2,Nx*Ny*2)
    Pup=(Pup+np.transpose(Pup,axes=(1,0)))/2
    Pup=Pup.reshape(Nx,Ny,2,Nx,Ny,2)

    np.random.seed(seed*73+1)
    Pdown=np.random.rand(Nx*Ny*2,Nx*Ny*2)
    Pdown=(Pdown+np.transpose(Pdown,axes=(1,0)))/2
    Pdown=Pdown.reshape(Nx,Ny,2,Nx,Ny,2)

    P=np.zeros((Nx,Ny,2,Nx,Ny,2,2),dtype=float)
    P[:,:,:,:,:,:,0]=Pup
    P[:,:,:,:,:,:,1]=Pdown
    return P

#In[] #PART 4. SELF CONSISTENT HARTREE-FOCK SCHEME
def HF_scheme(number__of_electrons,Max_Iterations,Nx,Ny,proj_in,proj_ref,HS,Vint,Vint_Hartree,Vform):
    #we do not specify B since that value is alredy encoded in HS,VH and VF

    Projector=proj_in
    P_old=100*np.ones((Nx,Ny,2,Nx,Ny,2,2)) #High initial values so that code is not interrupted in first iteration
    Energy_old=10^10                       #They will be replaced by the results of the first iteration

    projector_norms=[]
    iteration_energies=[]

    iteration=0
    while True:
        
        
        Hartree=np.einsum("zyuwvut,abcdefg,zywvut,abzydewv->abcdefg",Projector-proj_ref,Vint_Hartree,Vform,delta,optimize=True)
        Fock=np.einsum("zyfwvcg,abwvcg,zydefg,abzywvde->abcdefg",Projector-proj_ref,Vint,Vform,delta,optimize=True)
        
        if True:
            E_single=np.einsum("abcdefg,abcdefg->",HS,Projector)
            E_hartree=1/2*np.einsum("abcdefg,abcdefg->",Projector-proj_ref,Hartree,optimize=True)
            E_fock=1/2*np.einsum("abcdefg,abcdefg->",Projector-proj_ref,Fock,optimize=True)
            Energy=E_single+(E_hartree-E_fock)/Area
            print("Energy after iteration {} is {}".format(iteration,Energy))
        ##########################SAVING ENERGY AND PROJECTOR NORM##########################
            
        #saving energy and some info about the projector for each iteration
        projector_norms.append(np.sum(np.abs(Projector)))
        iteration_energies.append(Energy)

        #############COMPUTING DIFFERENCE IN ENERGY###########################
        diff_E=Energy_old-Energy
        #diff_P=np.abs(np.sum(np.abs(P_old))-np.sum(np.abs(Projector)))
        diff_P=np.abs(np.sum(np.abs(P_old)-np.abs(Projector)))
        print("diff_E={},diff_P={}".format(diff_E,diff_P))
        diff_E=diff_E.real
        #diff_P=np.sum(np.abs(P_old-Projector))
        if iteration>10 and diff_E<0:
            projector_norms.append(np.sum(np.abs(P_old)))
            iteration_energies.append(Energy_old)
            Projector=P_old
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("diff_E<0")
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            break
        if diff_E<1e-12 and diff_P<1e-12:
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("convergence reached")
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            break
        if iteration>Max_Iterations:
            print("After {} iterations the energy is {}".format(iteration,Energy))
            print("brocken {} th iteration".format(Max_Iterations))
            print("diff_E={},diff_P={}".format(diff_E,diff_P))
            break
        ########################CONSTUCTING NEW PROJECTOR##############################################
        HF_hamiltonian=HS+(Hartree-Fock)/Area
        HF_hamiltonian_up=HF_hamiltonian[:,:,:,:,:,:,0].reshape(Nx*Ny*2,Nx*Ny*2)
        HF_hamiltonian_down=HF_hamiltonian[:,:,:,:,:,:,1].reshape(Nx*Ny*2,Nx*Ny*2)
        
        evalsup,evecsup=np.linalg.eigh(HF_hamiltonian_up)
        evalsdown,evecsdown=np.linalg.eigh(HF_hamiltonian_down)

        evals=np.hstack((evalsup,evalsdown))
        evecs=np.hstack((evecsup,evecsdown))
        sorting_indices=np.argsort(evals)


        P_new=np.zeros((Nx,Ny,2,Nx,Ny,2,2),dtype=complex)
        P_new_up=np.zeros((Nx*Ny*2,Nx*Ny*2),dtype=complex)
        P_new_down=np.zeros((Nx*Ny*2,Nx*Ny*2),dtype=complex)
       
        for ind in range(number__of_electrons):
            index=sorting_indices[ind]
            if index<Nx*Ny*2:
                P_new_up+=np.einsum("i,j->ij",evecsup[:,index].conj(),evecsup[:,index])
            else:
                index=index-Nx*Ny*2
                P_new_down+=np.einsum("i,j->ij",evecsdown[:,index].conj(),evecsdown[:,index])

        P_new_up=P_new_up.reshape(Nx,Ny,2,Nx,Ny,2)
        P_new_down=P_new_down.reshape(Nx,Ny,2,Nx,Ny,2)
        P_new[:,:,:,:,:,:,0]=P_new_up
        P_new[:,:,:,:,:,:,1]=P_new_down

        P_old=np.copy(Projector)
        Energy_old=np.copy(Energy)
 

        Projector=P_new
        ########################OPTIMAL DAMPING##############################################
        if True:
            dP=P_new-P_old
            coef_1=np.einsum("abcdefg,abcdefg->",HS,dP,optimize=True)
            Hartree_dP=np.einsum("zyuwvut,abcdefg,zywvut,abzydewv->abcdefg",dP,Vint_Hartree,Vform,delta,optimize=True)
            Fock_dP=np.einsum("zyfwvcg,abwvcg,zydefg,abzywvde->abcdefg",dP,Vint,Vform,delta,optimize=True)
            
            H_dP=(Hartree_dP-Fock_dP)/Area
            coef_01=np.einsum("abcdefg,abcdefg->",H_dP,P_old-proj_ref,optimize=True)
            coef_11=np.einsum("abcdefg,abcdefg->",H_dP,dP,optimize=True)
            lin=coef_1+coef_01
            quad=0.5*coef_11

            print("lin={}".format(lin))
            print("quad={}".format(quad))

            lin=lin.real
            quad=quad.real            
            
            if lin<0 and quad>-lin/2:
                l=-lin/2/quad
                print("l=%.2e"%l)
            elif lin+quad<0:
                l=1
                print("l=%.2e"%l)   
            elif iteration<2:  ##Always allow first two iterations
                l=1
                print("l=%.2e"%l)
            else:
                l=1e-2
                print("l=%.2e"%l)
                print("close to convergence")
            Projector=(1-l)*P_old+l*P_new
            
        iteration+=1
    return evals,evecs,projector_norms,iteration_energies,Projector 


""" #EXAMPLE OF VALUES TO RUN THE CODE DIRECTLY ON THE CODE EDITOR
Iterations=10 
Nx=8 
Ny=7
Field=0 
cutoff=0.1 
Electrons=100
D=50e-3
Seed=1
Name="example"
Area=Nx*Ny*Auc*(2*np.pi)**2*2/((2*cutoff)**2*np.sqrt(3))
"""
#In[]   #RUNNING THE CALCULATIONS FOR AN SPECIFIC ELECTRON AND SEED
delta=deltatensor(Nx,Ny)
P_ref00,H_single00,Vint00,Vint_Hartree00,Vform00=initialize_HF(Nx,Ny,Field,0,0)
P_in00=projector_fixed_seed(Nx,Ny,Seed)

#t = time.time() 
evals0,evecs0,projector_norms0,iteration_energies0,Projector0=HF_scheme(Electrons,Iterations,Nx,Ny,P_in00,P_ref00,H_single00,Vint00,Vint_Hartree00,Vform00)
np.save("single_electron_{}_seed_{}_{}.npy".format(Electrons,Seed,Name),[evals0,1,evecs0,projector_norms0,iteration_energies0,Projector0])
#print("Total time of HF schem after {} iterations is {} s".format(Iterations,np.round_(time.time() - t, 3)))

# %%
