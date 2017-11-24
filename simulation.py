from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import scipy.sparse as sp # pour la construction de matrice creuse
from math import *

############################################################################
############################################################################
#### Pour obtenir les _animations_ dans spyder il faut :                ####
#### Outils>Préférences>Console IPython>Graphiques>Sortie : Automatique ####
#### Puis fermer et réouvrir spyder pour appliquer le changement        ####
############################################################################
############################################################################


############################################################################
############################################################################
####### [*] deplacement pour visualiser eta                         ########
####### [*] vaisseau pour visualiser l'onde sur un demi-cylindre    ########
####### [*] pression pour visualiser la pression                    ########
#######    Seule une des trois variables peut-être égale à True     ########
############################################################################
############################################################################


deplacement = False
vaisseau = True #En 3D -- Vaisseau est imparfait (symétrie non complète) mais néanmoins intéressant sur la partie supérieure
pression = False #NB : échelle non constante mais délicat d'en trouver une qui convienne pour tout temps t

if (deplacement and vaisseau) or (deplacement and pression) or (vaisseau and pression):
    print("Choisir seulement une des trois animations")
else:
    
    ############################################################################
    ############################################################################
    ########                                                            ########
    ########                     Variables                              ########
    ########                                                            ########
    ############################################################################
    ############################################################################
    
    #Paramètres :
    rhos = 50 #Diverge pour des valeurs inférieures à environ 41
    L = 6 #Si l'on change la valeur de L il faut multiplier dx par la nouvelle valeur divisée par 6 pour garder le même nombre de points
    ylim = 0.12 #Bornes en ordonnée pour le graphe de eta
    dt = 0.001
    
    
    #Constantes du problèmes
    tau = 0.005 #temps d'exitation en secondes
    hs = 0.1
    rhof = 1
    E = 0.75*10**6
    p_zero = 2*10**4
    mu = 0.5
    R = 1
    a = E*hs/(R**2*(1-mu**2))
    
    
    #Paramètres de la discrétisation :
    dx = 0.1*L/6
    dy = 0.1
    Nx = int(L/dx)
    Ny = int(R/dy)
    T = 1000*dt
    Nt = int(T/dt)
    
    #Initialisation du déplacement :
    eta = np.zeros(shape=(Nt,Nx))
    
    
    ############################################################################
    ############################################################################
    ########                                                            ########
    ########                     Programmation                          ########
    ########                                                            ########
    ############################################################################
    ############################################################################
    
    
    def excitation(t): #L'excitation en gamma 1 f
        if (t < tau):
            return p_zero
        else:
            return 0
            
    def conditionLimite(t,n):#Renvoi le membre de droite r dans : Lap*p = r
        r = [0]*(Nx*Ny)
        
        #Conditions aux limites sur sigma
        if n > 2:
            j = Ny-1
            for i in range(1,Nx-1):
                r[indiceToK(i,j)] = -rhof*(eta[n][i]-2*eta[n-1][i]+eta[n-2][i])/(dt)**2
    
        for j in range(0,Ny):
            #Conditions aux limites sur gamma 1 f    
            r[indiceToK(0,j)] = excitation(t)
            
            #Conditions aux limites sur gamma 2 f    
            r[indiceToK(Nx-1,j)] = 0
            
        return r
               
    def indiceToK(i,j): #Permet d'effectuer le changement d'indice pour exprimer matriciellement notre système linéaire
        return i + (j)*Nx
    
    def laplacien2D():#Génère la matrice du système linéaire
        A = 2/(dx)**2 + 2/(dy)**2
        B = -1/(dx)**2
        C = -1/(dy)**2
        E = 1/dy
        F = 0.5*dy/(dx)**2
        print(Nx,Ny)
        Lap = sp.eye(Nx*Ny).tolil()
        
        #Conditions aux limites sur gamma 3 f :
        j = 0
        for i in range(1,Nx-1):
            k = indiceToK(i,j)
            Lap[k,k] = -E-2*F
            if i >= 2:
                Lap[k,indiceToK(i-1,j)] = +F
            if i <= Nx-2:
                Lap[k,indiceToK(i+1,j)] = +F
            Lap[k,indiceToK(i,1)] = E
        
        #Conditions aux limites sur sigma
        j = Ny-1
        for i in range(1,Nx-1):
            k = indiceToK(i,j)
            Lap[k,k] = E+2*F
            if i >= 2:
                Lap[k,indiceToK(i-1,j)] = -F
            if i <= Nx-2:
                Lap[k,indiceToK(i+1,j)] = -F
            Lap[k,indiceToK(i,Ny-2)] = -E
            
        #Le laplacien partout ailleurs
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                k = indiceToK(i,j)
                Lap[k,k] = A
                Lap[k,indiceToK(i-1,j)] = B
                Lap[k,indiceToK(i+1,j)] = B
                Lap[k,indiceToK(i,j-1)] = C  
                Lap[k,indiceToK(i,j+1)] = C
                    
        Lap = Lap.tocsr() #CSR préféré pour la résolution de système linéaire
    
        return Lap
    
    def reshapeP(p): #Pour remettre sous forme de rectangle la carte de la pression plutôt qu'un vecteur (à cause du changement d'indice)
        r = np.zeros(shape=(Nx,Ny))
        for i in range(0,Nx):
            for j in range(0,Ny):
                r[i,j] = p[indiceToK(i,j)]
        
        return r
                
        
    def graphP(p,t): #Trace le graph de la pression
        #print("Graphe de la pression au temps t = " + str(t) )
        plt.imshow(p, cmap='hot', interpolation='nearest')
        plt.suptitle("t = " + str(t))   
        plt.colorbar()
        plt.pause(0.001)
        plt.clf()
    
        
    def solveEta(m): #Résout eta
        #Initialisation
        p = np.zeros(shape=(Nx*Ny))
        CL = np.zeros(shape=(Nx*Ny))
        CL = conditionLimite(0,0)
        Lap = laplacien2D()    
        p = sp.linalg.spsolve(Lap,CL)
        pshape = reshapeP(p)
        
        if pression and deplacement == False and vaisseau == False:
            graphP(pshape,0)
        
        
        for t in range(1,m):#On itère sur le temps
        
            #Résolution de p au temps t
            CL = conditionLimite(t*dt,t)
            p = sp.linalg.spsolve(Lap,CL)
            
            
            pshape = reshapeP(p)
            if pression and deplacement == False and vaisseau == False:               
                graphP(pshape,t)
                
            #Déduction de eta grâce à la formule de récurrence linéaire avec second membre
            for i in range(1,Nx-1): 
                eta[t+1][i] = ((pshape[i][Ny-1]-a*eta[t][i])*((dt)**2)/(rhos*hs)+2*eta[t][i]-eta[t-1][i])
    
            #Pour l'affichage d'eta
            if deplacement and vaisseau == False and pression == False:           
                plt.plot(eta[t])
                plt.ylim(-ylim,ylim)
                plt.suptitle("t = " + str(t))
                plt.gcf()
                plt.show()
                plt.pause(0.001)
                plt.clf()
    
            
    
    
    #On résout jusqu'à Nt-1 (sinon ça déborde dans le tableau)
    solveEta(Nt-1)
    
    #Pour l'animation 3d du vaisseau sanguin
    if vaisseau and deplacement == False and pression == False:           
        Ng = 50
        plt.ion()
        
        def generate(t,Ng,Z):
            for j in range(0,Ng):
                for i in range(0,Nx):
                    Z[j,i] = sin(j/(Ng-1.0)*3.14)*(R/2.0+eta[t][i])
                
            return Z
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        xs = np.linspace(0, L, Nx)
        #ys = np.linspace(-R/2.0, R/2.0, Ng)
        ys = [0]*Ng
        for i in range(0,Ng):
            ys[i] =L/2.0*cos(i/(Ng-1.0)*3.14)
        X, Y = np.meshgrid(xs, ys)
        Z = np.sqrt(X**2 + Y**2)
    
        
        wframe = None
        for t in range(0,Nt):
        
            oldcol = wframe
        
            Z = generate(t,Ng,Z)
            wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
        
            if oldcol is not None:
                ax.collections.remove(oldcol)
            
            plt.suptitle("t = " + str(t))
            plt.pause(.001)
        
