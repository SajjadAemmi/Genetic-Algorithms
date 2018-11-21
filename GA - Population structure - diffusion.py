import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def evaluate_all(P, C, σ, K):
    
    μ0, μ1, n = P.shape
    F = np.zeros((μ0, μ1))

    for r, row in enumerate(P):
        for c, X in enumerate(row):
            Σ = 0
            
            for i in range(len(C)):
                Σ += K[i] * np.exp(-np.linalg.norm(X - C[i, :]) ** 2 / σ[i])
            
            F[r, c] = Σ

    return F

#------------------------------------------------
    
def evaluate(X, C, σ, K):
    
    Σ = 0
            
    for i in range(len(C)):
        Σ += K[i] * np.exp(-np.linalg.norm(X - C[i, :]) ** 2 / σ[i])
    
    F = Σ

    return F

#------------------------------------------------

def crossover(P, Individual_Pos, Neighbor):
    
    μ0, μ1, n = P.shape
    
    Neighbor_Pos = np.zeros(2, dtype=np.int32)
    
    I = P[Individual_Pos[0]][Individual_Pos[1]]
    
    if(Neighbor == 'top'):
        Neighbor_Pos[0] = Individual_Pos[0] - 1
        Neighbor_Pos[1] = Individual_Pos[1]

    elif(Neighbor == 'bottom'):
        Neighbor_Pos[0] = Individual_Pos[0] + 1
        Neighbor_Pos[1] = Individual_Pos[1]
        
    elif(Neighbor == 'right'):
        Neighbor_Pos[0] = Individual_Pos[0]
        Neighbor_Pos[1] = Individual_Pos[1] + 1
        
    elif(Neighbor == 'left'):
        Neighbor_Pos[0] = Individual_Pos[0]
        Neighbor_Pos[1] = Individual_Pos[1] - 1
                
    if(Neighbor_Pos[0] == -1):
        Neighbor_Pos[0] = μ0 - 1
    elif(Neighbor_Pos[0] == μ0):
        Neighbor_Pos[0] = 0    

    if(Neighbor_Pos[1] == -1):
        Neighbor_Pos[1] = μ1 - 1    
    elif(Neighbor_Pos[1] == μ1):
        Neighbor_Pos[1] = 0    
    
    N = P[Neighbor_Pos[0]][Neighbor_Pos[1]]
    
    α = np.random.rand()
    C = α * I + (1 - α) * N
 
    return C

#------------------------------------------------

def mutate(Child):
    
    n = Child.shape
    mu, sigma = 0, 0.1 # mean and standard deviation

    Child = Child + np.random.normal(mu, sigma, n)

    return Child

#------------------------------------------------

n = 10 #Dimension
μ = [10, 10] #Number of people

C = np.array(np.random.rand(10, n), dtype='double') #center of local optima
σ = np.array(np.random.rand(10, 1), dtype='double') #var of local optima
K = np.array(np.random.rand(10, 1), dtype='double') #height of local optima
Neighbors = ['top', 'bottom', 'right', 'left']

max_gen = 1000 #Maximum generation
    
P = np.array(np.random.rand(μ[0], μ[1], n), dtype='double')

F = np.zeros((10, 10))

Fbest = np.zeros(max_gen) #Best Fitness
Fmean = np.zeros(max_gen) #Mean Fitness
Fworst = np.zeros(max_gen) #Worst Fitness 

fig, ax = plt.subplots()
AX = ax.imshow(F, extent=[0, 1, 0, 1])
cbar = fig.colorbar(AX)
cbar.set_clim(vmin=0,vmax=2)
cbar_ticks = np.linspace(0., 2., num=11, endpoint=True)
cbar.set_ticks(cbar_ticks) 
cbar.draw_all() 

for t in range(max_gen):

    F = evaluate_all(P, C, σ, K)
    Fbest[t] = np.amax(F)
    Fmean[t] = np.mean(F)
    Fworst[t] = np.amin(F)
    
    ax.imshow(F, extent=[0, 1, 0, 1])
    plt.title('time ' + str(t), fontdict=None, loc='center', pad=None)
    cbar.set_clim(vmin=Fworst[t], vmax=Fbest[t])
    cbar_ticks = np.linspace(Fworst[t], Fbest[t], num=11, endpoint=True)
    cbar.set_ticks(cbar_ticks) 
    cbar.draw_all() 
    plt.pause(0.01)
    ax.clear()
    
    Individual_Pos = np.random.randint(10, size=2)
    Individual = P[Individual_Pos[0]][Individual_Pos[1]]
    Neighbor = np.random.choice(Neighbors, 1)
    
    Child = crossover(P, Individual_Pos, Neighbor)
    Child = mutate(Child)
    
    Parent_Fitness = evaluate(Individual, C, σ, K)
    Child_Fitness = evaluate(Child, C, σ, K)
   
    if(Child_Fitness > Parent_Fitness):
        P[Individual_Pos[0]][Individual_Pos[1]] = Child.copy()

ax.imshow(F, extent=[0, 1, 0, 1])
plt.title('final', fontdict=None, loc='center', pad=None)
plt.show()

fig, ax = plt.subplots()
plt.plot(Fbest, 'g', label = "Best")
plt.plot(Fmean, 'b', label = "Mean")
plt.plot(Fworst, 'r', label = "Worst")
plt.legend(loc=0)
plt.xlabel('Generation')
plt.ylabel('Evaluation')
plt.show()