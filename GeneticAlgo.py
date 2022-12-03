from collections import namedtuple
import random as rand
from random import random
import numpy as np
from random import choices, randint, randrange
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

Genome = List[int]
Population = List[Genome]

FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population,FitnessFunc], Tuple[Genome , Genome]]
CrossoverFunc=Callable[[Genome,Genome], Tuple[Genome,Genome]]
MutationFunc=Callable[[Genome],Genome]
alfa = 950 #for penalize the population


def generate_genome(length: int = 10) -> Genome: 
    #generate random genom with length of 10 in binary
    #make sure that genome is less than 100
    genome = [rand.randint(0,1) for _ in range(length)]
    while genome_to_quantity(genome) > 20:
        genome = [rand.randint(0,1) for _ in range(length)] 
    return genome 

def genome_to_quantity(genome: Genome) -> int:
    #convert genome to quantity
    quantity=0
    for i in range(len(genome)):
        quantity+=genome[i]*2**i
    return quantity  

def price(totalQuantity , demand: int= 1023) -> int:
    return demand - totalQuantity

def profit(quantity,cost,price) -> int:
    return quantity*price - quantity * cost

def generate_population(size : int = 10, genome_length: int = 10 ) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]    

def mutation(genome: Genome, num: int =1, probablity: float = 0.004) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probablity else abs(genome[index] - 1 )

    return genome

def single_point_crossover(a: Genome, b: Genome) -> Tuple [Genome ,Genome]:
    length = len(a)

    if length < 2:
        return a , b
   
    prob = rand.random()
    if prob < 0.6:
        return a,b

    p = rand.randint(1,length - 1)
    
    return a[0:p] + b[p:], b[0:p] + a[p:]

def calculate_price(population: Population) -> int:
    totalQuantity=0
    for gen in population:
        totalQuantity+=genome_to_quantity(gen)
    return price(totalQuantity)    

def fitness(population: Population,genome: Genome ) -> int:  ###calculate in different way
    price = calculate_price(population)   
    quantity = genome_to_quantity(genome)
    x=profit(quantity,0,price)
    return profit(quantity,0,price)
def check_total_quantity(population: Population):
    totalQuantity=0
    for gen in population:
        totalQuantity+=genome_to_quantity(gen)
    if totalQuantity > 1023:
        if(np.var([genome_to_quantity(gen) for gen in population]) == 0):
            while totalQuantity > alfa:
                for x in range(len(population)):
                    if totalQuantity > alfa:
                        totalQuantity-=genome_to_quantity(population[x])
                        quantity=genome_to_quantity(population[x])
                        if(quantity==1):
                            totalQuantity+=1
                            continue
                        quantity = quantity - 1
                        gen = [int(x) for x in list('{0:0b}'.format(quantity))]
                        gen.reverse()
                        while len(gen) < 10:
                            gen.append(0)
                        population[x]=gen
                        totalQuantity+=genome_to_quantity(population[x])
                    else:
                        break
        else:
            Quantities = [genome_to_quantity(genome) for genome in population]
            max1 = np.max(Quantities)
            index1 = Quantities.index(max1)
            population.pop(index1)
            population.append([0,1,0,0,0,0,0,0,0,0])
              
        return population
    else:
        return population
            

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    if np.sum([fitness_func(population,genome) for genome in population]) <=0 :
        #return minimum two genomes
        Quantities = [genome_to_quantity(genome) for genome in population]

        min1 = np.min(Quantities)
        #find the index of min1
        index1 = Quantities.index(min1)
        min2 = np.min([Quantities[x] for x in range(len(Quantities)) if x != index1])
        genome1 = [int(x) for x in list('{0:0b}'.format(min1))]
        genome2 = [int(x) for x in list('{0:0b}'.format(min2))]

        #reverse genome
        genome1.reverse()
        genome2.reverse()
        #make sure that genome is 10 bit
   
        while len(genome1) < 10:
            genome1.append(0)
        while len(genome2) < 10:
            genome2.append(0)
        return genome1,genome2
        
    else:   
        return choices(population=population,
        weights=[fitness_func(population,genome) for genome in population],
        k=2
        )   



def run_evolution(
    populate_func: PopulateFunc ,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 1000
) -> Tuple[Population, int]:
    population = populate_func()
    history_var = []
    history_totalQ = []
    for i in range(generation_limit):
        #calculate total quantity
        totalQuantity=0
        for gen in population:
            totalQuantity+=genome_to_quantity(gen)
        history_totalQ.append(totalQuantity)
        #calculate variance
        history_var.append(np.var([genome_to_quantity(gen) for gen in population]))
        population=sorted(
            population, 
            key= lambda genome: fitness_func(population,genome),
            reverse= True
        )
        if fitness_func(population,population[0]) >= fitness_limit:
            break

        #next_generation = population[0:2]
        next_generation = []
        for j in range(int(len(population)/2) ):
            parent = selection_func(population , fitness_func)
            offspring_a, offspring_b = crossover_func(parent[0],parent[1])
            offspring_a=mutation_func(offspring_a)
            offspring_b=mutation_func(offspring_b)
            next_generation +=[offspring_a,offspring_b]

        next_generation=check_total_quantity(next_generation)
        population = next_generation   

    population=sorted(
            population, 
            key= lambda genome: fitness_func(population,genome),
            reverse= True
        )
    return history_var,history_totalQ, population, i



history_var,history_totalQ , population, generations = run_evolution(
    populate_func=generate_population,
    fitness_func=fitness,
    fitness_limit=500000,
)
#plot variance 
plt.plot(history_var)
plt.xlabel('Generation')
plt.ylabel('Variance')
plt.show()
#plot total quantity
plt.plot(history_totalQ)
plt.xlabel('Generation')
plt.ylabel('Total Quantity')
plt.show()

print("Number of iterations", (generations+1))
for i in range(len(population)):
    print("Last Quantity",genome_to_quantity(population[i]))
    print("Last Profit",fitness(population,population[i]))
