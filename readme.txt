Adam Carcione
12/30/18

Files Included:
N_QueensGA.py
readme.txt

--N_QueensGA.py--
to run this file, execute:

python3 N_QueensGA.py <n> <populationSize> <numGenerations> <probMut> <probCross> <percentElite>
Example: N_QueensGA.py 8 50 1000 0.1 0.9 0.2

In order to run, my program needs a predetermined population size, number of generations to run, probability of an individual mutating, probability of two individuals crossing their genes, and percentage of the population that will be taken in by elitism every generation.

I then encode the intial population with random board configurations and evaluate their fitness using a fitness function. Fitness Proportional Selection is then used to selec the <popSize-numElite>/2 sets of parents. Then I perform crossover on the set of parents and run a probability on each set of parents based on <probCross> to determine whether or not to actually crossover the genes. If the probability fails, the two parents get copied into the next generation, but if it passes the parents cross their genes and create two new children which get added to the next generation. 

Finally, I perform one of 3 mutation methods on my population, running the same type of probability on each individual as I did in crossover. Inverse mutation can be selected as the mutation choice, which simply selects a sub-array of the individual and inverts it. Insert mutation can also be selected as the mutation choice, which simply selects two random items in the individual and inserts the second one next to the first one, shifting everything afterwards over one. Swap mutation is the final mutation method that can be used, which works by picking two random items in the individual, and swapping only those two values.

Once I have mutated (or not mutated) all of my individuals the loop restarts all over again, evaluating the fitness, taking the best of the population and copying them into the next generation (Elitism), selecting the parents for the next generation based on their fitness, performing crossover on all of the parents, and performing mutation on all of the individuals of the new generation.

The program runs until it finds a board configuration with 0 clashes amongst the queens, or if it has run for <numGen> generations.

Upon completion, my program outputs the position of the queens on the board, what the board configuration should look like, and a graph which shows the maximum, average, and best fitness per generation.
