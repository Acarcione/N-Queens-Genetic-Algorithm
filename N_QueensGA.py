import sys
import random
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches

def initialize(n, popSize):
    adam = []   #First Individual
    for i in range(n):
        adam.append(i)

    p = []      #Initial Population
    for j in range(popSize):
        temp = adam.copy()
        random.shuffle(temp)
        p.append(temp)
    
    return p

def fitness(population, n):
    fScores = []
    for i in range(len(population)):
        score = 0
        for j in range(len(population[i])-1): #Check Diagonal Up Right
            x = population[i][j]
            for k in range(int((n-1) - j)):
                if (population[i][j+k+1] == x-k-1):
                    #print("CLASH")
                    score += 1
        for j in range(len(population[i])-1): #Check Diagonal Up Left
            x = population[i][-1 - j]
            for k in range(int((n-1) - j)):
                if (population[i][-1 -j -k -1] == x-k-1):
                    #print("CLASH")
                    score += 1
        for j in range(len(population[i])-1): #Check Diagonal Bottom Right
            x = population[i][j]
            for k in range(int((n-1) - j)):
                if (population[i][j+k+1] == x+k+1):
                    #print("CLASH")
                    score += 1
        for j in range(len(population[i])-1): #Check Diagonal Bottom Left
            x = population[i][-1 - j]
            for k in range(int((n-1) - j)):
                if (population[i][-1 -j -k -1] == x+k+1):
                    #print("CLASH")
                    score += 1
        fScores.append(score)
    return fScores

def selection(popFit, popSize, numElite):
    totalFit = 0

    for i in range(len(popFit)):    #Find Iverse proportional slices
        fitVal = 1/popFit[i][1]
        newTup = (popFit[i][0], fitVal)
        popFit[i] = newTup

    totalFit = 0
    for i in range(len(popFit)):    #Final Total Fitness Value or normalized values
        totalFit += popFit[i][1]

    for i in range(len(popFit)):
        fitVal = popFit[i][1]
        newTup = (popFit[i][0], (fitVal/totalFit))
        popFit[i] = newTup   

    popFit = sorted(popFit, key = lambda x : x[1])  #Sort Population by Fitness

    accNorm = []
    normSum = 0
    for i in range(len(popFit)):    #Create list of the accumulate normalization values
        accNorm.append(normSum + popFit[i][1])
        normSum += popFit[i][1]

    parents = []
    indexArr = []
    intPop = int(popSize)
    for i in range(int(popSize-numElite/2)): 
        parentss = []
        for j in range(2):
            randNum = random.randint(0,100)/100
            for j in range(len(accNorm)):
                if (j == 0):
                    if (randNum < accNorm[j]):
                        parentss.append(popFit[j][0])
                else:
                    if (randNum < accNorm[j] and randNum > accNorm[j-1]):
                        parentss.append(popFit[j][0])
        parents.append(parentss)

    #Ensure every value in parents is a couple and not a single
    for i in range(len(parents)-1):
        if (len(parents[i]) == 0):
            parents.pop(i)
        if (len(parents[i]) != 2):
            parents[i].append(parents[i][0])

    return parents
    
def crossover(parents, popSize, pCross, numElite):
    newPop = []
    for i in range(int(len(parents)-numElite)):
        prob = random.randint(0,100)/100
        if (prob > pCross): #Dont Crossover
            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])
            newPop.append(parents[i][0])
            newPop.append(parents[i][1])
            pass
        elif (prob <= pCross):  #Perform Crossover
            portionArr = []
            child1 = []
            child2 = []
            childArr = [child1, child2]
            cut = []
            cut2 = []
            cutArr = [cut, cut2]

            acc = 0

            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])

            #print(len(parents[i]))
            parentsArr = [parents[i][0],parents[i][1]]   #Parents
            for j in range(2):  #One Iteration per parent
                portionInd = random.sample(range(0, len(parents[j][0])+1), 2)  #Indexes used for isolation crossover portion
                portionInd = sorted(portionInd) 
                portion = parentsArr[j][portionInd[0]:portionInd[1]]
                portionArr.append(portion)

                for k in range(len(parentsArr[j])):     #Check if every value from pX is in the portion, if it is:ignore, if not:append to cutArr
                    if (j == 0):   
                        if (parentsArr[1][k] not in portion):
                            cutArr[0].append(parentsArr[1][k])
                    if (j == 1):
                        if (parentsArr[0][k] not in portion):
                            cutArr[1].append(parentsArr[0][k])

                tempInd = 0
                tempInd2 = 0

                for l in range(len(parentsArr[j])):
                    if (l < portionInd[0]):
                        childArr[j].append(cutArr[j][l])
                        tempInd +=1
                    if (l >= portionInd[0] and l < portionInd[1]):
                        childArr[j].append(portionArr[j][tempInd2])
                        tempInd2 +=1
                    if (l >= portionInd[1]):
                        childArr[j].append(cutArr[j][tempInd])
                        tempInd +=1
            newPop.append(childArr[0])
            newPop.append(childArr[1])

    return newPop

def insMut(pop, pMut, numElite): #Takes a child after crossover
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))

            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            indAcc = val2
            while (pop[i][ind[0]+1] != val2): #While the number after the first index isnt the desired value
                temp = pop[i][ind[1]-1]
                pop[i][ind[1]-1] = pop[i][ind[1]]
                pop[i][ind[1]] = temp
                ind[1] -= 1

    return pop

def invMut(pop, pMut, numElite):
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            #print("Mutate")
            chrom = pop[i]
            #print(pop[i])
            ind = sorted(random.sample(range(0,len(pop[i])), 2))

            s1 = chrom[0:ind[0]]
            seg = chrom[ind[0]:ind[1]]
            s2 = chrom[ind[1]:]

            seg.reverse()
            newChrom = s1 + seg + s2
            pop[i] = newChrom
            #print(pop[i])
    return pop

def swapMut(pop, pMut, numElite): #Takes a child after crossover
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))


            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            temp = val1
            pop[i][ind[0]] = val2
            pop[i][ind[1]] = temp

    return pop

def printBoard(ind, n):
    board = []
    
    for i in range(n):
        x = []
        board.append(x)
        for j in range(n):
            board[i].append(".")

    for i in range(len(ind)):
        board[ind[i]][i] = "o"

    board = np.asarray(board)

    for i in range(len(board)):
        for j in range(len(board[i])):
            print(board[i][j], " ", end = "")
        print()

def main():
    n = int(sys.argv[1])
    popSize = int(sys.argv[2])
    numGen = int(sys.argv[3])
    pMut = float(sys.argv[4])
    pCross = float(sys.argv[5]) 
    pElite = float(sys.argv[6]) 

    numElite = popSize*pElite

    population = initialize(n, popSize)
    gen = 0

    done = False
    allFit = []
    avgFit = []
    minFit = []
    firstItFit = []
    xaxis = []
    solution = []
    for i in range(numGen):
        print("Generation:", gen+1)
        elite = []

        fitScores = fitness(population, n) 
        
        if (i == 0):    #Record fitness of first iteration
            firstItFit = fitScores
            firstItFit = np.asarray(firstItFit)

        popFitArr = []
        for i in range(len(fitScores)):     # Link the fitness scores to the index of their population
            tup = (population[i],fitScores[i])
            popFitArr.append(tup)
        
              

        ####### Used for Testing and Plotting ######
        allFit.append(max(fitScores))
        minFit.append(np.min(fitScores))
        a = fitScores
        a = np.asarray(a)
        avgFit.append(np.average(a))
        ############################################

        for i in range(len(popFitArr)):
            if popFitArr[i][1] == 0:
                solution = popFitArr[i][0]
                solFit = popFitArr [i][1]
                done = True

        if (done == True):
            gen +=1
            xaxis.append(gen)

            break  

        temporary = sorted(popFitArr, key = lambda x : x[1])  #Sort Population by Fitness
        for i in range(int(numElite)):
            elite.append(temporary[i][0])

        ''' Selection '''
        parents = selection(popFitArr, popSize, numElite)

        ''' Crossover '''
        population = crossover(parents, popSize, pCross, numElite)

        ''' Mutation '''
        #population = insMut(population, pMut, numElite)
        #population = invMut(population, pMut, numElite)
        population = swapMut(population, pMut, numElite)
        
        population += elite
        gen+=1
        xaxis.append(gen)


    print()
    print("First Generation Average Fitness:",  np.average(firstItFit))
    #print()
    print("Last Generation Average Fitness:", np.average(fitScores))
    print("Converged to Solution at Generation", gen+1)
    print(solution)
    print(solFit, "Clashes")
    printBoard(solution, n)

    ################### PLOT ####################
    Title = str(n) + " Queens Problem"

    allFit = np.asarray(allFit)
    avgFit = np.asarray(avgFit)
    minFit = np.asarray(minFit)

    plot.plot(xaxis, allFit, "b") #plot max fitness of each generation
    plot.plot(xaxis, avgFit, "g") #plot average fitness of each generation
    plot.plot(xaxis, minFit, "r") #plot min fitness of each generation
    plot.xlabel("Generation")
    plot.ylabel("Fitness (Distance)")

    blue_patch = mpatches.Patch(color = 'blue', label = "Worst Fitness")
    green_patch = mpatches.Patch(color = 'green', label = "Average Fitness")
    red_patch = mpatches.Patch(color = 'red', label = "Best Fitness")

    plot.legend(handles = [blue_patch, red_patch, green_patch])
    plot.suptitle(Title, fontsize = 16)
    plot.axis([1,gen,0, max(allFit) + 10])
    plot.show()
    #############################################

main()