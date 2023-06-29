import numpy as np
from math import ceil, log10, factorial
import pandas as pd
from random import sample, randrange
from operator import attrgetter
import to_csv
import tqdm
import multiprocessing as mp
from copy import deepcopy
import matplotlib.pyplot as plt
import config

open('genomes_progress.csv', 'w').close()


def plot_convergence(fitness, fittest, generation, df):
    x = []
    x = np.quantile(fitness, [0.25, 0.75])
    x = np.append(fitness[fittest], x)
    x = np.insert(x, 0, generation)
    df.loc[len(df)] = x

    fig, ax = plt.subplots()
    x = df['gen']
    ax.plot(x, df['Best'], label='Fittest')
    ax.fill_between(
        x, df['25'], df['75'], color='b', alpha=.15, label='25-75 percentile')
    ax.set_ylim(ymin=0)
    ax.set_title('Convergence of GA')
    plt.legend()
    plt.savefig('Convergence.pdf', dpi=300)

    return fig, ax, df


class ship:
    def __init__(self, ship_id, name, departure_port, destination_port, load_type, vessel_type, costumer, agent, flag,
                 length, delta, eta, direction, tonnage, draft, dep, speed):
        self.name = name  # name of the ship
        self.departure_port = departure_port
        self.destination_port = destination_port
        self.load_type = load_type
        self.vessel_type = vessel_type
        self.costumer = costumer
        self.agent = agent
        self.flag = flag
        self.length = length  # length of the ship
        self.delta = delta  # estimated time to cross
        self.eta = eta  # estimated time of arrival  (expressed in minutes)
        self.direction = direction  # direction of the ship
        self.tonnage = tonnage
        self.draft = draft
        self.departure = dep  # the time that the ship leaves the canal (expressed in minutes)
        self.speed = speed
        self.ship_id = ship_id


# Class to represent a Vessel Scheduling Problem solution with GA
class GeneticAlgorithmVSP:
    def __init__(self, population_size=0, size=4, generations=0, mutationRate=0.1, elitismRate=0.1, dist_ratio=3,
                 plot_every_n_generations=5, gamma=0.96, early_stopping=False, patience=0, tol=1e-4):
        self.generations = generations
        self.population_size = population_size
        self.tournamentSize = size
        self.mutationRate = mutationRate
        self.elitismRate = elitismRate
        self.shipList = [];
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.dist_ratio = dist_ratio
        self.plot_every_n_generations = plot_every_n_generations
        self.gamma = gamma

    # ship arrives!
    def shipArrival(self, ship_id, name, departure_port, destination_port, load_type, vessel_type, costumer, agent,
                    flag, length, delta, eta, direction, tonnage, draft, dep, speed):
        self.shipList.append(
            ship(ship_id, name, departure_port, destination_port, load_type, vessel_type, costumer, agent, flag, length,
                 delta, eta, direction, tonnage, draft, dep, speed))

    # create initial population
    def makePopulation(self, q, shipList, my_set, population):
        i = 2
        while i < self.population_size:
            lista = []
            s = sample(shipList, len(shipList))
            lista = [j.ship_id for j in s]
            tup = tuple(lista)
            if tup not in my_set:
                population.append(deepcopy(s))
                my_set.add(tup)
                i += 1
        q.put(population)

    def injection(self, shipList, population):

        my_set = set()

        shipList.sort(key=attrgetter('eta', 'direction'))
        sorted_eta_direction = shipList
        population.append(deepcopy(sorted_eta_direction))

        shipList.sort(key=attrgetter('eta'))
        sorted_eta = shipList
        population.append(deepcopy(sorted_eta))

        lista1 = [j.ship_id for j in sorted_eta_direction]
        lista2 = [j.ship_id for j in sorted_eta]
        tup1 = tuple(lista1)
        if tup1 not in my_set:
            population.append(deepcopy(sorted_eta_direction))
            my_set.add(tup1)
        tup2 = tuple(lista2)
        if tup2 not in my_set:
            population.append(deepcopy(sorted_eta))
            my_set.add(tup2)

        return population, my_set

    # fitness function
    def computeFitness(self, population):
        # Compute cost for each permutation
        for permutation in population:
            total_cost = 0
            # find cost for all permutations
            for i in range(len(permutation)):
                delay = 0
                standard_estimated_time_to_cross = permutation[i].eta + permutation[i].delta
                if i > 0:
                    # case of two ships in different direction
                    if permutation[i - 1].direction != permutation[i].direction:
                        if (permutation[i - 1].departure > permutation[i].eta):
                            delay = permutation[i - 1].departure - permutation[i].eta
                            total_cost += delay
                    # case of two ships in same direction
                    else:
                        safe_distance = permutation[i].length * self.dist_ratio
                        time_for_safe_distance = safe_distance / (permutation[i].speed * config.KNOTS_TO_MIN_PER_SEC)

                        if (permutation[i - 1].eta <= permutation[i].eta):
                            dt = permutation[i].eta - (permutation[i - 1].departure - permutation[i - 1].delta)
                            distance_a = (permutation[i - 1].speed * config.KNOTS_TO_MIN_PER_SEC) * dt - permutation[
                                i - 1].length

                            if not (
                            (distance_a >= safe_distance or (permutation[i - 1].departure < permutation[i].eta))):
                                delay = (safe_distance - distance_a) / (
                                            permutation[i].speed * config.KNOTS_TO_MIN_PER_SEC)
                                total_cost += delay
                        else:
                            delay = permutation[i - 1].eta - permutation[i].eta + time_for_safe_distance
                            total_cost += permutation[i - 1].eta - permutation[i].eta + time_for_safe_distance
                permutation[i].departure = standard_estimated_time_to_cross + delay
            yield (total_cost)

    def TournamentSelection(self, population):
        tournament_contestants = [population[randrange(self.population_size)] for _ in range(self.tournamentSize)]
        tournament_contestants_fitness = list(self.computeFitness(tournament_contestants))

        return tournament_contestants[np.argmin(tournament_contestants_fitness)]

    def mutate(self, genome):
        if np.random.random() >= self.mutationRate:
            return genome

        ix_low, ix_high = self.__computeLowHighIndices(genome)
        genome[ix_low], genome[ix_high] = genome[ix_high], genome[ix_low]
        return genome

    def __computeLowHighIndices(self, allele):
        index_low = np.random.randint(0, len(allele) - 1)
        index_high = np.random.randint(index_low + 1, len(allele))
        while index_high - index_low > ceil(len(allele) / 2):
            try:
                index_low = np.random.randint(0, len(allele) - 1)
                index_high = np.random.randint(index_low + 1, len(allele))
            except ValueError:
                pass
        return (index_low, index_high)

    def crossover(self, parent1, parent2):
        offspring = [None for _ in range(len(self.shipList))]
        index_low, index_high = self.__computeLowHighIndices(parent1)

        offspring[index_low:index_high + 1] = parent1[index_low:index_high + 1]
        offspring_available_index = list(range(0, index_low)) + list(range(index_high + 1, len(self.shipList)))

        offspring_checksums = set([i.ship_id for i in offspring if i is not None])
        for allele in parent2:
            if None not in offspring:
                break

            if allele.ship_id not in offspring_checksums:
                offspring[offspring_available_index.pop(0)] = allele

        assert None not in offspring, ValueError(
            f'Error in Crossover. There are less vessels than before... {offspring}')
        return offspring

    def converged(self, population):
        return len(set([str(i) for i in population])) == 1

    def _create_new_allele(self, population):
        parent1 = self.TournamentSelection(population)
        parent2 = self.TournamentSelection(population)
        offspring = self.crossover(parent1, parent2)
        offspring = self.mutate(offspring)
        return offspring

    def optimize(self, shipList):
        if (self.population_size > factorial(len(shipList))):
            self.population_size = factorial(len(shipList))

        patience_counter, best_score = 0, np.inf
        q = mp.Queue()

        population1 = []

        population1, my_set = self.injection(shipList, population1)
        p1 = mp.Process(target=self.makePopulation, args=(q, shipList, my_set, population1))
        p1.start()
        #p1 = self.makePopulation(q,shipList,my_set,population1)
        #p1.start()
        print("w")

        population = q.get()
        # population = list(self.makePopulation(shipList))
        elitismOffset = ceil(self.population_size * self.elitismRate)

        if (elitismOffset > self.population_size):
            raise ValueError('Elitism Rate must be in [0,1].')

        # q2=mp.Queue()
        # p2= mp.Process(target=self.computeFitness_parallel, args=(q2,population))
        p1.join()
        # p2.start()
        # fitness= q2.get()
        fitness = list(self.computeFitness(population))
        fittest = np.argmin(fitness)

        check = True

        for generation in tqdm.tqdm(range(1, self.generations + 1)):

            newPopulation = []

            if elitismOffset:
                elites = np.array(fitness).argsort()[:elitismOffset]
                [newPopulation.append(population[i]) for i in elites]

            with mp.Pool(processes=mp.cpu_count() - 2) as pool:
                newPopulation.extend(
                    pool.map(self._create_new_allele, [population for _ in range(elitismOffset, self.population_size)]))

            population = newPopulation

            fitness = list(self.computeFitness(population))
            fittest = np.argmin(fitness)

            if check:
                df = pd.DataFrame(columns=['gen', 'Best', '25', '75'])
                check = False

            # if generation % self.plot_every_n_generations == 0 :
            #    x,y,df= plot_convergence(fitness, fittest, generation,df)

            if (best_score - fitness[fittest]) > self.tol:
                best_score = fitness[fittest]
                patience_counter = 0
            else:
                patience_counter += 1

            if self.converged(population) or (
                    patience_counter > self.patience and self.early_stopping or fitness[fittest] == 0):
                print(f'Early Stopping at Generation: {generation}')
                break

            self.mutationRate = self.mutationRate * self.gamma  # Decrease mutation rate per generation
            # self.elitismRate = self.elitismRate * (1 / self.gamma)  # Increase elitism rate per generation
            print(
                f'Generation {generation} | Cost: {np.around(fitness[fittest], 4)} ({to_csv.convert_to_time(fitness[fittest])}) | Mutation Rate: {self.mutationRate}')

        return (population[fittest], fitness[fittest])


def make_frame(filename, cols_to_read):
    data = pd.read_excel(filename, usecols=cols_to_read)
    return data


def scheduling():
    data = make_frame(config.data_file, config.data_columns)

    grouped = data.groupby(data.columns[0])
    # for name, group in grouped:
    group = grouped.get_group(config.date)
    dataList = group.values.tolist()
    year = dataList[0][0].year
    month = dataList[0][0].month
    day = dataList[0][0].day

    limit = round(640 * log10(len(dataList)) + 5)
    population_size = limit
    tournament_size = limit // 5

    ga_vsp = GeneticAlgorithmVSP(
        population_size, tournament_size, **config.kwargs
    )
    for ship_id, i in enumerate(range(len(dataList)), start=1):
        name = dataList[i][1]
        departure_port = dataList[i][2]
        destination_port = dataList[i][3]
        load = dataList[i][4]
        vessel_type = dataList[i][5]
        costumer = dataList[i][6]
        agent = dataList[i][7]
        flag = dataList[i][8]
        tonnage = dataList[i][10]
        length = dataList[i][11]
        draft = dataList[i][12]
        direction = dataList[i][9]
        eta = (int(dataList[i][13][0] + dataList[i][13][1])) * 60 + (int(dataList[i][13][3] + dataList[i][13][4]))
        departure = 0
        delta = (length + 6346) / (config.speed * 0.514) / 60
        ga_vsp.shipArrival(ship_id, name, departure_port, destination_port, load, vessel_type, costumer, agent, flag,
                           length, delta, eta, direction, tonnage, draft, departure, config.speed)
    permutation, cost = ga_vsp.optimize(ga_vsp.shipList)
    to_csv.create_csv(year, month, day, permutation, config.folder_with_results, cost, config.write_columns)

    print(to_csv.convert_to_time(cost))
