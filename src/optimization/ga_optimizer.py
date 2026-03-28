import cupy as cp
import numpy as np

class GeneticAlgorithmGPU:
    """
    A simple Genetic Algorithm implementation running entirely on the GPU via CuPy.
    """
    def __init__(self, 
                 population_size, 
                 num_genes, 
                 gene_bounds,
                 mutation_rate=0.1,
                 crossover_rate=0.8):
        
        self.pop_size = population_size
        self.num_genes = num_genes
        self.bounds = cp.asarray(gene_bounds) # Shape: (num_genes, 2) [min, max]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize population randomly within bounds
        self.population = self._initialize_population()
        self.fitness = cp.zeros(self.pop_size, dtype=cp.float32)
        
    def _initialize_population(self):
        # Generate normalized random values [0, 1]
        norm_pop = cp.random.rand(self.pop_size, self.num_genes)
        # Scale to bounds
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        mins = self.bounds[:, 0]
        
        return norm_pop * ranges + mins
        
    def evaluate(self, fitness_function):
        """
        Evaluates the fitness of each individual.
        fitness_function should accept a cupy array (pop_size, num_genes)
        and return a cupy array of fitness scores (pop_size,).
        """
        self.fitness = fitness_function(self.population)
        
    def select(self):
        """
        Roulette-wheel selection or tournament selection.
        We'll use a simple Tournament Selection (tournament size = 3) on the GPU.
        """
        tournaments = cp.random.randint(0, self.pop_size, size=(self.pop_size, 3))
        
        # Get fitness of selected individuals
        tourn_fitness = self.fitness[tournaments]
        
        # Find indices of winners (max fitness) in each tournament
        winners_idx = cp.argmax(tourn_fitness, axis=1)
        
        # Advanced indexing to get the actual population index of winners
        selected_indices = tournaments[cp.arange(self.pop_size), winners_idx]
        
        return self.population[selected_indices]
        
    def crossover(self, parents):
        """
        Single-point crossover.
        """
        children = parents.copy()
        
        # Select parents to cross
        cross_mask = cp.random.rand(self.pop_size // 2) < self.crossover_rate
        
        for i in range(self.pop_size // 2):
            if cross_mask[i]:
                parent1_idx = i * 2
                parent2_idx = i * 2 + 1
                
                # Crossover point
                pt = cp.random.randint(1, self.num_genes)
                
                # Swap genes
                temp = children[parent1_idx, pt:].copy()
                children[parent1_idx, pt:] = children[parent2_idx, pt:]
                children[parent2_idx, pt:] = temp
                
        return children
        
    def mutate(self, children):
        """
        Random uniform mutation within bounds.
        """
        mutation_mask = cp.random.rand(self.pop_size, self.num_genes) < self.mutation_rate
        
        # Generate random new genes for the mutated ones
        norm_mut = cp.random.rand(self.pop_size, self.num_genes)
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        mins = self.bounds[:, 0]
        new_genes = norm_mut * ranges + mins
        
        children = cp.where(mutation_mask, new_genes, children)
        return children

    def step(self, fitness_function):
        """
        Runs one generation of the GA.
        Returns the best fitness and best individual of the generation.
        """
        self.evaluate(fitness_function)
        
        best_idx = cp.argmax(self.fitness)
        best_fitness = self.fitness[best_idx]
        best_individual = self.population[best_idx].copy()
        
        # Selection
        parents = self.select()
        
        # Crossover
        children = self.crossover(parents)
        
        # Mutation
        self.population = self.mutate(children)
        
        # Elitism: keep best individual
        self.population[0] = best_individual
        
        return best_fitness, best_individual


class RadarFitnessEvaluator:
    """
    Evaluates fitness of a CA-CFAR parameter configuration.
    Genes: [N_train_R, N_guard_R, M_train_D, M_guard_D, PFA_exp]
    """
    def __init__(self, radar_pipeline, adc_data):
        self.pipeline = radar_pipeline
        # Precompute the Range-Doppler map once since it doesn't change per generation
        self.rd_map = self.pipeline.process_range_doppler(adc_data)
        
    def evaluate(self, population):
        """
        Vectorized evaluation (iterated per individual as CA-CFAR convolution is 2D)
        """
        pop_size = population.shape[0]
        fitness = cp.zeros(pop_size, dtype=cp.float32)
        
        for i in range(pop_size):
            genes = population[i]
            # Decode genes (round integers)
            Nt_R = max(1, int(cp.round(genes[0]).item()))
            Ng_R = max(1, int(cp.round(genes[1]).item()))
            Mt_D = max(1, int(cp.round(genes[2]).item()))
            Mg_D = max(1, int(cp.round(genes[3]).item()))
            
            pfa_exp = genes[4].item()
            pfa = 10.0 ** pfa_exp # e.g. 10^-5
            
            # Run CFAR
            detections, power_map, threshold = self.pipeline.ca_cfar_2d(
                self.rd_map,
                train_cells=(Nt_R, Mt_D),
                guard_cells=(Ng_R, Mg_D),
                pfa=pfa
            )
            
            # Fitness block: Maximize SNR of detections, penalize having too many / too few detections
            num_detections = cp.sum(detections)
            if num_detections == 0 or num_detections > 500: # too noisy or completely deaf
                fitness[i] = 1e-6
                continue
                
            # Signal power of targets
            sig_power = cp.sum(power_map[detections])
            # Noise power of background (non-targets)
            noise_power = cp.sum(power_map[~detections]) / cp.sum(~detections)
            
            snr_linear = sig_power / (noise_power + 1e-12)
            
            # Penalize huge number of detections as false alarms
            fitness[i] = snr_linear / cp.sqrt(num_detections)
            
        return fitness
