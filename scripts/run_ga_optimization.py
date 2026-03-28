import sys
import os
import cupy as cp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from signal_processing.radar_pipeline import RadarPipeline
from optimization.ga_optimizer import GeneticAlgorithmGPU, RadarFitnessEvaluator

def main():
    print("1. Generating Synthetic ADC Data for Optimization...")
    adc_gen = SyntheticADCGenerator()
    adc_data = adc_gen.generate_adc_data({
        'range': cp.array([30.0, 80.0, 150.0]),
        'velocity': cp.array([10.0, -5.0, 20.0]),
        'rcs': cp.array([5.0, 2.0, 10.0]) # dBsm
    })
    
    pipeline = RadarPipeline(num_samples=adc_gen.num_adc_samples)
    evaluator = RadarFitnessEvaluator(pipeline, adc_data)
    
    print("2. Initializing Genetic Algorithm on GPU...")
    # Genes: [Nt_R, Ng_R, Mt_D, Mg_D, PFA_exp]
    bounds = [
        [2, 16],   # Nt_R
        [1, 8],    # Ng_R
        [2, 16],   # Mt_D
        [1, 8],    # Mg_D
        [-10, -2]  # PFA_exp
    ]
    
    ga = GeneticAlgorithmGPU(population_size=100, 
                             num_genes=5, 
                             gene_bounds=bounds, 
                             mutation_rate=0.1)
                             
    num_generations = 20
    
    print("3. Starting Evolution process (Maximizing SNR metrics)...")
    for gen in range(num_generations):
        best_fitness, best_ind = ga.step(evaluator.evaluate)
        
        # Decode best genes
        Nt_R = max(1, int(cp.round(best_ind[0]).item()))
        Ng_R = max(1, int(cp.round(best_ind[1]).item()))
        Mt_D = max(1, int(cp.round(best_ind[2]).item()))
        Mg_D = max(1, int(cp.round(best_ind[3]).item()))
        pfa_exp = best_ind[4].item()
        
        print(f"Gen {gen:02d} | Best Fitness: {best_fitness.item():.4f} | "
              f"CFAR(train=({Nt_R},{Mt_D}), guard=({Ng_R},{Mg_D}), pfa=10^{pfa_exp:.2f})")
              
    print("Evolution complete. Use these optimal parameters for the CFAR pipeline!")

if __name__ == "__main__":
    main()
