import argparse
import os
import sys

from numba import cuda

cuda.current_context()

import cupy as cp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.radarscenes_loader import (
    RadarScenesLoader,
    frame_to_synthetic_points,
    resolve_radar_data_h5,
)
from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from optimization.ga_optimizer import GeneticAlgorithmGPU, RadarFitnessEvaluator
from signal_processing.radar_pipeline import RadarPipeline


def load_adc_from_radarscenes(root: str, sequence: int, max_targets: int):
    resolve_radar_data_h5(root, sequence)
    loader = RadarScenesLoader(root)
    adc_gen = SyntheticADCGenerator()
    for _ts, frame_rows in loader.iter_frames(sequence, max_frames=1):
        pts = frame_to_synthetic_points(
            frame_rows, max_targets=max_targets, velocity_field="vr_compensated"
        )
        if pts["range"] is None:
            continue
        cp_pts = {
            "range": cp.asarray(pts["range"], dtype=cp.float32),
            "velocity": cp.asarray(pts["velocity"], dtype=cp.float32),
            "rcs": cp.asarray(pts["rcs"], dtype=cp.float32),
        }
        return adc_gen.generate_adc_data(cp_pts, max_targets=max_targets)
    print("No radar frame found in sequence.")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="GA tuning of CA-CFAR on RadarScenes-derived IF")
    p.add_argument(
        "--radarscenes-root",
        default=os.environ.get("RADARSCENES_ROOT", ""),
        help="RadarScenes root (sequence_<N>/radar_data.h5)",
    )
    p.add_argument("--sequence", type=int, default=1)
    p.add_argument("--max-targets", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.radarscenes_root.strip()
    if not root:
        print("Set RADARSCENES_ROOT or --radarscenes-root. GA uses the first frame of the sequence.")
        sys.exit(1)

    print("Loading first RadarScenes frame → IF for optimization...")
    adc_data = load_adc_from_radarscenes(root, args.sequence, args.max_targets)

    adc_gen = SyntheticADCGenerator()
    pipeline = RadarPipeline(num_samples=adc_gen.num_adc_samples)
    evaluator = RadarFitnessEvaluator(pipeline, adc_data)

    print("Initializing Genetic Algorithm on GPU...")
    bounds = [[2, 16], [1, 8], [2, 16], [1, 8], [-10, -2]]
    ga = GeneticAlgorithmGPU(
        population_size=100, num_genes=5, gene_bounds=bounds, mutation_rate=0.1
    )
    num_generations = 20

    print("Evolution (maximizing SNR-style fitness)...")
    for gen in range(num_generations):
        best_fitness, best_ind = ga.step(evaluator.evaluate)
        Nt_R = max(1, int(cp.round(best_ind[0]).item()))
        Ng_R = max(1, int(cp.round(best_ind[1]).item()))
        Mt_D = max(1, int(cp.round(best_ind[2]).item()))
        Mg_D = max(1, int(cp.round(best_ind[3]).item()))
        pfa_exp = best_ind[4].item()
        print(
            f"Gen {gen:02d} | Best Fitness: {best_fitness.item():.4f} | "
            f"CFAR(train=({Nt_R},{Mt_D}), guard=({Ng_R},{Mg_D}), pfa=10^{pfa_exp:.2f})"
        )

    print("Evolution complete. Apply these CFAR parameters to run_radar_pipeline.py if desired.")


if __name__ == "__main__":
    main()
