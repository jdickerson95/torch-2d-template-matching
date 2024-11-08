import torch_2d_template_matching.simulate as sim
import multiprocessing as mp

def main():
    # Get the number of CPU cores
    n_cores = mp.cpu_count()  # Or specify a number like 4
    print(f"Using {n_cores} cores")
    sim.simulate_3d_volume(
        pdb_filename="/Users/josh/git/2dtm_tests/simulator/6n8j.pdb",
        sim_volume_shape=(400, 400, 400),
        sim_pixel_spacing=1.0,
        n_cpu_cores=n_cores,  # Pass the number of cores to use
        gpu_ids=[-999],  # [-999] is cpu, None is auto most available memory, [0, 1, 2], etc is specific gpu
        num_gpus=0
    )

if __name__ == "__main__":
    # This guard is essential for multiprocessing
    main()
