import torch_2d_template_matching.simulate as sim
import multiprocessing as mp

def main():
    # Get the number of CPU cores
    n_cores = mp.cpu_count()  # Or specify a number like 4
    
    sim.simulate_3d_volume(
        pdb_filename="/Users/josh/git/2dtm_tests/simulator/6n8j.pdb",
        sim_volume_shape=(400, 400, 400),
        sim_pixel_spacing=1.0,
        n_cpu_cores=n_cores,  # Pass the number of cores to use
        use_gpu=True  # Enable GPU acceleration
    )

if __name__ == "__main__":
    # This guard is essential for multiprocessing
    main()
