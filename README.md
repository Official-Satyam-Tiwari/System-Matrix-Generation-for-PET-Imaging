# Monte Carlo System Matrix Generation for PET/SPECT

This Jupyter Notebook (`Cleaned_code_system_matrix.ipynb`) implements an algorithm to generate a system matrix for tomographic imaging (e.g., PET or SPECT) from Monte Carlo simulation data stored in a ROOT file. The system matrix is a crucial component for iterative image reconstruction algorithms.

## Algorithm Overview

The process involves the following key steps:

1.  **Configuration:** Define parameters for file paths, event processing limits, image voxelization (dimensions, boundaries), and sinogram binning (number of angles, radial bins, ranges).
2.  **Data Loading:**
    *   Load specified branches (columns) from a ROOT TTree (`Coincidences` in this case) into a pandas DataFrame.
    *   Optimized to process a large number of events (`N_EVENTS_TO_PROCESS`).
3.  **Event Filtering:**
    *   Filter the loaded events to select "true" coincidence events (e.g., those with no Compton or Rayleigh scattering within the phantom).
4.  **Voxel Assignment (Source Position):**
    *   For each true event, its 3D source position (`sourcePosX1`, `sourcePosY1`, `sourcePosZ1`) is mapped to a discrete 1D voxel index within the defined image grid.
    *   This step is vectorized using CuPy for GPU acceleration to handle large datasets efficiently.
5.  **System Matrix Construction (Counts):**
    *   Initialize a list of sparse matrices (one for each projection angle `theta_idx`), where each matrix will store counts for `(s_idx, origin_voxel_id)`.
    *   Process true events in chunks:
        *   For each event, determine the sinogram bin (`theta_idx`, `s_idx`) based on its `sinogramTheta` and `sinogramS` values.
        *   Determine its `origin_voxel_id`.
        *   Increment the count in the corresponding `[s_idx, origin_voxel_id]` entry of the LIL matrix for that `theta_idx`.
        *   This step also utilizes CuPy for GPU-accelerated binning and sparse matrix construction (COO format per theta, then summed into CSR).
    *   Simultaneously, accumulate the total number of true events originating from each voxel, which is needed for normalization.
6.  **Finalize Counts Matrix:**
    *   After processing all events/chunks, the list of sparse matrices (one per `theta_idx`) is vertically stacked (`vstack`) to form a single sparse CSR matrix (`system_matrix_counts`). This matrix has dimensions `(N_Total_Sinogram_Bins x N_Total_Voxels)`.
7.  **System Matrix Normalization:**
    *   The `system_matrix_counts` is normalized to create the final system matrix `system_matrix_A`.
    *   Each element `A[i, j]` (representing the probability of an event from voxel `j` being detected in sinogram bin `i`) is calculated as:
        `A[i, j] = system_matrix_counts[i, j] / total_true_events_from_voxel[j]`
    *   This normalization is performed efficiently on the GPU using CuPy.
8.  **Save & Report:**
    *   The final normalized system matrix (`system_matrix_A`) is saved to a `.npz` file (e.g., `system_matrix_A.npz`).
    *   Statistics like shape, number of non-zero elements (NNZ), memory usage, and sparsity are reported.
9.  **Verification (Basic Sanity Checks):**
    *   Checks if all system matrix elements are non-negative.
    *   Checks if the sum of probabilities for each voxel column (total detection efficiency for that voxel) is less than or equal to 1.

## Dependencies

*   `uproot`: For reading data from ROOT files.
*   `numpy`: For numerical operations.
*   `pandas`: For data manipulation (primarily for initial data loading and `value_counts`).
*   `scipy`: For sparse matrix operations (LIL, CSR, vstack, save_npz).
*   `cupy`: For GPU-accelerated array operations and sparse matrices (`cupyx.scipy.sparse`). **A CUDA-enabled GPU and correctly configured CuPy installation are required for the GPU-accelerated parts.**
*   `matplotlib`: For plotting histograms (used for diagnostics).
*   `tqdm`: For progress bars (used in commented-out CPU-based loops).

## How to Use

1.  **Set Configuration Parameters:**
    *   Modify `FILE_PATH` and `TREE_NAME` to point to your ROOT file and TTree.
    *   Adjust `N_EVENTS_TO_PROCESS` based on your dataset size and computational resources.
    *   Define `VOXEL_PARAMS` (x, y, z mins/maxs, and number of voxels nx, ny, nz) to specify your image reconstruction grid.
    *   Define `SINOGRAM_PARAMS` (n_theta, n_s, and s_min, s_max) for your desired projection space binning.
2.  **Ensure Dependencies are Installed:**
    ```bash
    pip install uproot numpy pandas matplotlib scipy cupy-cudaXX # Replace XX with your CUDA version, e.g., cupy-cuda11x or cupy-cuda12x
    ```
3.  **Run the Notebook:** Execute the cells in the Jupyter Notebook sequentially.
4.  **Output:** The script will generate and save the system matrix as `system_matrix_A.npz` (or as defined in the save step). It will also print various statistics and progress messages.

## GPU Acceleration

This notebook has been updated to leverage CuPy for GPU acceleration in computationally intensive steps:
*   **Voxel Assignment:** Coordinate transformations, clipping, and index calculations are performed on the GPU.
*   **System Matrix Construction (Counts):** Binning into `theta_idx` and `s_idx`, and the creation of intermediate sparse matrices per theta angle are done on the GPU.
*   **System Matrix Normalization:** The element-wise division for normalization is performed on the GPU.

This significantly speeds up processing for large datasets compared to a purely CPU-based approach.

## Mathematical Basis

*   **System Matrix Element `A[i,j]`:** Represents the probability that an emission from voxel `j` is detected in sinogram bin `i`.
*   **Monte Carlo Estimation:** `A[i,j] ≈ (Number of detected true events in bin 'i' originating from voxel 'j') / (Total number of true events processed originating from voxel 'j')`
*   **Voxelization:** Discretizes continuous source positions into a 3D grid, then flattens to a 1D index.
*   **Sinogram Binning:** Discretizes continuous detection parameters (`sinogramTheta`, `sinogramS`) into a 2D grid, then flattens to a 1D index for matrix rows.

## Further Verification

The notebook includes basic sanity checks. For more thorough verification, consider implementing:

*   **Forward Projection Tests:** Create simple phantoms (e.g., point source, uniform cylinder), multiply by `system_matrix_A`, and visualize the resulting sinogram to see if it matches expectations.
*   **Backward Projection (Adjoint) Tests:** Create a simple sinogram (e.g., one active bin), multiply by `system_matrix_A.transpose()`, and visualize the back-projected image to see if it forms the expected Line-of-Response (LOR).
*   **Reconstruction Tests:** Use the generated system matrix in an iterative reconstruction algorithm (e.g., MLEM, OSEM) with known phantom data to evaluate image quality.

## Notes

*   The efficiency of the GPU-accelerated parts depends on the GPU's capabilities and the size of the data chunks being processed.
*   The definition of "true events" (filtering criteria) might need adjustment based on the specifics of your simulation data and imaging modality.
*   Ensure `VOXEL_PARAMS` and `SINOGRAM_PARAMS` are chosen appropriately to cover the region of interest and provide adequate sampling.
