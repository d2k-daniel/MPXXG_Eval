# MPXXG Evaluation

This repository contains benchmark scripts and collected results for evaluating quantum-circuit workloads on GPUs and FPGAs. The workloads include the Quantum Fourier Transform (QFT), 19 quanvolutional circuit ansatzes, and TA results in the FPGA dataset.

The GPU scripts run Qiskit Aer statevector simulations and record execution time, NVIDIA GPU power, and GPU memory use. The FPGA directory currently contains precomputed text results; FPGA source, build, and execution scripts are not included.

## Repository layout

```text
MPXXG_Eval/
|-- benchmarking_gpu/
|   `-- src/
|       |-- benchmark_qft_gpu.py       # QFT GPU benchmark
|       |-- benchmark_quanv_gpu.py     # Quanvolutional GPU benchmark
|       |-- ansatz.py                  # 19 quanvolutional circuit definitions
|       `-- result/
|           |-- No_Transpile/          # Collected GPU results
|           `-- Transpile/
`-- hardware_eval/
    `-- fpga/c/results/
        |-- execution_time/             # FPGA timing results
        `-- power/                      # FPGA power results
```

## GPU requirements

- Python 3.10 or newer
- An NVIDIA CUDA-capable GPU
- A working NVIDIA driver and CUDA 11.2 or newer
- A CUDA-enabled Qiskit Aer installation
- Python packages: `numpy`, `qiskit`, and `nvidia-ml-py`

The prebuilt Qiskit Aer GPU packages are available for x86-64 Linux. Other platforms require a source build; see the [Qiskit Aer installation guide](https://qiskit.github.io/qiskit-aer/getting_started.html). Create an environment and install the package that matches the CUDA version on the machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy qiskit nvidia-ml-py

# CUDA 12
python -m pip install qiskit-aer-gpu

# For CUDA 11, use this package instead:
# python -m pip install qiskit-aer-gpu-cu11
```

Confirm that the driver can see the intended GPU before running a benchmark:

```bash
nvidia-smi
```

## Running the GPU benchmarks

Run the scripts from `benchmarking_gpu/src` so the local `ansatz.py` module is available:

```bash
cd benchmarking_gpu/src

# Quantum Fourier Transform
python benchmark_qft_gpu.py

# All 19 quanvolutional circuits
python benchmark_quanv_gpu.py
```

Each script interactively asks for:

1. Whether to use the transpiled circuit form.
2. The physical GPU index shown by `nvidia-smi`.
3. The inclusive qubit range. The default is 3 through 30.
4. The output CSV filename.

Results are written to `benchmarking_gpu/src/result/Transpile` or `benchmarking_gpu/src/result/No_Transpile`. An existing file with the chosen name is overwritten, so use a new filename when preserving an earlier run.

For QFT, the `Transpile` option selects a manually decomposed construction for controlled-phase and swap operations. For the quanvolutional workload, it runs `qiskit.transpile` at optimization level 3 with the `h`, `s`, `cx`, `rx`, `ry`, and `rz` basis gates.

## Measurement method

Both GPU scripts use the same procedure:

- Run a Qiskit Aer statevector simulation on the selected GPU with one shot.
- Build circuits before the measured intervals.
- Wait 1 second, then measure a 3-second idle baseline.
- Perform 9 warm-up runs for each circuit.
- Measure at least 100 repetitions and at least 1 second per circuit.
- Sample power and device memory through NVML every 20 ms in a separate process.
- Write each result row immediately after its measurement.

Power and memory values are device-wide NVML measurements. Keep the selected GPU otherwise idle if the measurements need to be attributable to the benchmark. The quanvolutional circuits use randomly generated parameters without a fixed seed, so exact circuit parameters differ between runs.

## GPU result format

QFT result files contain one row per qubit count. Quanvolutional result files contain one row for each `(qubit count, ansatz)` pair.

| Column | Meaning |
| --- | --- |
| `Qubits` | Number of simulated qubits |
| `Quanv` | Quanvolutional ansatz number, 1-19; present only in quanvolutional results |
| `Time_s` | Mean execution time per simulator run, in seconds |
| `PBMin_W` | Minimum power observed during the idle baseline, in watts |
| `PPeak_W` | Peak power observed during the measured circuit interval, in watts |
| `PDiff_W` | `PPeak_W - PBMin_W` |
| `MBMinMiB` | Minimum GPU memory use during the idle baseline, in MiB |
| `MPeakMiB` | Peak GPU memory use during the measured circuit interval, in MiB |
| `MDiffMiB` | `MPeakMiB - MBMinMiB` |

The checked-in GPU dataset contains QFT and quanvolutional results, with and without transpilation, for GPU labels `3090`, `4090`, `A40`, `A100`, and `A100x`.

## FPGA results

The FPGA dataset is stored below `hardware_eval/fpga/c/results`:

- `execution_time/Transpile` and `execution_time/No_Transpile` contain QFT and quanvolutional timing logs.
- `execution_time/QEA`, `execution_time/HPQEA_CCPE`, and `execution_time/HPQEA_CANDAR` contain QFT, quanvolutional, and TA timing logs.
- `execution_time/TA` contains the standalone TA timing log.
- `power/Transpile` and `power/No_Transpile` contain QFT and quanvolutional power logs.

Timing logs report values in nanoseconds, with seconds also shown where present. Power logs report watts. These files are raw text rather than the CSV schema used by the GPU scripts.

## Reference

The quanvolutional ansatz implementation cites [arXiv:1905.10876](https://arxiv.org/abs/1905.10876).

## License

No license file is currently included in this repository.
