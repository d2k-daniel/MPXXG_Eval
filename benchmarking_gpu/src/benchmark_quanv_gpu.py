import csv
import multiprocessing
import os
import time
from pathlib import Path

try:
    from .benchmark_qft_gpu import (
        BASELINE_MEASUREMENT_SECONDS,
        BASELINE_SETTLE_SECONDS,
        DEFAULT_MAX_QUBITS,
        DEFAULT_MIN_QUBITS,
        GPU_SAMPLE_INTERVAL_SECONDS,
        GPU_WARMUP_RUNS,
        GpuMetricsMonitor,
        MIN_MEASUREMENT_SECONDS,
        MIN_REPETITIONS,
        MIN_SUPPORTED_QUBITS,
        gpu_card_name_slug,
    )
except ImportError:
    from benchmark_qft_gpu import (
        BASELINE_MEASUREMENT_SECONDS,
        BASELINE_SETTLE_SECONDS,
        DEFAULT_MAX_QUBITS,
        DEFAULT_MIN_QUBITS,
        GPU_SAMPLE_INTERVAL_SECONDS,
        GPU_WARMUP_RUNS,
        GpuMetricsMonitor,
        MIN_MEASUREMENT_SECONDS,
        MIN_REPETITIONS,
        MIN_SUPPORTED_QUBITS,
        gpu_card_name_slug,
    )


NUM_QUANVOLUTIONALS = 19
RESULT_DIR = Path(__file__).resolve().parent / "result"

RESULT_FIELDS = [
    "Qubits",
    "Quanv",
    "Time_s",
    "PBMin_W",
    "PPeak_W",
    "PDiff_W",
    "MBMinMiB",
    "MPeakMiB",
    "MDiffMiB",
]


def quanv(
    num_qubits,
    function_name,
    is_transpiled=False,
    basis_gate_list=None,
):
    if basis_gate_list is None:
        basis_gate_list = ["h", "s", "cx", "rx", "ry", "rz"]

    qc = qiskit.QuantumCircuit(num_qubits)
    qc = getattr(ansatz, function_name)(qc)

    if is_transpiled:
        qc = qiskit.transpile(
            qc,
            basis_gates=basis_gate_list,
            optimization_level=3,
        )

    qc.save_statevector()
    return qc


def run_quanv_on_gpu(qc, simulator):
    start = time.perf_counter()
    simulator.run(qc, shots=1).result()
    return time.perf_counter() - start


def read_gpu_id():
    try:
        gpu_id = int(
            input(
                "Enter the physical GPU ID "
                "(0-based, as shown by nvidia-smi): "
            )
        )
    except ValueError:
        raise SystemExit("The GPU ID must be a whole number.")

    if gpu_id < 0:
        raise SystemExit("The GPU ID must be zero or greater.")

    return gpu_id


def read_qubit_range():
    min_qubits_input = input(
        f"Enter the minimum number of qubits "
        f"(at least {MIN_SUPPORTED_QUBITS}) "
        f"[default: {DEFAULT_MIN_QUBITS}]: "
    ).strip()
    max_qubits_input = input(
        "Enter the maximum number of qubits "
        f"[default: {DEFAULT_MAX_QUBITS}]: "
    ).strip()

    try:
        min_qubits = (
            int(min_qubits_input)
            if min_qubits_input
            else DEFAULT_MIN_QUBITS
        )
        max_qubits = (
            int(max_qubits_input)
            if max_qubits_input
            else DEFAULT_MAX_QUBITS
        )
    except ValueError:
        raise SystemExit("The minimum and maximum must be whole numbers.")

    if min_qubits < MIN_SUPPORTED_QUBITS:
        raise SystemExit(
            f"The minimum number of qubits must be at least "
            f"{MIN_SUPPORTED_QUBITS}."
        )
    if max_qubits < min_qubits:
        raise SystemExit(
            "The maximum number of qubits must be greater than or equal to "
            "the minimum."
        )

    return min_qubits, max_qubits


def read_output_file(is_transpiled, device_name):
    transpilation_name = "trans" if is_transpiled else "no_trans"
    default_filename = (
        f"quanv_gpu_{gpu_card_name_slug(device_name)}_"
        f"{transpilation_name}.csv"
    )
    filename = input(
        "Enter the single output CSV filename "
        f"(name only) [default: {default_filename}]: "
    ).strip()

    if not filename:
        filename = default_filename
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise SystemExit(
            "Enter only a filename, not a path or directory name."
        )
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    result_subdirectory = "Transpile" if is_transpiled else "No_Transpile"
    return RESULT_DIR / result_subdirectory / filename


def read_transpilation_option():
    value = input(
        "Use the transpiled quanvolutional circuits? [y/n]: "
    ).strip().lower()

    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False

    raise SystemExit(
        "Enter y/yes/true/1 for Transpile or "
        "n/no/false/0 for No_Transpile."
    )


def main():
    is_transpiled = read_transpilation_option()
    gpu_id = read_gpu_id()
    # CUDA maps the selected physical GPU to logical device 0 for Aer. This
    # must be set before importing Qiskit Aer or initializing CUDA.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    global qiskit, ansatz
    import qiskit
    from qiskit_aer import AerSimulator
    import ansatz

    min_qubits, max_qubits = read_qubit_range()

    available_devices = AerSimulator().available_devices()
    print(f"Available simulation devices: {available_devices}")
    if "GPU" not in available_devices:
        raise SystemExit(
            "Qiskit Aer GPU support is unavailable. "
            "Check the qiskit-aer-gpu installation and CUDA libraries."
        )

    try:
        metrics_monitor = GpuMetricsMonitor(
            gpu_id=gpu_id,
            sample_interval_s=GPU_SAMPLE_INTERVAL_SECONDS,
        )
    except Exception as error:
        raise SystemExit(
            f"Unable to initialize GPU metric monitoring: {error}"
        ) from error

    try:
        output_file = read_output_file(
            is_transpiled,
            metrics_monitor.device_name,
        )
    except BaseException:
        metrics_monitor.stop()
        raise

    output_file.parent.mkdir(parents=True, exist_ok=True)

    simulator = AerSimulator(
        method="statevector",
        device="GPU",
    )

    print("================== Benchmarking device: GPU ==================")
    print(
        "Quanvolutional circuit mode: "
        f"{'Transpile' if is_transpiled else 'No_Transpile'}"
    )
    print(f"Selected physical GPU ID: {gpu_id} ({metrics_monitor.device_name})")
    print(f"The single results file will be stored in: {output_file}")
    print(
        "GPU sampling mode: independent process, "
        f"every {GPU_SAMPLE_INTERVAL_SECONDS:.3f} seconds"
    )
    if metrics_monitor.power_sampling_source == "instantaneous":
        print("Power source: NVML instantaneous-power field")
    else:
        print(
            "Power source: NVML device-default reading "
            "(a one-second average on most Ampere/Ada-or-newer GPUs)"
        )
    print(
        "Power and memory are device-wide NVML measurements; "
        "keep the selected GPU otherwise idle for attributable results."
    )
    print(
        f"Each circuit uses at least {MIN_REPETITIONS} repetitions and "
        f"{MIN_MEASUREMENT_SECONDS:.1f} second of measurement."
    )

    with (
        metrics_monitor,
        output_file.open("w", newline="", encoding="utf-8") as csv_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        metrics_monitor.start()
        print(
            f"Waiting {BASELINE_SETTLE_SECONDS:.1f} second before the "
            "baseline measurement..."
        )
        time.sleep(BASELINE_SETTLE_SECONDS)
        baseline_interval = metrics_monitor.begin_interval()
        time.sleep(BASELINE_MEASUREMENT_SECONDS)
        baseline = metrics_monitor.end_interval(baseline_interval)
        print(
            "Baseline minimum: power "
            f"{baseline['minimum_power_w']:.3f} W; memory "
            f"{baseline['minimum_gpu_memory_used_mib']:.3f} MiB"
        )

        print(
            "Preparing quanvolutional circuits outside the measured "
            "intervals..."
        )
        prepared_quanv_circuits = {}
        for num_qubits in range(min_qubits, max_qubits + 1):
            for quanvolutional_number in range(
                1, NUM_QUANVOLUTIONALS + 1
            ):
                function_name = f"quanvolutional{quanvolutional_number}"
                prepared_quanv_circuits[
                    num_qubits, quanvolutional_number
                ] = quanv(
                    num_qubits,
                    function_name,
                    is_transpiled=is_transpiled,
                )

        for num_qubits in range(min_qubits, max_qubits + 1):
            print(f"=================> Benchmarking {num_qubits} qubits")

            for quanvolutional_number in range(
                1, NUM_QUANVOLUTIONALS + 1
            ):
                qc = prepared_quanv_circuits[
                    num_qubits, quanvolutional_number
                ]
                print(
                    f"Warming up quanvolutional{quanvolutional_number} "
                    f"with {GPU_WARMUP_RUNS} runs..."
                )
                for _ in range(GPU_WARMUP_RUNS):
                    run_quanv_on_gpu(qc, simulator)

                times = []
                quanv_interval = metrics_monitor.begin_interval()

                while (
                    len(times) < MIN_REPETITIONS
                    or time.perf_counter() - quanv_interval["start_time"]
                    < MIN_MEASUREMENT_SECONDS
                ):
                    times.append(run_quanv_on_gpu(qc, simulator))

                quanv_metrics = metrics_monitor.end_interval(quanv_interval)
                mean_time = sum(times) / len(times)
                power_difference_w = (
                    quanv_metrics["peak_power_w"]
                    - baseline["minimum_power_w"]
                )
                memory_difference_mib = (
                    quanv_metrics["peak_gpu_memory_used_mib"]
                    - baseline["minimum_gpu_memory_used_mib"]
                )

                writer.writerow(
                    {
                        "Qubits": num_qubits,
                        "Quanv": quanvolutional_number,
                        "Time_s": mean_time,
                        "PBMin_W": baseline["minimum_power_w"],
                        "PPeak_W": quanv_metrics["peak_power_w"],
                        "PDiff_W": power_difference_w,
                        "MBMinMiB": baseline[
                            "minimum_gpu_memory_used_mib"
                        ],
                        "MPeakMiB": quanv_metrics[
                            "peak_gpu_memory_used_mib"
                        ],
                        "MDiffMiB": memory_difference_mib,
                    }
                )
                csv_file.flush()

                print(
                    f"num_qubits: {num_qubits}, "
                    f"quanvolutional: {quanvolutional_number}, "
                    f"average execution time: {mean_time:.9f} seconds "
                    f"({len(times)} repetitions), "
                    f"power baseline-min/peak/difference: "
                    f"{baseline['minimum_power_w']:.3f}/"
                    f"{quanv_metrics['peak_power_w']:.3f}/"
                    f"{power_difference_w:.3f} W, "
                    f"memory baseline-min/peak/difference: "
                    f"{baseline['minimum_gpu_memory_used_mib']:.3f}/"
                    f"{quanv_metrics['peak_gpu_memory_used_mib']:.3f}/"
                    f"{memory_difference_mib:.3f} MiB"
                )

    print(f"Benchmark complete. Results saved to: {output_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
