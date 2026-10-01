import csv
import multiprocessing
import os
import queue
import re
import time
import traceback
from pathlib import Path

try:
    import pynvml
except ImportError:
    pynvml = None

MIN_REPETITIONS = 100
GPU_WARMUP_RUNS = 9
MIN_SUPPORTED_QUBITS = 3
DEFAULT_MIN_QUBITS = 3
DEFAULT_MAX_QUBITS = 30
GPU_SAMPLE_INTERVAL_SECONDS = 0.02
BASELINE_SETTLE_SECONDS = 1.0
BASELINE_MEASUREMENT_SECONDS = 3.0
MIN_MEASUREMENT_SECONDS = 1.0
MONITOR_CONTROL_TIMEOUT_SECONDS = 30.0
RESULT_DIR = Path(__file__).resolve().parent / "result"

RESULT_FIELDS = [
    "Qubits",
    "Time_s",
    "PBMin_W",
    "PPeak_W",
    "PDiff_W",
    "MBMinMiB",
    "MPeakMiB",
    "MDiffMiB",
]


def _read_power_w(handle):
    """Prefer NVML's instantaneous-power field when the stack supports it."""
    try:
        field_id = pynvml.NVML_FI_DEV_POWER_INSTANT
        field_value = pynvml.nvmlDeviceGetFieldValues(handle, [field_id])[0]
        if (
            field_value.nvmlReturn == pynvml.NVML_SUCCESS
            and field_value.valueType
            == pynvml.NVML_VALUE_TYPE_UNSIGNED_INT
        ):
            return field_value.value.uiVal / 1000.0, "instantaneous"
    except (AttributeError, pynvml.NVMLError):
        pass

    # On most Ampere/Ada-or-newer GPUs this legacy call is a one-second
    # average, so it is a compatibility fallback rather than the first choice.
    return (
        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "device_default",
    )


def _read_gpu_metrics(handle):
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "power_w": _read_power_w(handle)[0],
        "gpu_memory_used_mib": memory.used / (1024 ** 2),
    }


def _gpu_metrics_worker(
    gpu_id,
    sample_interval_s,
    command_queue,
    output_queue,
):
    """Sample NVML outside the benchmark process and its Python GIL."""
    nvml_initialized = False

    try:
        if pynvml is None:
            raise RuntimeError(
                "GPU monitoring requires nvidia-ml-py. Install it with "
                "'python -m pip install nvidia-ml-py'."
            )

        pynvml.nvmlInit()
        nvml_initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        next_sample_time = time.perf_counter()
        output_queue.put(("ready",))

        running = True
        while running:
            wait_s = max(0.0, next_sample_time - time.perf_counter())
            try:
                command = command_queue.get(timeout=wait_s)
            except queue.Empty:
                sampled_at = time.perf_counter()
                metrics = _read_gpu_metrics(handle)
                # This queue preserves ordering from this single producer, so
                # a later barrier acknowledgement guarantees receipt of every
                # preceding sample.
                output_queue.put(("sample", sampled_at, metrics))

                next_sample_time += sample_interval_s
                if next_sample_time <= sampled_at:
                    next_sample_time = sampled_at + sample_interval_s
                continue

            command_name = command[0]
            if command_name == "barrier":
                _, token = command
                output_queue.put(("ack", token))
            elif command_name == "stop":
                _, token = command
                output_queue.put(("stopped", token))
                running = False
            else:
                raise RuntimeError(
                    f"Unknown GPU monitor command: {command_name}"
                )
    except Exception as error:
        output_queue.put(
            ("error", f"{type(error).__name__}: {error}", traceback.format_exc())
        )
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()


class GpuMetricsMonitor:
    def __init__(self, gpu_id, sample_interval_s):
        if pynvml is None:
            raise RuntimeError(
                "GPU metric monitoring requires nvidia-ml-py. Install it with "
                "'python -m pip install nvidia-ml-py'."
            )

        self.gpu_id = gpu_id
        self.sample_interval_s = sample_interval_s
        self._samples = []
        self._sampling_error = None
        self._nvml_initialized = False
        self._process = None
        self._command_queue = None
        self._output_queue = None
        self._next_token = 0
        self._started = False

        pynvml.nvmlInit()
        self._nvml_initialized = True

        device_count = pynvml.nvmlDeviceGetCount()
        if gpu_id < 0 or gpu_id >= device_count:
            self._shutdown_nvml()
            raise RuntimeError(
                f"GPU ID {gpu_id} is invalid; NVML found {device_count} GPU(s)."
            )

        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.device_name = pynvml.nvmlDeviceGetName(self._handle)
        if isinstance(self.device_name, bytes):
            self.device_name = self.device_name.decode("utf-8", errors="replace")

        try:
            _read_gpu_metrics(self._handle)
            _, self.power_sampling_source = _read_power_w(self._handle)
        except Exception:
            self._shutdown_nvml()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.stop()
        except Exception:
            if exception is None:
                raise
        return False

    def _shutdown_nvml(self):
        if self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False

    def _read_gpu_metrics(self):
        return _read_gpu_metrics(self._handle)

    def _new_token(self):
        self._next_token += 1
        return self._next_token

    def _wait_for_message(self, expected_name, expected_token=None):
        deadline = time.monotonic() + MONITOR_CONTROL_TIMEOUT_SECONDS
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                exit_code = (
                    self._process.exitcode if self._process is not None else None
                )
                raise RuntimeError(
                    "Timed out waiting for the GPU monitor process "
                    f"({expected_name}); exit code: {exit_code}."
                )

            try:
                message = self._output_queue.get(timeout=remaining_s)
            except queue.Empty as error:
                exit_code = (
                    self._process.exitcode if self._process is not None else None
                )
                raise RuntimeError(
                    "Timed out waiting for the GPU monitor process "
                    f"({expected_name}); exit code: {exit_code}."
                ) from error

            message_name = message[0]
            if message_name == "sample":
                _, sampled_at, metrics = message
                self._samples.append((sampled_at, metrics))
                continue
            if message_name == "error":
                _, error_summary, error_traceback = message
                self._sampling_error = error_summary
                raise RuntimeError(
                    "GPU metric sampling failed in the monitor process:\n"
                    f"{error_traceback}"
                )
            if message_name != expected_name:
                raise RuntimeError(
                    "GPU monitor returned an unexpected message: "
                    f"{message_name!r}; expected {expected_name!r}."
                )
            if expected_token is not None and message[1] != expected_token:
                raise RuntimeError(
                    "GPU monitor returned an unexpected acknowledgement "
                    f"token {message[1]!r}; expected {expected_token!r}."
                )
            return message

    def _send_and_wait(self, command_name, *values, response_name="ack"):
        self._raise_if_sampling_failed()
        if not self._started or self._process is None:
            raise RuntimeError("The GPU monitor process has not been started.")
        if not self._process.is_alive():
            raise RuntimeError(
                "The GPU monitor process stopped unexpectedly with exit "
                f"code {self._process.exitcode}."
            )

        token = self._new_token()
        self._command_queue.put((command_name, token, *values))
        return self._wait_for_message(response_name, token)

    def _synchronize_samples(self):
        self._send_and_wait("barrier")

    def _raise_if_sampling_failed(self):
        if self._sampling_error is not None:
            raise RuntimeError(
                f"GPU metric sampling failed: {self._sampling_error}"
            )

    def start(self):
        if self._started:
            return

        process_context = multiprocessing.get_context("spawn")
        self._command_queue = process_context.Queue()
        self._output_queue = process_context.Queue()
        self._process = process_context.Process(
            target=_gpu_metrics_worker,
            args=(
                self.gpu_id,
                self.sample_interval_s,
                self._command_queue,
                self._output_queue,
            ),
            name="gpu-metrics-monitor",
            daemon=True,
        )
        self._process.start()
        self._started = True
        self._wait_for_message("ready")

    def begin_interval(self):
        self._raise_if_sampling_failed()
        self._synchronize_samples()
        sample_index = len(self._samples)
        start_metrics = self._read_gpu_metrics()
        start_time = time.perf_counter()

        return {
            "start_time": start_time,
            "start_metrics": start_metrics,
            "sample_index": sample_index,
        }

    def end_interval(self, interval):
        self._raise_if_sampling_failed()
        end_time = time.perf_counter()
        end_metrics = self._read_gpu_metrics()
        # The barrier is queued after all preceding worker samples. Receiving
        # its acknowledgement guarantees that no in-interval sample remains
        # buffered in the inter-process queue.
        self._synchronize_samples()

        sampled_points = list(self._samples[interval["sample_index"] :])
        sampled_points = [
            point
            for point in sampled_points
            if interval["start_time"] <= point[0] <= end_time
        ]
        points = [
            (interval["start_time"], interval["start_metrics"]),
            *sampled_points,
            (end_time, end_metrics),
        ]
        points.sort(key=lambda point: point[0])

        duration_s = end_time - interval["start_time"]
        powers = [metrics["power_w"] for _, metrics in points]
        gpu_memories = [
            metrics["gpu_memory_used_mib"] for _, metrics in points
        ]

        minimum_power_w = min(powers)
        peak_power_w = max(powers)
        minimum_gpu_memory_used_mib = min(gpu_memories)
        peak_gpu_memory_used_mib = max(gpu_memories)

        return {
            "duration_s": duration_s,
            "minimum_power_w": minimum_power_w,
            "peak_power_w": peak_power_w,
            "minimum_gpu_memory_used_mib": minimum_gpu_memory_used_mib,
            "peak_gpu_memory_used_mib": peak_gpu_memory_used_mib,
            "sample_count": len(points),
        }

    def stop(self):
        stop_error = None
        try:
            if self._started and self._process is not None:
                if self._process.is_alive():
                    try:
                        self._send_and_wait("stop", response_name="stopped")
                    except Exception as error:
                        stop_error = error

                self._process.join(timeout=MONITOR_CONTROL_TIMEOUT_SECONDS)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5.0)
                    if stop_error is None:
                        stop_error = RuntimeError(
                            "The GPU monitor process did not stop cleanly."
                        )
        finally:
            self._started = False
            if self._command_queue is not None:
                self._command_queue.close()
                self._command_queue.join_thread()
            if self._output_queue is not None:
                self._output_queue.close()
                self._output_queue.join_thread()
            self._shutdown_nvml()

        if stop_error is not None:
            raise stop_error

def qft_Qiskit(num_qubits, is_transpiled=False):
    import qiskit
    import numpy as np
    """QFT on the first n qubits in circuit"""
    def qft_rotations_Qiskit(qc: qiskit.QuantumCircuit, num_qubits, transpile=False):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if num_qubits == 0:
            return qc
        num_qubits -= 1
        qc.h(num_qubits)
        for j in range(num_qubits):
            
            if(transpile):
                qc.rz(np.pi/2**(num_qubits-j) / 2, j)
                qc.cx(j, num_qubits)
                qc.rz(-np.pi/2**(num_qubits-j) / 2, num_qubits)
                qc.cx(j, num_qubits)
                qc.rz(+np.pi/2**(num_qubits-j) / 2, num_qubits)
            else:
                qc.cp(np.pi/2**(num_qubits-j), j, num_qubits)

            # qc.barrier()
        qft_rotations_Qiskit(qc, num_qubits, transpile=transpile)
    def swap_registers_Qiskit(qc: qiskit.QuantumCircuit, num_qubits, transpile=False):
        for j in range(num_qubits//2):
            if(transpile):
                qc.cx(j, num_qubits-j-1)
                qc.cx(num_qubits-j-1, j)
                qc.cx(j, num_qubits-j-1)
            else:
                qc.swap(j, num_qubits-j-1)
            # qc.barrier()
        return qc
    qc = qiskit.QuantumCircuit(num_qubits)
    qft_rotations_Qiskit(qc, num_qubits, transpile=is_transpiled)
    swap_registers_Qiskit(qc, num_qubits, transpile=is_transpiled)
    return qc


def prepare_qft_circuit(num_qubits, is_transpiled=False):
    qc = qft_Qiskit(num_qubits, is_transpiled=is_transpiled)
    qc.save_statevector()
    return qc


def run_qft_on_gpu(qc, simulator):
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


def gpu_card_name_slug(device_name):
    """Return a filename-friendly GPU model, such as '3090'."""
    generic_words = {"nvidia", "geforce", "graphics", "gpu", "rtx"}
    model_parts = [
        part
        for part in re.findall(r"[a-z0-9]+", str(device_name).lower())
        if part not in generic_words
    ]
    return "_".join(model_parts) or "unknown"


def read_output_file(is_transpiled, device_name):
    transpilation_name = "trans" if is_transpiled else "no_trans"
    default_filename = (
        f"qft_gpu_{gpu_card_name_slug(device_name)}_"
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
        "Use the manually transpiled QFT gate construction? [y/n]: "
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

    from qiskit_aer import AerSimulator

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
        "QFT gate construction: "
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
        f"Each qubit size uses at least {MIN_REPETITIONS} repetitions and "
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

        print("Preparing QFT circuits outside the measured intervals...")
        prepared_qft_circuits = {
            num_qubits: prepare_qft_circuit(
                num_qubits,
                is_transpiled=is_transpiled,
            )
            for num_qubits in range(min_qubits, max_qubits + 1)
        }

        for num_qubits in range(min_qubits, max_qubits + 1):
            print(f"=================> Benchmarking {num_qubits} qubits")
            qc = prepared_qft_circuits[num_qubits]
            print(
                f"Warming up this circuit with {GPU_WARMUP_RUNS} runs..."
            )
            for _ in range(GPU_WARMUP_RUNS):
                run_qft_on_gpu(qc, simulator)

            times = []
            qft_interval = metrics_monitor.begin_interval()

            while (
                len(times) < MIN_REPETITIONS
                or time.perf_counter() - qft_interval["start_time"]
                < MIN_MEASUREMENT_SECONDS
            ):
                times.append(run_qft_on_gpu(qc, simulator))

            qft_metrics = metrics_monitor.end_interval(qft_interval)
            mean_time = sum(times) / len(times)
            power_difference_w = (
                qft_metrics["peak_power_w"]
                - baseline["minimum_power_w"]
            )
            memory_difference_mib = (
                qft_metrics["peak_gpu_memory_used_mib"]
                - baseline["minimum_gpu_memory_used_mib"]
            )
            writer.writerow(
                {
                    "Qubits": num_qubits,
                    "Time_s": mean_time,
                    "PBMin_W": baseline["minimum_power_w"],
                    "PPeak_W": qft_metrics["peak_power_w"],
                    "PDiff_W": power_difference_w,
                    "MBMinMiB": baseline[
                        "minimum_gpu_memory_used_mib"
                    ],
                    "MPeakMiB": qft_metrics[
                        "peak_gpu_memory_used_mib"
                    ],
                    "MDiffMiB": memory_difference_mib,
                }
            )
            csv_file.flush()

            print(
                f"num_qubits: {num_qubits}, "
                f"average execution time: {mean_time:.9f} seconds "
                f"({len(times)} repetitions), "
                f"power baseline-min/peak/difference: "
                f"{baseline['minimum_power_w']:.3f}/"
                f"{qft_metrics['peak_power_w']:.3f}/"
                f"{power_difference_w:.3f} W, "
                f"memory baseline-min/peak/difference: "
                f"{baseline['minimum_gpu_memory_used_mib']:.3f}/"
                f"{qft_metrics['peak_gpu_memory_used_mib']:.3f}/"
                f"{memory_difference_mib:.3f} MiB"
            )

    print(f"Benchmark complete. Results saved to: {output_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
