# MPI Architecture Analysis: cuda-aes

The `cuda-aes` project implements a hybrid AES-128 brute-force engine using a **Master-Worker** architecture via MPI. This document provides a detailed breakdown of how MPI processes are structured and how they communicate.

---

## 1. Overall Architecture

The system consists of one Master process (Rank 0) and one or more Worker processes (Rank 1..N).

- **Master (Rank 0)**: Acts as a central controller. It partitions the 128-bit key space and manages work distribution and termination.
- **Worker (Rank 1..N)**: Performs the actual brute-force search. Each worker leverages its available local hardware (both GPUs and CPUs) for parallel processing.

---

## 2. Process Roles

### 2.1 Master (Rank 0)
The Master's lifecycle is managed by the `Master` class in `src/master.cpp`.

1.  **Work Splitting**:
    -   Divides the target 128-bit key range `[ks, ke)` into `N` equal-sized chunks (one for each worker).
    -   Currently, the splitting logic (`split_range`) operates on the lower 64 bits of the key space for simplicity.
2.  **Work Distribution**:
    -   Constructs `WorkAssignment` structures containing the plaintext, ciphertext, and the assigned key range.
    -   Dispatches work to each worker using `MPI_Send` with `TAG_WORK`.
3.  **Result Collection**:
    -   Enters a loop to wait for results from all workers via `MPI_Recv` with `TAG_RESULT`.
    -   If a `WorkResult` indicates the key was found (`found == 1`), the Master records the result.
4.  **Termination Control**:
    -   After receiving results from all workers, the Master broadcasts a STOP signal (a `WorkAssignment` with `key_end` set to zero) to all workers.

### 2.2 Worker (Rank 1..N)
The Worker's lifecycle is managed by the `Worker` class in `src/worker.cpp`.

1.  **Benchmarking**:
    -   Upon startup, each worker benchmarks its available GPUs and CPU threads to determine their search speeds.
2.  **Work Execution Loop**:
    -   Waits for a `WorkAssignment` from the Master using `MPI_Recv` with `TAG_WORK`.
    -   If a STOP signal is received, the worker exits.
    -   If valid work is received:
        -   Internal work splitting: Further divides its assigned range between local GPUs and CPU threads based on benchmark ratios.
        -   Launches multiple local threads for parallel searching.
        -   Once the search in its assigned range completes (or the key is found), it sends a `WorkResult` back to the Master via `MPI_Send` with `TAG_RESULT`.
3.  **Local Monitoring**:
    -   While searching, a dedicated monitoring thread logs progress (keys tried and current speed) locally.

---

## 3. Communication Protocol

The system uses point-to-point communication with specific MPI tags defined in `include/utils.h`.

### 3.1 Data Structures
Two primary structures are transmitted over MPI:

| Structure | Description | Key Fields |
| :--- | :--- | :--- |
| `WorkAssignment` | Task definition sent from Master to Worker. | `ciphertext`, `plaintext`, `key_start[4]`, `key_end[4]` |
| `WorkResult` | Result sent from Worker to Master. | `rank`, `found` (flag), `key[4]` (the found key) |

### 3.2 MPI Tags
-   `TAG_WORK` (1): Used for sending work assignments and the STOP signal.
-   `TAG_RESULT` (2): Used for reporting the search outcome back to the Master.
-   `TAG_PROGRESS` (3): Reserved for future progress reporting (currently not used).

---

## 4. Control Flow Sequence

The following sequence describes the interaction during a typical search:

1.  **Initialization**: `MPI_Init` is called on all ranks.
2.  **Master**: Partition the total key range into `N` chunks.
3.  **Master → Worker**: `MPI_Send(WorkAssignment, TAG_WORK)` for each worker.
4.  **Worker**: Perform local search (GPU + CPU).
5.  **Worker → Master**: `MPI_Send(WorkResult, TAG_RESULT)` once finished or key found.
6.  **Master**: Receive all results via `MPI_Recv(TAG_RESULT)`.
7.  **Master → Worker**: `MPI_Send(STOP_SIGNAL, TAG_WORK)` to all workers.
8.  **Finalization**: `MPI_Finalize` is called on all ranks.

---

## 6. Communication Overload Analysis

The potential for server or network overload due to MPI communication in this project is analyzed below.

### 6.1 Current Risk Assessment (Low)
In the current implementation, the risk of communication-induced overload is **extremely low** because:
- **Low Frequency**: Communication only occurs at the very beginning (work assignment) and the very end (result collection) of the search process. There is no mid-process heartbeat or status exchange.
- **Small Payload**: Both `WorkAssignment` (~48 bytes) and `WorkResult` (~24 bytes) are very small, minimizing network bandwidth usage.
- **Coarse Granularity**: The master divides the entire key space into exactly `N` large chunks. This "one-shot" distribution means the number of MPI messages is strictly $O(N)$ for the entire execution.

### 6.2 Potential Risk Scenarios
Overload may become a concern if the following features are implemented:
1.  **Dynamic Work Scheduling**: If the key space is divided into thousands of tiny "micro-tasks" to balance load between fast and slow workers, the Master (Rank 0) could become a bottleneck as it processes frequent requests for new work.
2.  **Frequent Progress Reporting**: If the `TAG_PROGRESS` tag is used to send updates (e.g., every second) from hundreds of workers to the Master, the aggregate interrupt rate on Rank 0 could degrade performance.
3.  **Massive Scale-out**: On a cluster with thousands of nodes, a simultaneous "found" signal or a barrier-like result collection could lead to "Incast" congestion at the Master node's network interface.

### 6.3 Mitigation Strategies
To prevent potential overload as the project evolves:
- **Optimal Chunk Sizing**: Maintain a balance between task granularity and communication overhead. The search time for a single chunk should be significantly higher (e.g., >1000x) than the communication latency.
- **Asynchronous/Non-blocking I/O**: Use `MPI_Isend` and `MPI_Irecv` to allow the Master to overlap communication handling with other management tasks, preventing CPU stalls.
- **Hierarchical Collection**: For extreme scales, implement a tree-based reduction for results and progress reports to distribute the communication load across multiple ranks instead of a single Master.
- **Exponential Backoff/Throttling**: If using a dynamic queue, implement worker-side throttling for work requests to prevent flooding the Master during network jitter.
