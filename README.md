# 🌲 B+ Tree — GPU-Accelerated with CUDA

A high-performance **B+ Tree** index structure implemented in CUDA C++, designed
to leverage GPU parallelism for executing multiple database-style queries
simultaneously. Built on top of CUDA Unified/Pinned Memory, the tree structure
is accessible from both host and device, enabling concurrent GPU kernel
execution for point lookups, range queries, record updates, and path tracing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Structure Details](#data-structure-details)
- [Supported Operations](#supported-operations)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Building & Running](#building--running)
- [Implementation Notes](#implementation-notes)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project implements a **B+ Tree** backed by CUDA pinned memory
(`cudaHostAlloc`), allowing the tree nodes to be accessed directly by GPU
threads without explicit device-to-host memory transfers. The tree is built on
the host CPU, and query operations (search, range query) are then dispatched as
parallel GPU kernels — each CUDA thread handles one independent query.

This is particularly well-suited for database-style workloads where you need to
run many independent lookups or range scans over a pre-built index
simultaneously.

---

## Features

- ✅ **GPU-parallel point search** — batch of `p` search keys dispatched as `p`
  GPU threads
- ✅ **GPU-parallel range queries** — multiple `[A, B]` range scans run in
  parallel on GPU
- ✅ **Record update (addition)** — modify specific attributes of matched
  records in-place
- ✅ **Path tracing** — trace the root-to-leaf path for a key, recording all
  first-keys at each level
- ✅ **Tree height** — O(log n) height computation via a GPU kernel
- ✅ **Pinned memory** — all nodes and the database reside in CUDA host-pinned
  memory, enabling zero-copy access from device
- ✅ **Configurable order** — `MAXI = 7` (max keys per node), easily changed at
  compile time
- ✅ **File-driven** — fully driven by a structured input file; results written
  to an output file

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Host (CPU)                           │
│                                                             │
│  ┌─────────────┐      ┌───────────────────────────────┐    │
│  │  Input File │ ───► │   BPTree (pinned memory)      │    │
│  │  (records + │      │   ┌──────────────────────┐    │    │
│  │   queries)  │      │   │  Internal Nodes       │    │    │
│  └─────────────┘      │   │  (key arrays + ptrs)  │    │    │
│                        │   ├──────────────────────┤    │    │
│  Build phase:          │   │  Leaf Nodes           │    │    │
│  insert() called       │   │  (keys → record ptrs) │    │    │
│  per record            │   └──────────────────────┘    │    │
│                        └───────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │  zero-copy access (pinned mem)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Device (GPU)                         │
│                                                             │
│   ┌──────────────────┐  ┌───────────────────────────────┐  │
│   │  Search kernel   │  │  RangeQuery kernel            │  │
│   │  (p threads)     │  │  (p threads, one per range)   │  │
│   └──────────────────┘  └───────────────────────────────┘  │
│   ┌──────────────────┐                                      │
│   │  Height kernel   │                                      │
│   │  (1 thread)      │                                      │
│   └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

### Class Hierarchy

| Class     | Role                                                                                    |
| --------- | --------------------------------------------------------------------------------------- |
| `Managed` | Base class overriding `new`/`delete` to use `cudaHostAlloc` / `cudaFree`                |
| `Node`    | A B+ Tree node holding an integer key array, pointer array, a size, and a leaf flag     |
| `BPTree`  | The B+ Tree with insert, search, range query, path trace, addition, display, and height |

---

## Data Structure Details

### Node Layout

Each `Node` stores:

- `int* key` — array of up to `MAXI` integer keys (allocated via
  `cudaHostAlloc`)
- `Node** ptr` — array of `MAXI + 1` child/record pointers (allocated via
  `cudaHostAlloc`)
- `int size` — number of keys currently in the node
- `bool IS_LEAF` — distinguishes internal nodes from leaf nodes

**Tree Order:** `MAXI = 7` — each node holds at most 7 keys and 8 pointers.

### Leaf Node vs Internal Node

| Property        | Internal Node             | Leaf Node                                 |
| --------------- | ------------------------- | ----------------------------------------- |
| `IS_LEAF`       | `false`                   | `true`                                    |
| `ptr[i]`        | Points to child `Node*`   | Points to the database record (`int*`)    |
| `ptr[size]`     | Points to rightmost child | Points to the **next leaf** (linked list) |
| Split behaviour | Middle key pushed up      | Middle key **copied up**                  |

### Pinned Memory Allocation

Both `Node` and `BPTree` inherit from `Managed`, which overrides `operator new`
to call `cudaHostAlloc(..., cudaHostAllocDefault)`. This places all allocations
in **page-locked host memory** that is directly accessible from the GPU without
copying, making the tree traversal possible inside device kernels with no extra
`cudaMemcpy`.

---

## Supported Operations

The program reads a sequence of operations from the input file. Four operation
types are supported:

### Op 1 — Parallel Point Search

Dispatches `p` CUDA threads, each independently searching the B+ tree for one
key. Returns the full record for each key, or `-1` if not found.

```
CUDA kernel: Search<<<grid, 64>>>(p, node, gkey, gptr)
```

- Each thread calls `BPTree::search(key[id])` and stores the resulting record
  pointer.
- Grid size = ⌈p / 64⌉ blocks of 64 threads.

### Op 2 — Parallel Range Query

Dispatches `p` CUDA threads, each performing an independent range scan
`[A[id], B[id]]`. Results are stored into a pre-allocated flat result array
using prefix-sum offsets (`sz[]`), avoiding write conflicts between threads.

```
CUDA kernel: RangeQuery<<<grid, 64>>>(p, gA, gB, gsz, gz, node, gres)
```

- Each thread calls `BPTree::rangequery(A[id], B[id], &res[sz[id]])`.
- `sz[id]` ensures each thread writes to a non-overlapping segment of the output
  buffer.
- The result count for each query is returned in `z[id]`.

### Op 3 — Parallel Record Update (Addition)

Searches for `p` records by key (GPU-parallel point search), then on the CPU
side adds `Val[j]` to attribute `Att[j]` (1-indexed) of each found record
in-place.

- Uses the `Search` kernel to batch-resolve record pointers.
- CPU then applies `*(res[j] + (Att[j] - 1)) += Val[j]` for each hit.
- Since the database lives in pinned memory, updates are immediately visible to
  subsequent GPU operations.

### Op 4 — Path Trace

CPU-side operation. Traverses from root to the leaf containing key `k`,
recording the first key of each node encountered at every level.

```
int lt = node.pathtrace(k, out);
// out[0..lt-1] contains the first key at each level along the search path
```

Output is a space-separated sequence of keys from root to leaf's first key.

---

## Input Format

The program is invoked with two file arguments:

```
./b+tree_gpu <input_file> <output_file>
```

**Input file structure:**

```
<n> <m>
<record_1_attr_1> <record_1_attr_2> ... <record_1_attr_m>
<record_2_attr_1> ...
...
<record_n_attr_1> ...
<q>
<op_1> [op_1_arguments]
<op_2> [op_2_arguments]
...
```

- **Line 1:** `n` = number of records, `m` = number of attributes per record
- **Lines 2 to n+1:** The database — each record is `m` integers; the **first
  attribute is the key**
- **Line n+2:** `q` = number of operations
- **Remaining lines:** Operations, one per line:

| Op  | Format                              | Description                                   |
| --- | ----------------------------------- | --------------------------------------------- |
| `1` | `1 p k1 k2 ... kp`                  | Point search for `p` keys                     |
| `2` | `2 p a1 b1 a2 b2 ... ap bp`         | Range query `p` ranges                        |
| `3` | `3 p k1 att1 val1 ... kp attp valp` | Add `val` to `att`-th attribute of record `k` |
| `4` | `4 k`                               | Path-trace for key `k`                        |

**Example:**

```
5 3
10 100 200
20 200 300
30 300 400
40 400 500
50 500 600
3
1 2 10 30
2 1 15 35
4 25
```

---

## Output Format

Results are written to the output file.

| Operation             | Output                                                             |
| --------------------- | ------------------------------------------------------------------ |
| **Op 1** (search)     | One line per key: space-separated attributes, or `-1` if not found |
| **Op 2** (range)      | One line per matched record; `-1` if range is empty                |
| **Op 3** (update)     | No output; updates are applied in-place                            |
| **Op 4** (path trace) | Space-separated first-keys at each level from root to leaf         |

---

## Building & Running

### Prerequisites

| Requirement  | Recommended Version               |
| ------------ | --------------------------------- |
| NVIDIA GPU   | Compute Capability 3.5+           |
| CUDA Toolkit | 11.x or 12.x                      |
| C++ Compiler | MSVC (Windows) or GCC/G++ (Linux) |
| `nvcc`       | Included with CUDA Toolkit        |

### Compile

**Windows (with CUDA Toolkit installed):**

```powershell
nvcc -o b+tree_gpu b+tree_gpu.cu
```

**Linux:**

```bash
nvcc -o b+tree_gpu b+tree_gpu.cu -std=c++14
```

### Run

```bash
./b+tree_gpu input.txt output.txt
```

---

## Implementation Notes

### Memory Management

- All `Node` objects and the `BPTree` itself are allocated in CUDA **pinned host
  memory** via `cudaHostAlloc`. This allows GPU kernels to traverse the tree
  directly using the same host pointers, with no need for separate device copies
  of the structure.
- The flat database array `db` (size `n × m`) is similarly allocated in pinned
  memory so that record pointers stored in leaf nodes are directly
  dereferenceable on the device.

### Parallel Search Strategy

- GPU threads execute independent traversals of the **same shared tree**
  (read-only during query phase). Because multiple threads share the same tree
  without modification, no synchronization or locking is needed.
- Grid configuration: `block = 64`, `grid = ⌈p / 64⌉`.

### Leaf Linked List

Leaf nodes form a **singly linked list** through `ptr[size]`. The range query
exploits this: after finding the first leaf containing keys ≥ `a`, it walks the
linked list until it passes `b`, collecting all keys in `[a, b]`.

### Split Policy

- **Leaf split:** Left leaf gets `⌊(MAXI+1)/2⌋` keys; right gets the remainder.
  The first key of the right leaf is **copied up** to the parent.
- **Internal split:** Left node gets `⌊(MAXI+1)/2⌋` keys; right gets
  `MAXI - ⌊(MAXI+1)/2⌋` keys. The middle key is **pushed up** (not retained in
  either child).

---

## Limitations & Future Work

| Limitation                  | Notes                                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **No concurrent insertion** | The tree is built sequentially on the CPU before any GPU queries                                                              |
| **No deletion**             | Only insertion and query operations are implemented                                                                           |
| **Fixed order**             | `MAXI` is a compile-time constant; changing it requires recompilation                                                         |
| **Single tree**             | Only one B+ tree index (on the first attribute) is built                                                                      |
| **No error checking**       | CUDA API return values are generally not checked                                                                              |
| **Range buffer sizing**     | Range query result buffers are sized assuming worst-case `B - A + 1` results per query, which may over-allocate significantly |

### Potential Improvements

- Add `cudaError_t` checking with descriptive error messages
- Support concurrent GPU insertions using atomic operations or lock-free
  techniques
- Support deletion with underflow merging/redistribution
- Add secondary index support (multi-attribute indexing)
- Benchmark against CPU-only baseline and document speedups

---

## License

This project is provided for educational and research purposes.
