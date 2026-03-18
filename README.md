# HPC 환경 기반 하이브리드 AES-128 브루트 포싱 엔진

NVIDIA GPU(CUDA)와 CPU(AES-NI)를 결합한 하이브리드 아키텍처로 AES-128 ECB 암호문을 전수 조사(Brute-force)하는 고성능 분산 컴퓨팅 시스템입니다. MPI를 통해 120개 이상의 HPC 노드에 연산을 분산합니다.

---

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [빌드 방법](#빌드-방법)
3. [실행 방법](#실행-방법)
4. [인수 설명](#인수-설명)
5. [아키텍처 개요](#아키텍처-개요)
6. [성능 지표](#성능-지표)
7. [로그 시스템](#로그-시스템)
8. [HPC 클러스터 제출](#hpc-클러스터-제출)
9. [유의사항 및 제약](#유의사항-및-제약)

---

## 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|---|---|---|
| OS | Linux (x86_64) | Ubuntu 22.04 / RHEL 9 |
| CUDA | 11.0 이상 | 12.x 이상 |
| GPU | NVIDIA (Compute 7.0+) | A100 / H100 |
| CPU | AES-NI 지원 Intel/AMD | Xeon / EPYC |
| MPI | OpenMPI 3.x / MPICH 3.x | OpenMPI 4.x |
| CMake | 3.18 이상 | 3.25 이상 |
| GCC | 9.0 이상 | 12.x 이상 |

### 패키지 설치 (Ubuntu/Debian)

```bash
# MPI
apt-get install -y libopenmpi-dev openmpi-bin

# CUDA Toolkit (NVIDIA 공식 가이드 참조)
# https://developer.nvidia.com/cuda-downloads
```

### AES-NI 지원 확인

```bash
grep -m1 'aes' /proc/cpuinfo && echo "AES-NI 지원됨" || echo "AES-NI 미지원"
```

> AES-NI가 없는 CPU 환경에서는 CPU 브루트 포스 모듈이 컴파일 오류를 냅니다.
> 해당 경우 `CMakeLists.txt`에서 `-maes -msse4.1` 플래그를 제거하고 `src/aes_cpu.cpp`의 intrinsic 코드를 소프트웨어 구현으로 대체해야 합니다.

---

## 빌드 방법

```bash
# 1. 저장소 클론 / 디렉토리 이동
cd /path/to/cuda-aes

# 2. CMake 구성
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. 빌드 (병렬)
cmake --build build -j$(nproc)

# 실행 파일 위치
ls build/aes_bruteforce
```

### CUDA 아키텍처 지정

기본값은 `70;80;86;90` (Volta, Ampere, Ada, Hopper)입니다. 환경에 맞게 조정하세요.

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;90"
```

---

## 실행 방법

### 기본 구조

```
mpirun -n <총 프로세스 수> ./build/aes_bruteforce [옵션]
```

- **Rank 0** : Master 노드 — 키 공간 분할 및 결과 수집
- **Rank 1 ~ N** : Worker 노드 — GPU/CPU 병렬 브루트 포스 수행

> **최소 2개 rank 필요** (master 1 + worker 1 이상)

### 로컬 단일 노드 테스트

```bash
mpirun --allow-run-as-root -n 2 ./build/aes_bruteforce \
    --pt 3243f6a8885a308d313198a2e0370734 \
    --ct 3925841d02dc09fbdc118597196a0b32 \
    --ks 2b7e151628aed2a6abf7158809cf4f00 \
    --ke 2b7e151628aed2a6abf7158809cf4f40
```

### 다중 노드 실행 (hostfile 사용)

```bash
# hostfile 예시
cat hosts.txt
# node01 slots=1
# node02 slots=1
# node03 slots=1

mpirun -n 121 --hostfile hosts.txt ./build/aes_bruteforce \
    --pt <평문_32자리_hex> \
    --ct <암호문_32자리_hex> \
    --ks 00000000000000000000000000000000 \
    --ke ffffffffffffffffffffffffffffffff
```

---

## 인수 설명

모든 값은 **16바이트 = 32자리 16진수 문자열** (소문자, `0x` 접두사 없음)입니다.

| 인수 | 설명 | 예시 |
|---|---|---|
| `--pt` | 알려진 평문 (Plaintext) | `3243f6a8885a308d313198a2e0370734` |
| `--ct` | 해독 대상 암호문 (Ciphertext) | `3925841d02dc09fbdc118597196a0b32` |
| `--ks` | 탐색 시작 키 (Key Start, inclusive) | `00000000000000000000000000000000` |
| `--ke` | 탐색 종료 키 (Key End, exclusive) | `00000000000000000000000100000000` |

> 인수를 생략하면 FIPS-197 표준 테스트 벡터의 좁은 범위를 탐색합니다 (개발/검증용).

### 테스트 벡터 (FIPS-197 Appendix B)

```
Key       : 2b7e151628aed2a6abf7158809cf4f3c
Plaintext : 3243f6a8885a308d313198a2e0370734
Ciphertext: 3925841d02dc09fbdc118597196a0b32
```

---

## 아키텍처 개요

```
mpirun
├── Rank 0 (Master)
│   ├── 전체 키 공간을 N등분
│   ├── 각 Worker에게 WorkAssignment 전송 (MPI_Send)
│   ├── 모든 Worker의 WorkResult 수신 (MPI_Recv)
│   └── 키 발견 시 결과 출력 후 전체 종료 신호 전파
│
└── Rank 1..N (Worker)
    ├── [Benchmark] GPU마다 T-table 커널과 비트슬라이싱 커널 속도를 각각 측정 및 비교
    ├── [Split] 비트슬라이싱 GPU 속도 + CPU 속도 비율에 따라 키 범위 자동 분배
    │   예) GPU(BS) 8 Gkeys/s + CPU 0.5 Gkeys/s
    │       → GPU:CPU = 94%:6%
    ├── [GPU Threads] 비트슬라이싱 커널로 브루트 포스 수행 (aes_cuda_bs.cu)
    ├── [CPU Threads] AES-NI 멀티스레드 (hardware_concurrency - 1)
    ├── [Monitor Thread] 5초마다 진행률 로깅 (예약 스레드)
    └── 키 발견 시 즉시 Master에 결과 보고
```

### GPU 커널 구현 방식 비교

| 구현 | 파일 | 방식 | 키/스레드 |
|---|---|---|---|
| T-table | `src/aes_cuda.cu` | 공유 메모리 T-table 룩업 | 1 (range 반복) |
| 비트슬라이싱 | `src/aes_cuda_bs.cu` | GF(2) 논리 연산, 룩업 테이블 없음 | 32 (레인 병렬) |

- **T-table 커널**: 1개 스레드가 키를 순차 반복. 공유 메모리에 T0 테이블(256×32 뱅크)을 캐싱해 라운드당 4번의 고속 룩업으로 AES 라운드를 처리.
- **비트슬라이싱 커널**: 1개 스레드가 32개 키를 동시 처리. 각 비트 위치를 `uint32_t` 1개로 표현(레인 j = 비트 j)하며, AES S-box를 GF(2) 조합 논리로 구현. 룩업 테이블 없이 순수 산술 연산으로 동작해 캐시 오염이 없음.

---

## 성능 지표

실측 환경 A: GPU 2장 (A계열), CPU 32코어

| 장치 | T-table 커널 | 비트슬라이싱 커널 |
|---|---|---|
| GPU 1개 (A계열) | ~26 Gkeys/s | 측정 예정 |
| GPU 2개 (단일 노드) | ~53 Gkeys/s | 측정 예정 |
| CPU 31 threads | ~1 Gkeys/s | — |
| **단일 노드 합계** | **~54 Gkeys/s** | — |
| **120 Worker 노드 추산** | **~6.4 Tkeys/s** | — |

실측 환경 B: RTX 4090 1장 (sm_89), CPU 28코어

| 장치 | T-table 커널 | 비트슬라이싱 커널 | 비율 |
|---|---|---|---|
| GPU (RTX 4090) | ~26.3 Gkeys/s | ~7.96 Gkeys/s | 0.30× |
| CPU (27 threads) | — | — | ~0.48 Gkeys/s |

> RTX 4090처럼 공유 메모리 대역폭이 넓은 GPU에서는 T-table이 비트슬라이싱보다 약 3.3배 빠릅니다.
> 비트슬라이싱은 룩업 테이블 없이 동작해 캐시 오염이 없으며, 공유 메모리가 부족한 환경이나 레지스터 압력이 낮은 아키텍처에서 상대적으로 유리할 수 있습니다.

> AES-128 전체 키 공간은 2¹²⁸ ≈ 3.4 × 10³⁸개로 현재 기술로는 완전 탐색이 불가능합니다.
> 본 시스템은 **부분 키 공간 탐색** (known plaintext attack에서 키 힌트가 있는 경우) 또는 **HPC 성능 벤치마크** 목적으로 설계되었습니다.

---

## 로그 시스템

- 각 노드는 `logs/node_<rank>.log` 에 로컬 로그를 기록합니다.
- Master(rank 0)는 전체 요약 정보만 관리합니다.
- 실행 전 `logs/` 디렉토리가 자동 생성됩니다.

```
logs/
├── node_0.log   ← Master: 키 범위 분배, 결과 수집, 총 소요시간
├── node_1.log   ← Worker 1: 벤치마크, 진행률, 발견 결과
├── node_2.log   ← Worker 2: ...
└── ...
```

로그 예시:
```
[2026-03-18 11:29:01][INFO][rank=1] Benchmarking devices…
[2026-03-18 11:29:02][INFO][rank=1]   GPU[0] T-table:   2.63e+10 keys/s
[2026-03-18 11:29:02][INFO][rank=1]   GPU[0] Bitsliced: 7.96e+09 keys/s  (0.30x vs T-table)
[2026-03-18 11:29:03][INFO][rank=1]   CPU (27 threads): 4.84e+08 keys/s
[2026-03-18 11:29:08][INFO][rank=1] Progress: 1.99e+02 keys tried, speed=3.98e+01 keys/s
```

---

## HPC 클러스터 제출

제공된 `submit.sh`를 SLURM 환경에 맞게 수정하여 사용하세요.

```bash
# 수정 후 제출
sbatch submit.sh

# 상태 확인
squeue -u $USER

# 로그 확인
tail -f logs/node_0.log
```

### SLURM 주요 설정 항목

```bash
#SBATCH -N 121              # 1 Master + 120 Worker 노드
#SBATCH --ntasks-per-node=1 # 노드당 1 MPI 프로세스
#SBATCH --gres=gpu:2        # 노드당 GPU 수 (실제 환경에 맞게 조정)
#SBATCH --time=24:00:00     # 최대 실행 시간
```

---

## 유의사항 및 제약

### 보안 및 법적 사항

- **반드시 본인이 권한을 가진 데이터에 대해서만 사용하세요.**
- 타인의 암호화 데이터에 무단으로 적용하는 것은 불법입니다.
- 본 소프트웨어는 HPC 성능 연구 및 암호학 교육 목적으로 제작되었습니다.

### 기술적 제약

| 항목 | 현재 제약 | 향후 개선 방향 |
|---|---|---|
| 키 공간 분할 | 하위 64비트 기준 | 완전한 128비트 분할 지원 |
| 노드 장애 처리 | 미구현 | Fault-tolerance (PRD 향후 과제) |
| 암호 모드 | AES-128 ECB | CTR 모드, AES-256 확장 구조 준비됨 |
| MPI 통신 | 블로킹 (Blocking) | 비동기 MPI_Irecv 도입 시 오버헤드 감소 가능 |
| 진행률 보고 | Worker 로컬 로그 | Master 집계 대시보드 추가 가능 |

### root 권한 실행

OpenMPI는 기본적으로 root 실행을 차단합니다. 개발/테스트 환경에서는 아래 플래그를 사용하세요.

```bash
mpirun --allow-run-as-root -n 2 ./build/aes_bruteforce ...
```

> 프로덕션 HPC 환경에서는 일반 계정으로 실행하는 것을 권장합니다.

### CUDA 아키텍처 불일치

실행 시 아래 오류가 발생하면 `CMakeLists.txt`의 `CMAKE_CUDA_ARCHITECTURES`를 GPU 모델에 맞게 수정 후 재빌드하세요.

```
CUDA error: no kernel image is available for execution on the device
```

```bash
# GPU 아키텍처 확인
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# 예: 8.0 → ARCHITECTURES="80"
```

---

## 참고 자료

- [burcel/aes-cuda](https://github.com/burcel/aes-cuda) — CUDA AES 커널 레퍼런스
- [FIPS-197](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf) — AES 표준 문서
- [Intel AES-NI](https://www.intel.com/content/www/us/en/developer/articles/technical/advanced-encryption-standard-instructions-aes-ni.html) — AES-NI 인트린식 가이드
- [OpenMPI Docs](https://www.open-mpi.org/doc/) — MPI 환경 설정
# cuda-aes
