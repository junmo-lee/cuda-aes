
# [PRD] HPC 환경 기반 하이브리드 AES-128 브루트 포싱 엔진

**버전:** 1.1

**작성일:** 2026년 3월 10일

**프로젝트 성격:** HPC(High Performance Computing) 기반 암호 해독 성능 검증 및 가속

---

## 1. 프로젝트 개요 (Executive Summary)

본 프로젝트는 초대규모 HPC 인프라(120개 노드)를 활용하여 AES-128 ECB 암호문을 브루트 포싱(전수 조사)으로 해독하는 고성능 시스템 구축을 목표로 합니다. NVIDIA GPU의 병렬 연산 능력과 CPU의 AES-NI 명령어셋을 결합한 하이브리드 아키텍처를 채택하며, MPI를 통해 수백 대의 노드를 효율적으로 관리합니다.

## 2. 시스템 목표 (Objectives)

* **하이브리드 가속:** GPU(CUDA)와 CPU(AES-NI)의 연산 자원을 동시에 활용하여 처리량 최적화.
* **확장성(Scalability):** MPI를 통해 120개 이상의 노드로 연산 범위를 분산하고 관리.
* **유연성:** 현재 AES-128 ECB를 타겟으로 하되, 향후 CTR 모드 및 AES-256으로의 확장 구조 확보.

## 3. 상세 요구사항 (Technical Requirements)

### 3.1 알고리즘 및 연산 (Core Engine)

* **타겟:** AES-128 ECB (Electronic Codebook).
* **커널 참조:** [burcel/aes-cuda](https://github.com/burcel/aes-cuda) 레포지토리의 CUDA 가속 코드를 기반으로 복호화 엔진 추출 및 최적화.
* **브루트 포싱 범위:** * 기본 128비트 키 공간 조사.
* 256비트 모드 시, 상위/하위 128비트를 0으로 고정하고 나머지 128비트 영역만 집중 조사하는 구조 설계.


* **CPU 연산:** Intel/AMD의 **AES-NI(New Instructions)**를 사용하여 GPU와 병렬로 복호화 수행.

### 3.2 입력 및 출력 (I/O)

* **입력값:**
* `Ciphertext Block`: 해독 대상 암호문 (16 bytes).
* `Plaintext Block`: 일치 여부를 확인할 원본 평문 (16 bytes).
* `Key Range (Start, End)`: 해당 노드/프로세스가 담당할 키 탐색 시작점과 끝점.


* **출력값:**
* 성공 시: 찾아낸 `Correct Key` 값.
* 진행 중: 실시간 탐색 속도 및 진행률 로그.



### 3.3 리소스 및 노드 관리 (Infrastructure & Management)

* **분산 환경:** 1개의 메인(Master) 노드와 120개의 연산(Worker) 노드 운영 (MPI 통신).
* **노드 구성:** 노드당 최소 2개 이상의 GPU 탑재.
* **부하 분산(Load Balancing):** * 각 노드 내 CPU와 GPU의 성능 측정(Benchmarking) 결과를 기반으로 Key Space 분할 비율 결정.
* 예: GPU 성능이 CPU보다 10배 빠를 경우, GPU에 90%, CPU에 10%의 범위를 할당.


* **모니터링 쓰레드:** 각 노드에서 1개의 CPU 쓰레드를 'Reservation Thread'로 지정하여 하드웨어 상태 모니터링 및 MPI 통신 전담.

---

## 4. 시스템 아키텍처 (System Architecture)

### 4.1 소프트웨어 스택

* **Communication:** MPI (OpenMPI 또는 MPICH).
* **GPU Engine:** CUDA C++ (Reference: burcel/aes-cuda).
* **CPU Engine:** C++ with Intrinsic Functions (`wmmintrin.h`) for AES-NI.
* **OS:** Linux (HPC Cluster 환경).

### 4.2 데이터 흐름 (Data Flow)

1. **Master Node:** 전체 키 공간을 120개 노드 단위로 분할하여 MPI로 전달.
2. **Worker Node:** * 받은 범위를 다시 내부 GPU(0, 1...)와 CPU 쓰레드에 분배.
* 모니터링 쓰레드는 주기적으로 진행률을 수집.


3. **Find Key:** 일치하는 키 발견 시 즉시 MPI 인터럽트/메시지를 통해 Master 노드에 보고 후 전체 작업 종료.

---

## 5. 마일스톤 (Milestones)

| 단계 | 주요 과업 (Key Tasks) | 비고 |
| --- | --- | --- |
| **M1. 커널 추출** | `aes-cuda` 레포지토리 분석 및 독립적인 복호화 커널 함수 추출 | 검증용 테스트 케이스 작성 |
| **M2. 다중 GPU 통합** | 한 노드 내에서 여러 개의 GPU(Multi-GPU)를 제어하는 코드 구현 | `cudaSetDevice` 활용 |
| **M3. AES-NI 구현** | CPU 기반 AES-NI 복호화 모듈 개발 및 성능 최적화 | Intrinsics 활용 |
| **M4. 내부 부하 분산** | 노드 내 CPU/GPU 연산 속도비 계산 및 자동 키 범위 분배 알고리즘 적용 | 가중치 기반 분할 |
| **M5. MPI 및 로그** | 120개 노드 연동, 모니터링 쓰레드 설계 및 통합 로그 시스템(파일) 구축 | 진행률 실시간 기록 |

---

## 6. 기타 고려사항

* **로그 관리:** 120개 노드에서 발생하는 로그가 과도할 수 있으므로, 각 노드는 로컬 로그를 남기고 Master는 요약 정보만 수집하도록 설계.
* **예외 처리:** 특정 노드 다운 시 해당 노드의 키 범위를 다른 노드에 재할당하는 Fault-tolerance 구조 검토(향후 과제).
