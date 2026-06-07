# 이사 물품 부피 추정 Agent

이 프로젝트는 이사 물품 사진과 물품 목록 텍스트를 입력받아, 이미지 속 물체를 분할하고 깊이를 추정한 뒤 물체별 부피와 전체 적재 부피를 계산하는 시스템입니다. 계산 결과를 바탕으로 추천 차량 톤수와 견적서 PDF를 생성하며, `agent_ui/server.py`를 통해 로컬 웹 UI에서 실행할 수 있습니다.

## 폴더 구성

- `agent_ui/`: 로컬 웹 서버와 프론트엔드 UI 파일
- `pipeline.py`: 전체 추론 파이프라인 실행 파일
- `config.py`: 모델 경로, 백엔드, 임계값 등 설정
- `segmentation.py`: SAM3 기반 물체 분할
- `depth_estimation.py`: AnyCalib + UniDepth 기반 깊이 추정
- `text_parser.py`: 한국어 물품 목록을 영어 객체명으로 변환
- `volume_calculator.py`, `shape_priors.py`, `standard_size_agent.py`: 부피 및 표준 크기 보정 로직
- `retpdf/`: 견적서 PDF 생성용 HTML 템플릿, JSON 데이터, 요금 규칙
- `realdata_anonymized/_anonymized/anon_0003/`: 기본 실행 예시 입력 데이터
- `setup.sh`: 실행 환경 및 필요한 패키지 설치 스크립트

## 필요한 라이브러리 및 환경

기본 실행 환경은 Python 3.11과 CUDA 사용 가능한 PyTorch 환경을 기준으로 합니다. 주요 라이브러리는 다음과 같습니다.

- `torch`, `torchvision`: 딥러닝 모델 실행
- `transformers`, `accelerate`, `qwen-vl-utils`: Qwen 텍스트/VLM 모델 실행
- `opencv-python`, `numpy`, `pillow`: 이미지 처리
- `sam3`, `timm`, `ftfy`, `iopath`: SAM3 물체 분할
- `open3d`, `scipy`: 3D 포인트 및 부피 계산
- `huggingface_hub`, `safetensors`, `omegaconf`: 모델 가중치 및 설정 로딩

설치는 `setup.sh`를 실행해 진행합니다.

```bash
cd /home/irteam/data-vol1/agent/agent_submit
bash setup.sh
```

실제 서버 실행은 기존 프로젝트 환경과 동일하게 `volume_est` conda 환경을 사용합니다. 견적서 PDF 생성에는 `pdf` conda 환경과 WeasyPrint가 필요할 수 있습니다.

## 외부 모델 및 경로

제출 폴더에는 코드와 실행에 필요한 설정 파일만 포함되어 있으며, 대용량 모델 체크포인트는 포함하지 않았습니다. 실행 환경에는 아래 리소스가 준비되어 있어야 합니다.

- SAM3: `facebook/sam3` 또는 로컬 SAM3 설치 경로
- SAM3 BPE 파일: 기본값 `$HOME/data-vol1/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- AnyCalib: 기본값 `/home/irteam/data-vol1/AnyCalib`
- UniDepth 모델 가중치: Hugging Face cache 또는 실행 중 다운로드 가능한 환경
- Qwen 텍스트 모델: `Qwen/Qwen3-4B-Instruct-2507`

필요하면 환경변수로 경로를 바꿀 수 있습니다.

```bash
export ANYCALIB_ROOT=/path/to/AnyCalib
export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
```

## 실행 방법

웹 UI 서버를 실행합니다.

```bash
cd /home/irteam/data-vol1/agent/agent_submit
python agent_ui/server.py --host 0.0.0.0 --port 7860
```

브라우저에서 다음 주소로 접속합니다.

```text
http://localhost:7860
```

UI에서 이미지 폴더, 물품 목록 텍스트 파일, 출력 폴더를 입력한 뒤 실행 버튼을 누르면 서버가 내부적으로 아래 형태의 명령을 실행합니다.

```bash
conda activate volume_est
python -u pipeline.py \
  --image-dir realdata_anonymized/_anonymized/anon_0003 \
  --text-file realdata_anonymized/_anonymized/anon_0003/003_items.txt \
  --depth-backend anycalib_unidepth \
  --seg-backend sam3 \
  --output out_anycalib_unidepth/anon_0003_item \
  --json-only \
  --pdf
```

## 입력 형식

이미지 입력은 하나의 폴더에 여러 장의 사진을 넣는 방식입니다. 지원 확장자는 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`입니다.

텍스트 입력은 이사할 물품 목록을 자연어로 작성한 `.txt` 파일입니다. 예시는 다음과 같습니다.

```text
침대, 책상, 의자, 냉장고, 전자레인지 옮겨줘
```

## 출력 결과

실행이 끝나면 출력 폴더에 다음 결과가 생성됩니다.

- `batch_summary.json`: 전체 이미지 처리 결과, 총 부피, 추천 차량 톤수
- `photo_xxx/result.json`: 이미지별 검출 물체, 크기, 부피
- `photo_xxx/depth_with_objects.png`: 깊이 및 검출 물체 시각화 이미지
- `quote.html`, `quote.pdf`: 자동 생성 견적서

웹 UI에서는 진행 로그, 현재 처리 이미지, 총 부피, 추천 차량, 결과 이미지, PDF 다운로드를 확인할 수 있습니다.

## 모델 동작 방식

1. 한국어 물품 목록을 파싱해 추정 대상 물체명을 구성합니다.
2. SAM3로 사진 속 물체 영역을 분할합니다.
3. AnyCalib + UniDepth로 깊이 맵과 카메라 스케일을 추정합니다.
4. 분할 마스크와 깊이 정보를 이용해 물체별 3D 크기와 부피를 계산합니다.
5. 표준 크기 사전을 활용해 비정상적인 크기 추정값을 보정합니다.
6. 전체 부피를 합산해 추천 차량 톤수를 계산합니다.
7. 계산 결과와 고객 정보를 바탕으로 견적서 PDF를 생성합니다.

## 평가 방법

기본 예시 데이터로 파이프라인이 정상 동작하는지 확인합니다.

```bash
cd /home/irteam/data-vol1/agent/agent_submit
conda activate volume_est
python -u pipeline.py \
  --image-dir realdata_anonymized/_anonymized/anon_0003 \
  --text-file realdata_anonymized/_anonymized/anon_0003/003_items.txt \
  --depth-backend anycalib_unidepth \
  --seg-backend sam3 \
  --output out_anycalib_unidepth/anon_0003_item \
  --json-only \
  --pdf
```

평가 시에는 다음 항목을 확인합니다.

- 입력 이미지 수와 처리된 이미지 수가 일치하는지
- `batch_summary.json`이 생성되었는지
- 물체별 `result.json`에 검출 객체와 부피가 기록되는지
- `total_volume_m3`와 `recommended_truck` 값이 생성되는지
- `depth_with_objects.png` 시각화 결과가 생성되는지
- `quote.pdf` 견적서가 생성되는지

## 참고 사항

`agent_ui/server.py` 자체는 Python 표준 라이브러리만 사용하지만, 실제 추론은 `pipeline.py`에서 딥러닝 모델을 로드하므로 GPU와 모델 가중치가 필요합니다. 제출 폴더에는 실행에 불필요한 기존 결과물, 로그, 캐시, 대용량 체크포인트는 포함하지 않았습니다.
