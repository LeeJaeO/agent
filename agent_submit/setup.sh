#!/bin/bash
# Indoor Object Volume Estimation Pipeline - Setup Script
# 환경: volume_est (conda, Python 3.11)
# GPU: NVIDIA H200, CUDA 12.6
# 실행: bash setup.sh
#
# 주의사항:
#   - open3d는 headless 서버(libX11 없음)에서 동작 안 함 → 설치하지 않음 (PCA fallback 사용)
#   - opencv-python (non-headless)도 libGL 필요 → opencv-python-headless만 설치
#   - WeasyPrint는 pip만으로는 부족 → pango/libffi/gdk-pixbuf를 conda-forge로 함께 설치
#   - 한글 PDF 출력 시 font-ttf-noto-cjk(Noto CJK)도 같이 설치해야 글리프가 깨지지 않음

set -e

USER_HOME="${HOME}"
AGENT_ROOT="${AGENT_ROOT:-$USER_HOME/data-vol1/agent}"
SAM3_ROOT="${SAM3_ROOT:-$USER_HOME/data-vol1/sam3}"
UNIDEPTH_ROOT="${UNIDEPTH_ROOT:-$USER_HOME/data-vol1/UniDepth}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-$SAM3_ROOT/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"

echo "=== 1. Conda 환경 생성 (이미 있으면 삭제 후 재생성) ==="
conda env remove -n volume_est -y 2>/dev/null || true
conda create -n volume_est python=3.11 -y

echo ""
echo "=== 2. PyTorch (CUDA 12.6) ==="
conda run -n volume_est pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== 3. Transformers / Accelerate / PEFT / Qwen ==="
conda run -n volume_est pip install "transformers>=5.0" accelerate peft qwen-vl-utils einops tiktoken sentencepiece

echo ""
echo "=== 4. OpenCV (headless - libGL 불필요) ==="
conda run -n volume_est pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true
conda run -n volume_est pip install opencv-python-headless

echo ""
echo "=== 5. SAM3 dependencies ==="
conda run -n volume_est pip install "timm>=1.0.17" ftfy "iopath>=0.1.10" setuptools pycocotools

echo ""
echo "=== 6. Volume calculation / common ==="
conda run -n volume_est pip install scipy "numpy>=1.26,<2" pillow tqdm huggingface_hub safetensors matplotlib

echo ""
echo "=== 7. SAM3 (editable install) ==="
conda run -n volume_est pip install -e "$SAM3_ROOT" --no-deps

echo ""
echo "=== 8. C 컴파일러 (detectron2 빌드에 필요) ==="
conda install -n volume_est -c conda-forge gcc_linux-64 gxx_linux-64 -y
ln -sf /opt/conda/envs/volume_est/bin/x86_64-conda-linux-gnu-gcc /opt/conda/envs/volume_est/bin/gcc
ln -sf /opt/conda/envs/volume_est/bin/x86_64-conda-linux-gnu-g++ /opt/conda/envs/volume_est/bin/g++
ln -sf /opt/conda/envs/volume_est/bin/x86_64-conda-linux-gnu-gcc /opt/conda/envs/volume_est/bin/cc

echo ""
echo "=== 9. OpenWorldSAM (sam3_fallback 세그멘테이션 보완용) ==="
# detectron2: OpenWorldSAM의 핵심 의존성 (--no-build-isolation: 빌드 시 기존 torch 사용)
PATH="/opt/conda/envs/volume_est/bin:$PATH" \
conda run -n volume_est pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
# detectron2가 iopath를 낮출 수 있음 → SAM3 호환 버전으로 재고정
conda run -n volume_est pip install "iopath>=0.1.10"
# OpenWorldSAM 추가 의존성
conda run -n volume_est pip install hydra-core einops sentencepiece fairscale
# torchscale이 timm==0.4.12를 요구하지만 SAM3에 timm>=1.0.17 필요 → --no-deps로 설치
conda run -n volume_est pip install torchscale==0.2.0 --no-deps
# torchscale이 timm을 0.4.x로 다운그레이드함 → SAM3 호환 버전으로 재고정
conda run -n volume_est pip install "timm>=1.0.17"

echo ""
echo "=== 10. UniDepthV2 (depth + intrinsics 동시 추정) ==="
conda run -n volume_est pip install -e "$UNIDEPTH_ROOT" --no-deps
# UniDepth 런타임 의존성
conda run -n volume_est pip install h5py tabulate termcolor wandb xformers omegaconf

echo ""
echo "=== 11. WeasyPrint (retpdf 견적서 PDF 생성용) ==="
# Python 패키지: HTML → PDF 렌더러
conda run -n volume_est pip install weasyprint
# 시스템 라이브러리 (pango, libffi, gdk-pixbuf)를 conda-forge로 같은 env에 설치
# → 시스템 sudo 없이도 동작. 누락 시 weasyprint import 단계에서 OSError.
# font-ttf-noto-cjk: 한글/CJK 폰트. 누락 시 PDF에서 한글이 박스(豆腐)로 깨짐.
conda install -n volume_est -c conda-forge \
    pango libffi gdk-pixbuf font-ttf-noto-cjk -y

echo ""
echo "=== 12. libstdc++ 호환성 ==="
conda install -n volume_est -c conda-forge libstdcxx-ng -y
# conda activate 시 자동으로 LD_PRELOAD, LD_LIBRARY_PATH, CC, CXX 설정
mkdir -p /opt/conda/envs/volume_est/etc/conda/activate.d
cat > /opt/conda/envs/volume_est/etc/conda/activate.d/libstdcxx.sh << 'ACTIVATE_EOF'
export LD_PRELOAD=/opt/conda/envs/volume_est/lib/libstdc++.so.6
export _VOLUME_EST_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH=/opt/conda/envs/volume_est/lib/python3.11/site-packages/nvidia/cu13/lib:/opt/conda/envs/volume_est/lib:${LD_LIBRARY_PATH:-}
export CC=/opt/conda/envs/volume_est/bin/x86_64-conda-linux-gnu-gcc
export CXX=/opt/conda/envs/volume_est/bin/x86_64-conda-linux-gnu-g++
ACTIVATE_EOF
mkdir -p /opt/conda/envs/volume_est/etc/conda/deactivate.d
cat > /opt/conda/envs/volume_est/etc/conda/deactivate.d/libstdcxx.sh << 'DEACTIVATE_EOF'
unset LD_PRELOAD
export LD_LIBRARY_PATH="${_VOLUME_EST_OLD_LD_LIBRARY_PATH:-}"
unset _VOLUME_EST_OLD_LD_LIBRARY_PATH
unset CC
unset CXX
DEACTIVATE_EOF

echo ""
echo "=== 설치 확인 ==="
LD_PRELOAD=/opt/conda/envs/volume_est/lib/libstdc++.so.6 \
LD_LIBRARY_PATH=/opt/conda/envs/volume_est/lib/python3.11/site-packages/nvidia/cu13/lib:/opt/conda/envs/volume_est/lib:${LD_LIBRARY_PATH:-} \
conda run -n volume_est python -c "
import torch
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import transformers; print('Transformers:', transformers.__version__)
import sam3; print('SAM3: OK')
import detectron2; print('Detectron2:', detectron2.__version__)
from unidepth.models import UniDepthV2; print('UniDepthV2: OK')
import cv2; print('OpenCV:', cv2.__version__, '| INTER_AREA:', hasattr(cv2, 'INTER_AREA'))
import scipy; print('SciPy:', scipy.__version__)
import numpy; print('NumPy:', numpy.__version__)
import weasyprint
from weasyprint import HTML as _HTML
_HTML(string='<p>ok</p>').write_pdf()  # pango/cairo 시스템 라이브러리까지 검증
import subprocess as _sp
_ko_fonts = _sp.run(
    ['fc-list', ':lang=ko'], capture_output=True, text=True,
).stdout.strip().splitlines()
assert _ko_fonts, '한글 폰트가 fontconfig에 보이지 않습니다 (font-ttf-noto-cjk 누락 가능성).'
print('WeasyPrint:', weasyprint.__version__, '| 한글 폰트:', len(_ko_fonts), '개')
print()
print('All imports OK - 설치 완료!')
"

echo ""
echo "=== 사용법 ==="
echo "conda activate volume_est"
echo "cd $AGENT_ROOT"
echo "export SAM3_BPE_PATH=$SAM3_BPE_PATH"
echo ""
echo "# UniDepthV2 (depth + intrinsics 동시 추정, EXIF 불필요)"
echo 'python pipeline.py --image test_indoor.jpg --text "책상, 의자, 소파 옮겨줘" --output ./results_unidepth/'
