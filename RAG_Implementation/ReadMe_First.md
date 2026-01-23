
How to run the following Project
1. python -m venv venv
1. source venv/bin/activate
2. pip install -r requirements.txt
- Console output looks like this 
```
pip install -r requirements.txt 
Collecting PyMuPDF==1.23.26 (from -r requirements.txt (line 1))
  Using cached PyMuPDF-1.23.26-cp312-none-manylinux2014_x86_64.whl.metadata (3.4 kB)
Collecting matplotlib==3.8.3 (from -r requirements.txt (line 2))
  Using cached matplotlib-3.8.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.8 kB)
Collecting numpy==1.26.4 (from -r requirements.txt (line 3))
  Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting pandas==2.2.1 (from -r requirements.txt (line 4))
  Using cached pandas-2.2.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)
Collecting Requests==2.31.0 (from -r requirements.txt (line 5))
  Using cached requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Collecting sentence_transformers==2.5.1 (from -r requirements.txt (line 6))
  Using cached sentence_transformers-2.5.1-py3-none-any.whl.metadata (11 kB)
Collecting spacy (from -r requirements.txt (line 7))
  Using cached spacy-3.8.11-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (27 kB)
Collecting tqdm==4.66.2 (from -r requirements.txt (line 8))
  Using cached tqdm-4.66.2-py3-none-any.whl.metadata (57 kB)
Collecting transformers==4.38.2 (from -r requirements.txt (line 9))
  Using cached transformers-4.38.2-py3-none-any.whl.metadata (130 kB)
Collecting accelerate (from -r requirements.txt (line 10))
  Using cached accelerate-1.12.0-py3-none-any.whl.metadata (19 kB)
Collecting bitsandbytes (from -r requirements.txt (line 11))
  Using cached bitsandbytes-0.49.1-py3-none-manylinux_2_24_x86_64.whl.metadata (10 kB)
Collecting jupyter (from -r requirements.txt (line 12))
  Using cached jupyter-1.1.1-py2.py3-none-any.whl.metadata (2.0 kB)
Collecting wheel (from -r requirements.txt (line 13))
  Downloading wheel-0.46.2-py3-none-any.whl.metadata (2.4 kB)
Collecting PyMuPDFb==1.23.22 (from PyMuPDF==1.23.26->-r requirements.txt (line 1))
  Using cached PyMuPDFb-1.23.22-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.4 kB)
Collecting contourpy>=1.0.1 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (114 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Collecting packaging>=20.0 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Downloading packaging-26.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pillow>=8 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached pillow-12.1.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting pyparsing>=2.3.1 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Downloading pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
Collecting python-dateutil>=2.7 (from matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas==2.2.1->-r requirements.txt (line 4))
  Using cached pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas==2.2.1->-r requirements.txt (line 4))
  Using cached tzdata-2025.3-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting charset-normalizer<4,>=2 (from Requests==2.31.0->-r requirements.txt (line 5))
  Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Collecting idna<4,>=2.5 (from Requests==2.31.0->-r requirements.txt (line 5))
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from Requests==2.31.0->-r requirements.txt (line 5))
  Using cached urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2017.4.17 (from Requests==2.31.0->-r requirements.txt (line 5))
  Using cached certifi-2026.1.4-py3-none-any.whl.metadata (2.5 kB)
Collecting torch>=1.11.0 (from sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Downloading torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (31 kB)
Collecting scikit-learn (from sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (11 kB)
Collecting scipy (from sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached scipy-1.17.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting huggingface-hub>=0.15.1 (from sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached huggingface_hub-1.3.2-py3-none-any.whl.metadata (13 kB)
Collecting filelock (from transformers==4.38.2->-r requirements.txt (line 9))
  Using cached filelock-3.20.3-py3-none-any.whl.metadata (2.1 kB)
Collecting huggingface-hub>=0.15.1 (from sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
Collecting pyyaml>=5.1 (from transformers==4.38.2->-r requirements.txt (line 9))
  Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting regex!=2019.12.17 (from transformers==4.38.2->-r requirements.txt (line 9))
  Using cached regex-2026.1.15-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting tokenizers<0.19,>=0.14 (from transformers==4.38.2->-r requirements.txt (line 9))
  Using cached tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Collecting safetensors>=0.4.1 (from transformers==4.38.2->-r requirements.txt (line 9))
  Using cached safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting spacy-legacy<3.1.0,>=3.0.11 (from spacy->-r requirements.txt (line 7))
  Using cached spacy_legacy-3.0.12-py2.py3-none-any.whl.metadata (2.8 kB)
Collecting spacy-loggers<2.0.0,>=1.0.0 (from spacy->-r requirements.txt (line 7))
  Using cached spacy_loggers-1.0.5-py3-none-any.whl.metadata (23 kB)
Collecting murmurhash<1.1.0,>=0.28.0 (from spacy->-r requirements.txt (line 7))
  Using cached murmurhash-1.0.15-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (2.3 kB)
Collecting cymem<2.1.0,>=2.0.2 (from spacy->-r requirements.txt (line 7))
  Using cached cymem-2.0.13-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (9.7 kB)
Collecting preshed<3.1.0,>=3.0.2 (from spacy->-r requirements.txt (line 7))
  Using cached preshed-3.0.12-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (2.5 kB)
Collecting thinc<8.4.0,>=8.3.4 (from spacy->-r requirements.txt (line 7))
  Using cached thinc-8.3.10-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (15 kB)
Collecting wasabi<1.2.0,>=0.9.1 (from spacy->-r requirements.txt (line 7))
  Using cached wasabi-1.1.3-py3-none-any.whl.metadata (28 kB)
Collecting srsly<3.0.0,>=2.4.3 (from spacy->-r requirements.txt (line 7))
  Using cached srsly-2.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
Collecting catalogue<2.1.0,>=2.0.6 (from spacy->-r requirements.txt (line 7))
  Using cached catalogue-2.0.10-py3-none-any.whl.metadata (14 kB)
Collecting weasel<0.5.0,>=0.4.2 (from spacy->-r requirements.txt (line 7))
  Using cached weasel-0.4.3-py3-none-any.whl.metadata (4.6 kB)
Collecting typer-slim<1.0.0,>=0.3.0 (from spacy->-r requirements.txt (line 7))
  Using cached typer_slim-0.21.1-py3-none-any.whl.metadata (16 kB)
Collecting pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 (from spacy->-r requirements.txt (line 7))
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
Collecting jinja2 (from spacy->-r requirements.txt (line 7))
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting setuptools (from spacy->-r requirements.txt (line 7))
  Downloading setuptools-80.10.1-py3-none-any.whl.metadata (6.7 kB)
Collecting psutil (from accelerate->-r requirements.txt (line 10))
  Using cached psutil-7.2.1-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl.metadata (22 kB)
Collecting notebook (from jupyter->-r requirements.txt (line 12))
  Using cached notebook-7.5.2-py3-none-any.whl.metadata (10 kB)
Collecting jupyter-console (from jupyter->-r requirements.txt (line 12))
  Using cached jupyter_console-6.6.3-py3-none-any.whl.metadata (5.8 kB)
Collecting nbconvert (from jupyter->-r requirements.txt (line 12))
  Using cached nbconvert-7.16.6-py3-none-any.whl.metadata (8.5 kB)
Collecting ipykernel (from jupyter->-r requirements.txt (line 12))
  Using cached ipykernel-7.1.0-py3-none-any.whl.metadata (4.5 kB)
Collecting ipywidgets (from jupyter->-r requirements.txt (line 12))
  Using cached ipywidgets-8.1.8-py3-none-any.whl.metadata (2.4 kB)
Collecting jupyterlab (from jupyter->-r requirements.txt (line 12))
  Using cached jupyterlab-4.5.2-py3-none-any.whl.metadata (16 kB)
Collecting fsspec>=2023.5.0 (from huggingface-hub>=0.15.1->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached fsspec-2026.1.0-py3-none-any.whl.metadata (10 kB)
Collecting typing-extensions>=3.7.4.3 (from huggingface-hub>=0.15.1->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting hf-xet<2.0.0,>=1.1.3 (from huggingface-hub>=0.15.1->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting annotated-types>=0.6.0 (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->-r requirements.txt (line 7))
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.41.5 (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->-r requirements.txt (line 7))
  Using cached pydantic_core-2.41.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting typing-inspection>=0.4.2 (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->-r requirements.txt (line 7))
  Using cached typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib==3.8.3->-r requirements.txt (line 2))
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting blis<1.4.0,>=1.3.0 (from thinc<8.4.0,>=8.3.4->spacy->-r requirements.txt (line 7))
  Using cached blis-1.3.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.5 kB)
Collecting confection<1.0.0,>=0.0.1 (from thinc<8.4.0,>=8.3.4->spacy->-r requirements.txt (line 7))
  Using cached confection-0.1.5-py3-none-any.whl.metadata (19 kB)
Collecting sympy>=1.13.3 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Collecting cuda-bindings==12.9.4 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Downloading cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (2.6 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cublas-cu12==12.8.4.1 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft-cu12==11.3.3.83 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand-cu12==10.3.9.90 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparselt-cu12==0.7.1 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
Collecting nvidia-nccl-cu12==2.27.5 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
Collecting nvidia-nvshmem-cu12==3.4.5 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Downloading nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvtx-cu12==12.8.90 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufile-cu12==1.13.1.3 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting triton==3.6.0 (from torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Downloading triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting cuda-pathfinder~=1.1 (from cuda-bindings==12.9.4->torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Downloading cuda_pathfinder-1.3.3-py3-none-any.whl.metadata (1.9 kB)
Collecting click>=8.0.0 (from typer-slim<1.0.0,>=0.3.0->spacy->-r requirements.txt (line 7))
  Using cached click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
Collecting cloudpathlib<1.0.0,>=0.7.0 (from weasel<0.5.0,>=0.4.2->spacy->-r requirements.txt (line 7))
  Using cached cloudpathlib-0.23.0-py3-none-any.whl.metadata (16 kB)
Collecting smart-open<8.0.0,>=5.2.1 (from weasel<0.5.0,>=0.4.2->spacy->-r requirements.txt (line 7))
  Using cached smart_open-7.5.0-py3-none-any.whl.metadata (24 kB)
Collecting comm>=0.1.1 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached comm-0.2.3-py3-none-any.whl.metadata (3.7 kB)
Collecting debugpy>=1.6.5 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached debugpy-1.8.19-cp312-cp312-manylinux_2_34_x86_64.whl.metadata (1.4 kB)
Collecting ipython>=7.23.1 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached ipython-9.9.0-py3-none-any.whl.metadata (4.6 kB)
Collecting jupyter-client>=8.0.0 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_client-8.8.0-py3-none-any.whl.metadata (8.4 kB)
Collecting jupyter-core!=5.0.*,>=4.12 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_core-5.9.1-py3-none-any.whl.metadata (1.5 kB)
Collecting matplotlib-inline>=0.1 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached matplotlib_inline-0.2.1-py3-none-any.whl.metadata (2.3 kB)
Collecting nest-asyncio>=1.4 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
Collecting pyzmq>=25 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached pyzmq-27.1.0-cp312-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl.metadata (6.0 kB)
Collecting tornado>=6.2 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached tornado-6.5.4-cp39-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.8 kB)
Collecting traitlets>=5.4.0 (from ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)
Collecting widgetsnbextension~=4.0.14 (from ipywidgets->jupyter->-r requirements.txt (line 12))
  Using cached widgetsnbextension-4.0.15-py3-none-any.whl.metadata (1.6 kB)
Collecting jupyterlab_widgets~=3.0.15 (from ipywidgets->jupyter->-r requirements.txt (line 12))
  Using cached jupyterlab_widgets-3.0.16-py3-none-any.whl.metadata (20 kB)
Collecting MarkupSafe>=2.0 (from jinja2->spacy->-r requirements.txt (line 7))
  Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting prompt-toolkit>=3.0.30 (from jupyter-console->jupyter->-r requirements.txt (line 12))
  Using cached prompt_toolkit-3.0.52-py3-none-any.whl.metadata (6.4 kB)
Collecting pygments (from jupyter-console->jupyter->-r requirements.txt (line 12))
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting async-lru>=1.0.0 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached async_lru-2.1.0-py3-none-any.whl.metadata (5.3 kB)
Collecting httpx<1,>=0.25.0 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting jupyter-lsp>=2.0.0 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_lsp-2.3.0-py3-none-any.whl.metadata (1.8 kB)
Collecting jupyter-server<3,>=2.4.0 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_server-2.17.0-py3-none-any.whl.metadata (8.5 kB)
Collecting jupyterlab-server<3,>=2.28.0 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jupyterlab_server-2.28.0-py3-none-any.whl.metadata (5.9 kB)
Collecting notebook-shim>=0.2 (from jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached notebook_shim-0.2.4-py3-none-any.whl.metadata (4.0 kB)
Collecting beautifulsoup4 (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached beautifulsoup4-4.14.3-py3-none-any.whl.metadata (3.8 kB)
Collecting bleach!=5.0.0 (from bleach[css]!=5.0.0->nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached bleach-6.3.0-py3-none-any.whl.metadata (31 kB)
Collecting defusedxml (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)
Collecting jupyterlab-pygments (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached jupyterlab_pygments-0.3.0-py3-none-any.whl.metadata (4.4 kB)
Collecting mistune<4,>=2.0.3 (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached mistune-3.2.0-py3-none-any.whl.metadata (1.9 kB)
Collecting nbclient>=0.5.0 (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached nbclient-0.10.4-py3-none-any.whl.metadata (8.3 kB)
Collecting nbformat>=5.7 (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached nbformat-5.10.4-py3-none-any.whl.metadata (3.6 kB)
Collecting pandocfilters>=1.4.1 (from nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached pandocfilters-1.5.1-py2.py3-none-any.whl.metadata (9.0 kB)
Collecting joblib>=1.3.0 (from scikit-learn->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting threadpoolctl>=3.2.0 (from scikit-learn->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting webencodings (from bleach!=5.0.0->bleach[css]!=5.0.0->nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
Collecting tinycss2<1.5,>=1.1.0 (from bleach[css]!=5.0.0->nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached tinycss2-1.4.0-py3-none-any.whl.metadata (3.0 kB)
Collecting anyio (from httpx<1,>=0.25.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached anyio-4.12.1-py3-none-any.whl.metadata (4.3 kB)
Collecting httpcore==1.* (from httpx<1,>=0.25.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.25.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting decorator>=4.3.2 (from ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached decorator-5.2.1-py3-none-any.whl.metadata (3.9 kB)
Collecting ipython-pygments-lexers>=1.0.0 (from ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl.metadata (1.1 kB)
Collecting jedi>=0.18.1 (from ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting pexpect>4.3 (from ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached pexpect-4.9.0-py2.py3-none-any.whl.metadata (2.5 kB)
Collecting stack_data>=0.6.0 (from ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)
Collecting platformdirs>=2.5 (from jupyter-core!=5.0.*,>=4.12->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached platformdirs-4.5.1-py3-none-any.whl.metadata (12 kB)
Collecting argon2-cffi>=21.1 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached argon2_cffi-25.1.0-py3-none-any.whl.metadata (4.1 kB)
Collecting jupyter-events>=0.11.0 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_events-0.12.0-py3-none-any.whl.metadata (5.8 kB)
Collecting jupyter-server-terminals>=0.4.4 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jupyter_server_terminals-0.5.4-py3-none-any.whl.metadata (5.9 kB)
Collecting prometheus-client>=0.9 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached prometheus_client-0.24.1-py3-none-any.whl.metadata (2.1 kB)
Collecting send2trash>=1.8.2 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached send2trash-2.1.0-py3-none-any.whl.metadata (4.1 kB)
Collecting terminado>=0.8.3 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached terminado-0.18.1-py3-none-any.whl.metadata (5.8 kB)
Collecting websocket-client>=1.7 (from jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached websocket_client-1.9.0-py3-none-any.whl.metadata (8.3 kB)
Collecting babel>=2.10 (from jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached babel-2.17.0-py3-none-any.whl.metadata (2.0 kB)
Collecting json5>=0.9.0 (from jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached json5-0.13.0-py3-none-any.whl.metadata (36 kB)
Collecting jsonschema>=4.18.0 (from jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jsonschema-4.26.0-py3-none-any.whl.metadata (7.6 kB)
Collecting fastjsonschema>=2.15 (from nbformat>=5.7->nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached fastjsonschema-2.21.2-py3-none-any.whl.metadata (2.3 kB)
Collecting wcwidth (from prompt-toolkit>=3.0.30->jupyter-console->jupyter->-r requirements.txt (line 12))
  Downloading wcwidth-0.3.0-py3-none-any.whl.metadata (25 kB)
Collecting wrapt (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.4.2->spacy->-r requirements.txt (line 7))
  Using cached wrapt-2.0.1-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (9.0 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch>=1.11.0->sentence_transformers==2.5.1->-r requirements.txt (line 6))
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting soupsieve>=1.6.1 (from beautifulsoup4->nbconvert->jupyter->-r requirements.txt (line 12))
  Using cached soupsieve-2.8.3-py3-none-any.whl.metadata (4.6 kB)
Collecting argon2-cffi-bindings (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached argon2_cffi_bindings-25.1.0-cp39-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl.metadata (7.4 kB)
Collecting parso<0.9.0,>=0.8.4 (from jedi>=0.18.1->ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached parso-0.8.5-py2.py3-none-any.whl.metadata (8.3 kB)
Collecting attrs>=22.2.0 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached attrs-25.4.0-py3-none-any.whl.metadata (10 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jsonschema_specifications-2025.9.1-py3-none-any.whl.metadata (2.9 kB)
Collecting referencing>=0.28.4 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached referencing-0.37.0-py3-none-any.whl.metadata (2.8 kB)
Collecting rpds-py>=0.25.0 (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.28.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached rpds_py-0.30.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting python-json-logger>=2.0.4 (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached python_json_logger-4.0.0-py3-none-any.whl.metadata (4.0 kB)
Collecting rfc3339-validator (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached rfc3339_validator-0.1.4-py2.py3-none-any.whl.metadata (1.5 kB)
Collecting rfc3986-validator>=0.1.1 (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached rfc3986_validator-0.1.1-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting ptyprocess>=0.5 (from pexpect>4.3->ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached ptyprocess-0.7.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting executing>=1.2.0 (from stack_data>=0.6.0->ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached executing-2.2.1-py2.py3-none-any.whl.metadata (8.9 kB)
Collecting asttokens>=2.1.0 (from stack_data>=0.6.0->ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached asttokens-3.0.1-py3-none-any.whl.metadata (4.9 kB)
Collecting pure-eval (from stack_data>=0.6.0->ipython>=7.23.1->ipykernel->jupyter->-r requirements.txt (line 12))
  Using cached pure_eval-0.2.3-py3-none-any.whl.metadata (6.3 kB)
Collecting fqdn (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached fqdn-1.5.1-py3-none-any.whl.metadata (1.4 kB)
Collecting isoduration (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached isoduration-20.11.0-py3-none-any.whl.metadata (5.7 kB)
Collecting jsonpointer>1.13 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached jsonpointer-3.0.0-py2.py3-none-any.whl.metadata (2.3 kB)
Collecting rfc3987-syntax>=1.1.0 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached rfc3987_syntax-1.1.0-py3-none-any.whl.metadata (7.7 kB)
Collecting uri-template (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached uri_template-1.3.0-py3-none-any.whl.metadata (8.8 kB)
Collecting webcolors>=24.6.0 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached webcolors-25.10.0-py3-none-any.whl.metadata (2.2 kB)
Collecting cffi>=1.0.1 (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached cffi-2.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.6 kB)
Collecting pycparser (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Downloading pycparser-3.0-py3-none-any.whl.metadata (8.2 kB)
Collecting lark>=1.2.2 (from rfc3987-syntax>=1.1.0->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached lark-1.3.1-py3-none-any.whl.metadata (1.8 kB)
Collecting arrow>=0.15.0 (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->jupyter->-r requirements.txt (line 12))
  Using cached arrow-1.4.0-py3-none-any.whl.metadata (7.7 kB)
Using cached PyMuPDF-1.23.26-cp312-none-manylinux2014_x86_64.whl (4.4 MB)
Using cached matplotlib-3.8.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.6 MB)
Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
Using cached pandas-2.2.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.7 MB)
Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Using cached sentence_transformers-2.5.1-py3-none-any.whl (156 kB)
Using cached tqdm-4.66.2-py3-none-any.whl (78 kB)
Using cached transformers-4.38.2-py3-none-any.whl (8.5 MB)
Using cached PyMuPDFb-1.23.22-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (30.6 MB)
Using cached spacy-3.8.11-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (33.2 MB)
Using cached accelerate-1.12.0-py3-none-any.whl (380 kB)
Using cached bitsandbytes-0.49.1-py3-none-manylinux_2_24_x86_64.whl (59.1 MB)
Using cached jupyter-1.1.1-py2.py3-none-any.whl (2.7 kB)
Downloading wheel-0.46.2-py3-none-any.whl (29 kB)
Using cached catalogue-2.0.10-py3-none-any.whl (17 kB)
Using cached certifi-2026.1.4-py3-none-any.whl (152 kB)
Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (153 kB)
Using cached contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Using cached cymem-2.0.13-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (260 kB)
Using cached fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (5.0 MB)
Using cached huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
Using cached murmurhash-1.0.15-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (134 kB)
Downloading packaging-26.0-py3-none-any.whl (74 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.4/74.4 kB 2.8 MB/s eta 0:00:00
Using cached pillow-12.1.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.0 MB)
Using cached preshed-3.0.12-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (874 kB)
Using cached pydantic-2.12.5-py3-none-any.whl (463 kB)
Using cached pydantic_core-2.41.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
Downloading pyparsing-3.3.2-py3-none-any.whl (122 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 122.8/122.8 kB 3.8 MB/s eta 0:00:00
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached pytz-2025.2-py2.py3-none-any.whl (509 kB)
Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (807 kB)
Using cached regex-2026.1.15-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (803 kB)
Using cached safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (507 kB)
Using cached spacy_legacy-3.0.12-py2.py3-none-any.whl (29 kB)
Using cached spacy_loggers-1.0.5-py3-none-any.whl (22 kB)
Using cached srsly-2.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
Using cached thinc-8.3.10-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.9 MB)
Using cached tokenizers-0.15.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
Downloading torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl (915.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 915.7/915.7 MB 675.2 kB/s eta 0:00:00
Downloading cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.2/12.2 MB 849.8 kB/s eta 0:00:00
Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl (287.2 MB)
Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.3 MB)
Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
Downloading nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (139.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.1/139.1 MB 822.2 kB/s eta 0:00:00
Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Downloading triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (188.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 188.3/188.3 MB 833.2 kB/s eta 0:00:00
Using cached typer_slim-0.21.1-py3-none-any.whl (47 kB)
Using cached tzdata-2025.3-py2.py3-none-any.whl (348 kB)
Using cached urllib3-2.6.3-py3-none-any.whl (131 kB)
Using cached wasabi-1.1.3-py3-none-any.whl (27 kB)
Using cached weasel-0.4.3-py3-none-any.whl (50 kB)
Using cached filelock-3.20.3-py3-none-any.whl (16 kB)
Using cached ipykernel-7.1.0-py3-none-any.whl (117 kB)
Using cached psutil-7.2.1-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl (154 kB)
Using cached ipywidgets-8.1.8-py3-none-any.whl (139 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached jupyter_console-6.6.3-py3-none-any.whl (24 kB)
Using cached jupyterlab-4.5.2-py3-none-any.whl (12.4 MB)
Downloading setuptools-80.10.1-py3-none-any.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 459.4 kB/s eta 0:00:00
Using cached nbconvert-7.16.6-py3-none-any.whl (258 kB)
Using cached notebook-7.5.2-py3-none-any.whl (14.5 MB)
Using cached scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (8.9 MB)
Using cached scipy-1.17.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.0 MB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached async_lru-2.1.0-py3-none-any.whl (6.9 kB)
Using cached bleach-6.3.0-py3-none-any.whl (164 kB)
Using cached blis-1.3.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (11.4 MB)
Using cached click-8.3.1-py3-none-any.whl (108 kB)
Using cached cloudpathlib-0.23.0-py3-none-any.whl (62 kB)
Using cached comm-0.2.3-py3-none-any.whl (7.3 kB)
Using cached confection-0.1.5-py3-none-any.whl (35 kB)
Using cached debugpy-1.8.19-cp312-cp312-manylinux_2_34_x86_64.whl (4.3 MB)
Using cached fsspec-2026.1.0-py3-none-any.whl (201 kB)
Using cached hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Using cached httpcore-1.0.9-py3-none-any.whl (78 kB)
Using cached ipython-9.9.0-py3-none-any.whl (621 kB)
Using cached joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached jupyter_client-8.8.0-py3-none-any.whl (107 kB)
Using cached jupyter_core-5.9.1-py3-none-any.whl (29 kB)
Using cached jupyter_lsp-2.3.0-py3-none-any.whl (76 kB)
Using cached jupyter_server-2.17.0-py3-none-any.whl (388 kB)
Using cached jupyterlab_server-2.28.0-py3-none-any.whl (59 kB)
Using cached jupyterlab_widgets-3.0.16-py3-none-any.whl (914 kB)
Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Using cached matplotlib_inline-0.2.1-py3-none-any.whl (9.5 kB)
Using cached mistune-3.2.0-py3-none-any.whl (53 kB)
Using cached nbclient-0.10.4-py3-none-any.whl (25 kB)
Using cached nbformat-5.10.4-py3-none-any.whl (78 kB)
Using cached nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Using cached networkx-3.6.1-py3-none-any.whl (2.1 MB)
Using cached notebook_shim-0.2.4-py3-none-any.whl (13 kB)
Using cached pandocfilters-1.5.1-py2.py3-none-any.whl (8.7 kB)
Using cached prompt_toolkit-3.0.52-py3-none-any.whl (391 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Using cached pyzmq-27.1.0-cp312-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl (840 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached smart_open-7.5.0-py3-none-any.whl (63 kB)
Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Using cached tornado-6.5.4-cp39-abi3-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (445 kB)
Using cached traitlets-5.14.3-py3-none-any.whl (85 kB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Using cached widgetsnbextension-4.0.15-py3-none-any.whl (2.2 MB)
Using cached beautifulsoup4-4.14.3-py3-none-any.whl (107 kB)
Using cached defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
Using cached jupyterlab_pygments-0.3.0-py3-none-any.whl (15 kB)
Using cached anyio-4.12.1-py3-none-any.whl (113 kB)
Using cached argon2_cffi-25.1.0-py3-none-any.whl (14 kB)
Using cached babel-2.17.0-py3-none-any.whl (10.2 MB)
Downloading cuda_pathfinder-1.3.3-py3-none-any.whl (27 kB)
Using cached decorator-5.2.1-py3-none-any.whl (9.2 kB)
Using cached fastjsonschema-2.21.2-py3-none-any.whl (24 kB)
Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl (8.1 kB)
Using cached jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
Using cached json5-0.13.0-py3-none-any.whl (36 kB)
Using cached jsonschema-4.26.0-py3-none-any.whl (90 kB)
Using cached jupyter_events-0.12.0-py3-none-any.whl (19 kB)
Using cached jupyter_server_terminals-0.5.4-py3-none-any.whl (13 kB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Using cached pexpect-4.9.0-py2.py3-none-any.whl (63 kB)
Using cached platformdirs-4.5.1-py3-none-any.whl (18 kB)
Using cached prometheus_client-0.24.1-py3-none-any.whl (64 kB)
Using cached send2trash-2.1.0-py3-none-any.whl (17 kB)
Using cached soupsieve-2.8.3-py3-none-any.whl (37 kB)
Using cached stack_data-0.6.3-py3-none-any.whl (24 kB)
Using cached terminado-0.18.1-py3-none-any.whl (14 kB)
Using cached tinycss2-1.4.0-py3-none-any.whl (26 kB)
Using cached webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
Using cached websocket_client-1.9.0-py3-none-any.whl (82 kB)
Downloading wcwidth-0.3.0-py3-none-any.whl (85 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.5/85.5 kB 574.7 kB/s eta 0:00:00
Using cached wrapt-2.0.1-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (121 kB)
Using cached asttokens-3.0.1-py3-none-any.whl (27 kB)
Using cached attrs-25.4.0-py3-none-any.whl (67 kB)
Using cached executing-2.2.1-py2.py3-none-any.whl (28 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Using cached jsonschema_specifications-2025.9.1-py3-none-any.whl (18 kB)
Using cached parso-0.8.5-py2.py3-none-any.whl (106 kB)
Using cached ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
Using cached python_json_logger-4.0.0-py3-none-any.whl (15 kB)
Using cached referencing-0.37.0-py3-none-any.whl (26 kB)
Using cached rfc3986_validator-0.1.1-py2.py3-none-any.whl (4.2 kB)
Using cached rpds_py-0.30.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (394 kB)
Using cached argon2_cffi_bindings-25.1.0-cp39-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl (87 kB)
Using cached pure_eval-0.2.3-py3-none-any.whl (11 kB)
Using cached rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)
Using cached cffi-2.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (219 kB)
Using cached jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)
Using cached rfc3987_syntax-1.1.0-py3-none-any.whl (8.0 kB)
Using cached webcolors-25.10.0-py3-none-any.whl (14 kB)
Using cached fqdn-1.5.1-py3-none-any.whl (9.1 kB)
Using cached isoduration-20.11.0-py3-none-any.whl (11 kB)
Using cached uri_template-1.3.0-py3-none-any.whl (11 kB)
Using cached arrow-1.4.0-py3-none-any.whl (68 kB)
Using cached lark-1.3.1-py3-none-any.whl (113 kB)
Downloading pycparser-3.0-py3-none-any.whl (48 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.2/48.2 kB 316.9 kB/s eta 0:00:00
Installing collected packages: webencodings, pytz, pure-eval, ptyprocess, nvidia-cusparselt-cu12, mpmath, fastjsonschema, wrapt, widgetsnbextension, websocket-client, webcolors, wcwidth, wasabi, urllib3, uri-template, tzdata, typing-extensions, triton, traitlets, tqdm, tornado, tinycss2, threadpoolctl, sympy, spacy-loggers, spacy-legacy, soupsieve, six, setuptools, send2trash, safetensors, rpds-py, rfc3986-validator, regex, pyzmq, pyyaml, python-json-logger, pyparsing, PyMuPDFb, pygments, pycparser, psutil, prometheus-client, platformdirs, pillow, pexpect, parso, pandocfilters, packaging, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, nest-asyncio, murmurhash, mistune, MarkupSafe, lark, kiwisolver, jupyterlab_widgets, jupyterlab-pygments, jsonpointer, json5, joblib, idna, hf-xet, h11, fsspec, fqdn, fonttools, filelock, executing, defusedxml, decorator, debugpy, cymem, cycler, cuda-pathfinder, comm, cloudpathlib, click, charset-normalizer, certifi, catalogue, bleach, babel, attrs, async-lru, asttokens, annotated-types, wheel, typing-inspection, typer-slim, terminado, stack_data, srsly, smart-open, scipy, rfc3987-syntax, rfc3339-validator, Requests, referencing, python-dateutil, PyMuPDF, pydantic-core, prompt-toolkit, preshed, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, matplotlib-inline, jupyter-core, jinja2, jedi, ipython-pygments-lexers, httpcore, cuda-bindings, contourpy, cffi, blis, beautifulsoup4, anyio, scikit-learn, pydantic, pandas, nvidia-cusolver-cu12, matplotlib, jupyter-server-terminals, jupyter-client, jsonschema-specifications, ipython, huggingface-hub, httpx, arrow, argon2-cffi-bindings, torch, tokenizers, jsonschema, isoduration, ipywidgets, ipykernel, confection, argon2-cffi, weasel, transformers, thinc, nbformat, jupyter-console, bitsandbytes, accelerate, spacy, sentence_transformers, nbclient, jupyter-events, nbconvert, jupyter-server, notebook-shim, jupyterlab-server, jupyter-lsp, jupyterlab, notebook, jupyter
Successfully installed MarkupSafe-3.0.3 PyMuPDF-1.23.26 PyMuPDFb-1.23.22 Requests-2.31.0 accelerate-1.12.0 annotated-types-0.7.0 anyio-4.12.1 argon2-cffi-25.1.0 argon2-cffi-bindings-25.1.0 arrow-1.4.0 asttokens-3.0.1 async-lru-2.1.0 attrs-25.4.0 babel-2.17.0 beautifulsoup4-4.14.3 bitsandbytes-0.49.1 bleach-6.3.0 blis-1.3.3 catalogue-2.0.10 certifi-2026.1.4 cffi-2.0.0 charset-normalizer-3.4.4 click-8.3.1 cloudpathlib-0.23.0 comm-0.2.3 confection-0.1.5 contourpy-1.3.3 cuda-bindings-12.9.4 cuda-pathfinder-1.3.3 cycler-0.12.1 cymem-2.0.13 debugpy-1.8.19 decorator-5.2.1 defusedxml-0.7.1 executing-2.2.1 fastjsonschema-2.21.2 filelock-3.20.3 fonttools-4.61.1 fqdn-1.5.1 fsspec-2026.1.0 h11-0.16.0 hf-xet-1.2.0 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-0.36.0 idna-3.11 ipykernel-7.1.0 ipython-9.9.0 ipython-pygments-lexers-1.1.1 ipywidgets-8.1.8 isoduration-20.11.0 jedi-0.19.2 jinja2-3.1.6 joblib-1.5.3 json5-0.13.0 jsonpointer-3.0.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 jupyter-1.1.1 jupyter-client-8.8.0 jupyter-console-6.6.3 jupyter-core-5.9.1 jupyter-events-0.12.0 jupyter-lsp-2.3.0 jupyter-server-2.17.0 jupyter-server-terminals-0.5.4 jupyterlab-4.5.2 jupyterlab-pygments-0.3.0 jupyterlab-server-2.28.0 jupyterlab_widgets-3.0.16 kiwisolver-1.4.9 lark-1.3.1 matplotlib-3.8.3 matplotlib-inline-0.2.1 mistune-3.2.0 mpmath-1.3.0 murmurhash-1.0.15 nbclient-0.10.4 nbconvert-7.16.6 nbformat-5.10.4 nest-asyncio-1.6.0 networkx-3.6.1 notebook-7.5.2 notebook-shim-0.2.4 numpy-1.26.4 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.4.5 nvidia-nvtx-cu12-12.8.90 packaging-26.0 pandas-2.2.1 pandocfilters-1.5.1 parso-0.8.5 pexpect-4.9.0 pillow-12.1.0 platformdirs-4.5.1 preshed-3.0.12 prometheus-client-0.24.1 prompt-toolkit-3.0.52 psutil-7.2.1 ptyprocess-0.7.0 pure-eval-0.2.3 pycparser-3.0 pydantic-2.12.5 pydantic-core-2.41.5 pygments-2.19.2 pyparsing-3.3.2 python-dateutil-2.9.0.post0 python-json-logger-4.0.0 pytz-2025.2 pyyaml-6.0.3 pyzmq-27.1.0 referencing-0.37.0 regex-2026.1.15 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rfc3987-syntax-1.1.0 rpds-py-0.30.0 safetensors-0.7.0 scikit-learn-1.8.0 scipy-1.17.0 send2trash-2.1.0 sentence_transformers-2.5.1 setuptools-80.10.1 six-1.17.0 smart-open-7.5.0 soupsieve-2.8.3 spacy-3.8.11 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.5.2 stack_data-0.6.3 sympy-1.14.0 terminado-0.18.1 thinc-8.3.10 threadpoolctl-3.6.0 tinycss2-1.4.0 tokenizers-0.15.2 torch-2.10.0 tornado-6.5.4 tqdm-4.66.2 traitlets-5.14.3 transformers-4.38.2 triton-3.6.0 typer-slim-0.21.1 typing-extensions-4.15.0 typing-inspection-0.4.2 tzdata-2025.3 uri-template-1.3.0 urllib3-2.6.3 wasabi-1.1.3 wcwidth-0.3.0 weasel-0.4.3 webcolors-25.10.0 webencodings-0.5.1 websocket-client-1.9.0 wheel-0.46.2 widgetsnbextension-4.0.15 wrapt-2.0.1
```
2. python3 PdfParserForGemma.py
- Console output 
```
Reading pages: 1208it [00:01, 749.98it/s]
Sentence segmentation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1208/1208 [00:01<00:00, 703.24it/s]
Chunking sentences: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1208/1208 [00:00<00:00, 28752.40it/s]
/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Batches:  11%|███████████████████████████████████████████████▏                                                                                                                                                                                                                                                                                                                                          Batches:  15%|██████████████████████████████████████████████████████████████▉                                                                                                                                                                                                                                                                                                                           Batches:  19%|██████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                                                                                                                           Batches:  22%|██████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                                                                                                           Batches:  26%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                                                                                           Batches:  30%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                                                                            Batches:  33%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                                                            Batches:  37%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                                                             Batches:  41%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                             Batches:  44%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                             Batches:  48%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                             Batches:  52%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                              Batches:  56%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                              Batches:  59%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                              Batches:  63%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                               Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [01:19<00:00,  2.96s/it]
   page_number                                     sentence_chunk  chunk_char_count  chunk_word_count  chunk_token_count                                          embedding
0          -39  Human Nutrition: 2020 Edition UNIVERSITY OF HA...               308                42              77.00  tensor([ 6.7424e-02,  9.0228e-02, -5.0955e-03,...
1          -38  Human Nutrition: 2020 Edition by University of...               210                30              52.50  tensor([ 5.5216e-02,  5.9214e-02, -1.6617e-02,...
2          -37  Contents Preface University of Hawai‘i at Māno...               766               116             191.50  tensor([ 2.7980e-02,  3.3981e-02, -2.0643e-02,...
3          -36  Lifestyles and Nutrition University of Hawai‘i...               941               144             235.25  tensor([ 6.8257e-02,  3.8127e-02, -8.4686e-03,...
4          -35  The Cardiovascular System University of Hawai‘...               998               152             249.50  tensor([ 3.3026e-02, -8.4977e-03,  9.5716e-03,...
```

> for the warning encountered in above example 

```
(venv) prichai@CMTCLXX86213450:~/AI_ML/Group_I_Project-/RAG_Implementation$ pip install --upgrade huggingface_hub
Requirement already satisfied: huggingface_hub in ./venv/lib/python3.12/site-packages (0.36.0)
Collecting huggingface_hub
  Using cached huggingface_hub-1.3.2-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: filelock in ./venv/lib/python3.12/site-packages (from huggingface_hub) (3.20.3)
Requirement already satisfied: fsspec>=2023.5.0 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (2026.1.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (1.2.0)
Requirement already satisfied: httpx<1,>=0.23.0 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (0.28.1)
Requirement already satisfied: packaging>=20.9 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (26.0)
Requirement already satisfied: pyyaml>=5.1 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (6.0.3)
Collecting shellingham (from huggingface_hub)
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Requirement already satisfied: tqdm>=4.42.1 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (4.66.2)
Requirement already satisfied: typer-slim in ./venv/lib/python3.12/site-packages (from huggingface_hub) (0.21.1)
Requirement already satisfied: typing-extensions>=4.1.0 in ./venv/lib/python3.12/site-packages (from huggingface_hub) (4.15.0)
Requirement already satisfied: anyio in ./venv/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (4.12.1)
Requirement already satisfied: certifi in ./venv/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (2026.1.4)
Requirement already satisfied: httpcore==1.* in ./venv/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (1.0.9)
Requirement already satisfied: idna in ./venv/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (3.11)
Requirement already satisfied: h11>=0.16 in ./venv/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub) (0.16.0)
Requirement already satisfied: click>=8.0.0 in ./venv/lib/python3.12/site-packages (from typer-slim->huggingface_hub) (8.3.1)
Using cached huggingface_hub-1.3.2-py3-none-any.whl (534 kB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Installing collected packages: shellingham, huggingface_hub
  Attempting uninstall: huggingface_hub
    Found existing installation: huggingface-hub 0.36.0
    Uninstalling huggingface-hub-0.36.0:
      Successfully uninstalled huggingface-hub-0.36.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tokenizers 0.15.2 requires huggingface_hub<1.0,>=0.16.4, but you have huggingface-hub 1.3.2 which is incompatible.
transformers 4.38.2 requires huggingface-hub<1.0,>=0.19.3, but you have huggingface-hub 1.3.2 which is incompatible.
Successfully installed huggingface_hub-1.3.2 shellingham-1.5.4
(venv) prichai@CMTCLXX86213450:~/AI_ML/Group_I_Project-/RAG_Implementation$ python3 PdfParserForGemma.py 
Traceback (most recent call last):
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/PdfParserForGemma.py", line 9, in <module>
    from sentence_transformers import SentenceTransformer
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/sentence_transformers/__init__.py", line 3, in <module>
    from .datasets import SentencesDataset, ParallelSentencesDataset
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/sentence_transformers/datasets/__init__.py", line 1, in <module>
    from .DenoisingAutoEncoderDataset import DenoisingAutoEncoderDataset
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/sentence_transformers/datasets/DenoisingAutoEncoderDataset.py", line 5, in <module>
    from transformers.utils.import_utils import is_nltk_available, NLTK_IMPORT_ERROR
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/transformers/__init__.py", line 26, in <module>
    from . import dependency_versions_check
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/transformers/dependency_versions_check.py", line 57, in <module>
    require_version_core(deps[pkg])
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/transformers/utils/versions.py", line 117, in require_version_core
    return require_version(requirement, hint)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/transformers/utils/versions.py", line 111, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/home/prichai/AI_ML/Group_I_Project-/RAG_Implementation/venv/lib/python3.12/site-packages/transformers/utils/versions.py", line 44, in _compare_versions
    raise ImportError(
ImportError: huggingface-hub>=0.19.3,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.3.2.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main
(venv) prichai@CMTCLXX86213450:~/AI_ML/Group_I_Project-/RAG_Implementation$ pip install transformers -U
Requirement already satisfied: transformers in ./venv/lib/python3.12/site-packages (4.38.2)
Collecting transformers
  Using cached transformers-4.57.6-py3-none-any.whl.metadata (43 kB)
Requirement already satisfied: filelock in ./venv/lib/python3.12/site-packages (from transformers) (3.20.3)
Collecting huggingface-hub<1.0,>=0.34.0 (from transformers)
  Using cached huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: numpy>=1.17 in ./venv/lib/python3.12/site-packages (from transformers) (1.26.4)
Requirement already satisfied: packaging>=20.0 in ./venv/lib/python3.12/site-packages (from transformers) (26.0)
Requirement already satisfied: pyyaml>=5.1 in ./venv/lib/python3.12/site-packages (from transformers) (6.0.3)
Requirement already satisfied: regex!=2019.12.17 in ./venv/lib/python3.12/site-packages (from transformers) (2026.1.15)
Requirement already satisfied: requests in ./venv/lib/python3.12/site-packages (from transformers) (2.31.0)
Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers)
  Using cached tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Requirement already satisfied: safetensors>=0.4.3 in ./venv/lib/python3.12/site-packages (from transformers) (0.7.0)
Requirement already satisfied: tqdm>=4.27 in ./venv/lib/python3.12/site-packages (from transformers) (4.66.2)
Requirement already satisfied: fsspec>=2023.5.0 in ./venv/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (2026.1.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in ./venv/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (4.15.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in ./venv/lib/python3.12/site-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (1.2.0)
Requirement already satisfied: charset-normalizer<4,>=2 in ./venv/lib/python3.12/site-packages (from requests->transformers) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in ./venv/lib/python3.12/site-packages (from requests->transformers) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./venv/lib/python3.12/site-packages (from requests->transformers) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in ./venv/lib/python3.12/site-packages (from requests->transformers) (2026.1.4)
Using cached transformers-4.57.6-py3-none-any.whl (12.0 MB)
Using cached huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
Using cached tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Installing collected packages: huggingface-hub, tokenizers, transformers
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface_hub 1.3.2
    Uninstalling huggingface_hub-1.3.2:
      Successfully uninstalled huggingface_hub-1.3.2
  Attempting uninstall: tokenizers
    Found existing installation: tokenizers 0.15.2
    Uninstalling tokenizers-0.15.2:
      Successfully uninstalled tokenizers-0.15.2
  Attempting uninstall: transformers
    Found existing installation: transformers 4.38.2
    Uninstalling transformers-4.38.2:
      Successfully uninstalled transformers-4.38.2
Successfully installed huggingface-hub-0.36.0 tokenizers-0.22.2 transformers-4.57.6
```
3. python3 GenrateOP.py

```
python3 GenrateOP.py 
[INFO] Using attention implementation: sdpa
Loaded metadata rows: 1680
Shape of loaded embeddings tensor: torch.Size([1680, 768])
   page_number                                     sentence_chunk  chunk_char_count  chunk_word_count  chunk_token_count                                          embedding
0          -39  Human Nutrition: 2020 Edition UNIVERSITY OF HA...               308                42              77.00  tensor([ 6.7424e-02,  9.0228e-02, -5.0955e-03,...
1          -38  Human Nutrition: 2020 Edition by University of...               210                30              52.50  tensor([ 5.5216e-02,  5.9214e-02, -1.6617e-02,...
2          -37  Contents Preface University of Hawai‘i at Māno...               766               116             191.50  tensor([ 2.7980e-02,  3.3981e-02, -2.0643e-02,...
3          -36  Lifestyles and Nutrition University of Hawai‘i...               941               144             235.25  tensor([ 6.8257e-02,  3.8127e-02, -8.4686e-03,...
4          -35  The Cardiovascular System University of Hawai‘...               998               152             249.50  tensor([ 3.3026e-02, -8.4977e-03,  9.5716e-03,...
Query: symptoms of pellagra
[INFO] Time taken to get scores on 1680 embeddings: 0.00031 seconds.
tensor([0.5000, 0.3741, 0.2959, 0.2793, 0.2721], device='cuda:0') tensor([ 822,  853, 1536, 1555, 1531], device='cuda:0')
[INFO] Time taken to get scores on 1680 embeddings: 0.00029 seconds.
Query: symptoms of pellagra

Results:
Score: 0.5000
Niacin deficiency is commonly known as pellagra and the symptoms include
fatigue, decreased appetite, and indigestion. These symptoms are then commonly
followed by the four D’s: diarrhea, dermatitis, dementia, and sometimes death.
Figure 9.12 Conversion of Tryptophan to Niacin Water-Soluble Vitamins | 565
Page number: 565


Score: 0.3741
car. Does it drive faster with a half-tank of gas or a full one?It does not
matter; the car drives just as fast as long as it has gas. Similarly, depletion
of B vitamins will cause problems in energy metabolism, but having more than is
required to run metabolism does not speed it up. Buyers of B-vitamin supplements
beware; B vitamins are not stored in the body and all excess will be flushed
down the toilet along with the extra money spent. B vitamins are naturally
present in numerous foods, and many other foods are enriched with them. In the
United States, B-vitamin deficiencies are rare; however in the nineteenth
century some vitamin-B deficiencies plagued many people in North America. Niacin
deficiency, also known as pellagra, was prominent in poorer Americans whose main
dietary staple was refined cornmeal. Its symptoms were severe and included
diarrhea, dermatitis, dementia, and even death. Some of the health consequences
of pellagra are the result of niacin being in insufficient supply to support the
body’s metabolic functions.
Page number: 591


Score: 0.2959
The carbon dioxide gas bubbles infiltrate the stretchy gluten, giving bread its
porosity and tenderness. For those who are sensitive to gluten, it is good to
know that corn, millet, buckwheat, and oats do not contain the proteins that
make gluten. However, some people who have celiac disease also may have a
response to products containing oats. This is most likely the result of cross-
contamination of grains during harvest, storage, packaging, and processing.
Celiac disease is most common in people of European descent and is rare in
people of African American, Japanese, and Chinese descent. It is much more
prevalent in women and in people with Type 1 diabetes, autoimmune thyroid
disease, and Down and Turner syndromes. Symptoms can range from mild to severe
and can include pale, fatty, loose stools, gastrointestinal upset, abdominal
pain, weight loss and, in children, a failure to grow and thrive. The symptoms
can appear in infancy or much later in life, even Nutrition, Health and Disease
| 1079
Page number: 1079


Score: 0.2793
Image by BruceBlaus/ CC BY 4.0 When the vertebral bone tissue is weakened, it
can cause the spine to curve. The increase in spine curvature not only causes
pain, but also decreases a person’s height. Curvature of the upper spine
produces what is called Dowager’s hump, also known as kyphosis. Severe upper-
spine deformity can compress the chest cavity and cause difficulty breathing. It
may also cause abdominal pain and loss of appetite because of the increased
pressure on the abdomen. 1090 | Nutrition, Health and Disease
Page number: 1090


Score: 0.2721
esophagus and cause irritation. It is estimated that GERD affects 25 to 35
percent of the US population. An analysis of several studies published in the
August 2005 issue of Annals of Internal Medicine concludes that GERD is much
more prevalent in people who are obese.1 The most common GERD symptom is
heartburn, but people with GERD may also experience regurgitation (flow of the
stomach’s acidic contents into the mouth), frequent coughing, and trouble
swallowing. There are other causative factors of GERD that may be separate from
or intertwined with obesity. The sphincter that separates the stomach’s internal
contents from the esophagus often does not function properly and acidic gastric
contents seep upward. Sometimes the peristaltic contractions of the esophagus
are also sluggish and compromise the clearance of acidic contents. In addition
to having an unbalanced, high-fat diet, some people with GERD are sensitive to
particular foods—chocolate, garlic, spicy foods, fried foods, and tomato-based
foods—which worsen symptoms. Drinks containing alcohol or caffeine may also
worsen GERD symptoms. GERD is diagnosed most often by a history of the frequency
of recurring symptoms. A more proper diagnosis can be made when a doctor inserts
a small device into the lower esophagus that measures the acidity of the
contents during one’s daily activities.
Page number: 1077


Highest score: 0.5000
Corresponding page number: 565
Best match is on page number: 565
Saved matched page as 'matched_page.png'
Available GPU memory: 16 GB
GPU memory: 16 | Recommended: Gemma 2B float16 or Gemma 7B in 4-bit.
use_quantization_config set to: False
model_id set to: google/gemma-2b-it
[INFO] Using model_id: google/gemma-2b-it
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 126.30it/s]
Input text:
What are the macronutrients, and what roles do they play in the human body?

Prompt (formatted):
<bos><start_of_turn>user
What are the macronutrients, and what roles do they play in the human body?<end_of_turn>
<start_of_turn>model

Model output (tokens):
tensor([     2,      2,    106,   1645,    108,   1841,    708,    573, 186809,
        184592, 235269,    578,   1212,  16065,    749,    984,   1554,    575,
           573,   3515,   2971, 235336,    107,    108,    106,   2516,    108,
         21404, 235269,   1517, 235303, 235256,    476,  25497,    576,    573,
        186809, 184592,    578,   1024,  16065,    575,    573,   3515,   2971,
        235292,    109,    688,  12298,   1695, 184592,  66058,    109, 235287,
          5231, 156615,  56227,  66058,    108,    141, 235287,  34428,   4134,
           604,    573,   2971, 235303, 235256,   5999,    578,  29703, 235265,
           108,    141, 235287, 110165,  56227,    708,    573,   7920,   4303,
           576,   4134,    604,   1546,   5999, 235265,    108,    141, 235287,
         25280,  72780,    708,   1941,    674,   1987,   5543,    577,  55997,
        235269,   1582,    685,   3733,  29907, 235269,  16803, 235269,    578,
         19574, 235265,    108,    141, 235287,  13702,  72780,    708,   1941,
           674,    708,   7290, 122712, 235269,   1582,    685,   9347, 235269,
         57634, 235269,    578, 109955, 235265,    109, 235287,   5231, 216954,
         66058,    108,    141, 235287,   8108,    578,  12158,  29703, 235269,
         44760, 235269,    578,  53186, 235265,    108,    141, 235287,  96084,
           708,   8727,    604,  24091,   1411, 235269,  42696,   4584, 235269,
           578,  14976,  12158, 235265,    108,    141, 235287,   2456,    708,
          2167,   5088,    576,  20361, 235269,   1853,    675,   3724,   7257,
        235265,    109, 235287,   5231, 235311,   1989,  66058,    108,    141,
        235287,  34428,   4134, 235269,  38823, 235269,    578,   1707,  33398,
         48765, 235265,    108,    141, 235287,  41137,  61926,   3707,  21361,
          4684, 235269,  59269, 235269,  22606, 235269,    578,  15741, 235265,
           108,    141, 235287,   3776,  61926,    798,  12310,  45365,   5902,
           578,   4740,    573,   5685,    576,   3760,   7197, 235265,    109,
           688,  33771,    576,  97586, 184592,    575,    573,   9998,  14427,
         66058,    109, 235287,   5231,  23920,   4584,  66058, 110165,  56227,
        235269,  20361, 235269,    578,  61926,   3658,    573,   2971,    675,
          4134, 235265,    108, 235287,   5231,  25251,    578,  68808,  29703,
         66058,  96084,    708,   8727,    604,   4547,    578,  68808,  29703,
        235269,   3359,  22488, 235269], device='cuda:0')

Model output (decoded):
<bos><bos><start_of_turn>user
What are the macronutrients, and what roles do they play in the human body?<end_of_turn>
<start_of_turn>model
Sure, here's a breakdown of the macronutrients and their roles in the human body:

**Macronutrients:**

* **Carbohydrates:**
    * Provide energy for the body's cells and tissues.
    * Carbohydrates are the primary source of energy for most cells.
    * Complex carbohydrates are those that take longer to digest, such as whole grains, fruits, and vegetables.
    * Simple carbohydrates are those that are quickly digested, such as sugar, starch, and lactose.

* **Proteins:**
    * Build and repair tissues, enzymes, and hormones.
    * Proteins are essential for immune function, hormone production, and tissue repair.
    * There are different types of proteins, each with specific functions.

* **Fats:**
    * Provide energy, insulation, and help absorb vitamins.
    * Healthy fats include olive oil, avocado, nuts, and seeds.
    * Trans fats can raise cholesterol levels and increase the risk of heart disease.

**Roles of Macronutrients in the Human Body:**

* **Energy production:** Carbohydrates, proteins, and fats provide the body with energy.
* **Building and repairing tissues:** Proteins are essential for building and repairing tissues, including muscles,

Query: What is the RDI for protein per day?
[INFO] Time taken to get scores on 1680 embeddings: 0.00010 seconds.
My Name Is Chaitanya 
 <bos><start_of_turn>user
Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
- Most nitrogen is lost as urea in the urine, but urea is also excreted in the feces. Proteins are also lost in sweat and as hair and nails grow. The RDA, therefore, is the amount of protein a person should consume in their diet to balance the amount of protein used up and lost from the body. For healthy adults, this amount of protein was determined to be 0.8 grams of protein per kilogram of body weight. You can calculate 410 | Proteins, Diet, and Personal Choices
- Proteins, Diet, and Personal Choices UNIVERSITY OF HAWAI‘I AT MĀNOA FOOD SCIENCE AND HUMAN NUTRITION PROGRAM AND HUMAN NUTRITION PROGRAM We have discussed what proteins are, how they are made, how they are digested and absorbed, the many functions of proteins in the body, and the consequences of having too little or too much protein in the diet. This section will provide you with information on how to determine the recommended amount of protein for you, and your many choices in designing an optimal diet with high-quality protein sources. How Much Protein Does a Person Need in Their Diet? The recommendations set by the IOM for the Recommended Daily Allowance (RDA) and AMDR for protein for different age groups are listed in Table 6.2 “Dietary Reference Intakes for Protein”. A Tolerable Upper Intake Limit for protein has not been set, but it is recommended that you do not exceed the upper end of the AMDR. Table 6.2 Dietary Reference Intakes for Protein Proteins, Diet, and Personal Choices | 409
- Age Group Protein (%) Carbohydrates (%) Fat (%) Children (1–3) 5–20 45–65 30–40 Children and Adolescents (4–18) 10–30 45–65 25–35 Adults (>19) 10–35 45–65 20–35 Source: Food and Nutrition Board of the Institute of Medicine. Dietary Reference Intakes for Energy, Carbohydrate, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids. http://www.nationalacademies.org/hmd/~/media/Files/ Activity%20Files/Nutrition/DRI-Tables/ 8_Macronutrient%20Summary.pdf?la=en. Published 2002. Accessed November 22, 2017.  Tips for Using the Dietary Reference Intakes to Plan Your Diet You can use the DRIs to help assess and plan your diet. Keep in mind when evaluating your nutritional intake that the values established have been devised with an ample safety margin and should be used as guidance for optimal intakes. Also, the values are meant to assess and plan average intake over time; that is, you don’t need to meet these recommendations every single day—meeting them on average over several days is sufficient. Understanding Dietary Reference Intakes | 715
- Age Group RDA (g/day) AMDR (% calories) Infants (0–6 mo) 9.1* Not determined Infants (7–12 mo) 11.0 Not determined Children (1–3) 13.0 5–20 Children (4–8) 19.0 10–30 Children (9–13) 34.0 10–30 Males (14–18) 52.0 10–30 Females (14–18) 46.0 10–30 Adult Males (19+) 56.0 10–35 Adult Females (19+) 46.0 10–35 * Denotes Adequate Intake Source: Dietary Reference Intakes: Macronutrients. Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids. Institute of Medicine. September 5, 2002. Accessed September 28, 2017. Protein Input = Protein Used by the Body + Protein Excreted The appropriate amount of protein in a person’s diet is that which maintains a balance between what is taken in and what is used. The RDAs for protein were determined by assessing nitrogen balance. Nitrogen is one of the four basic elements contained in all amino acids. When proteins are broken down and amino acids are catabolized, nitrogen is released. Remember that when the liver breaks down amino acids, it produces ammonia, which is rapidly converted to nontoxic, nitrogen-containing urea, which is then transported to the kidneys for excretion.
- percent of the population meets their nutrient need is the EAR, and the point at which 97 to 98 percent of the population meets their needs is the RDA. The UL is the highest level at which you can consume a nutrient without it being too much—as nutrient intake increases beyond the UL, the risk of health problems resulting from that nutrient increases. Source: Dietary Reference Intakes Tables and Application. The National Academies of Science, Engineering, and Medicine. Health and Medicine Division. http://nationalacademies.org/HMD/ Activities/Nutrition/SummaryDRIs/DRI-Tables.aspx. Accessed November 22, 2017. The Acceptable Macronutrient Distribution Range (AMDR) is the calculated range of how much energy from carbohydrates, fats, and protein is recommended for a healthy diet adequate of the essential nutrients and is associated with a reduced risk of chronic disease. The ranges listed in Table 12.1 “Acceptable Macronutrient Distribution Ranges (AMDR) For Various Age Groups” allows individuals to personalize their diets taking into consideration that different subgroups in a population often require different requirements. The DRI committee recommends using the midpoint of the AMDRs as an approach to focus on moderation2.

Relevant passages: <extract relevant passages from the context here>
User query: What is the RDI for protein per day?
Answer:<end_of_turn>
<start_of_turn>model

Query: What is the RDI for protein per day?
RAG answer:
<bos>The context does not specify the RDI for protein per day, so I cannot answer this question from the provided context.<eos>
```

Prerequisite:

pip install -r requirements.txt


File description 

PdfParserForGemma.py - THis file is used to parse the text in pdf and generate embedding for the texxt 
it also converts teh texts into paragraph chunk so that its easier for the GEMMA Model to understand the context and give apprpriate answers.
THis part deals with RETRIAVAL of data


GenrateOP.py - THis file is used to augment the Gemmma model from the PDF file to genrate the relevant output 

