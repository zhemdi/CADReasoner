FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update \
    && apt-get install -y git git-lfs wget libgl1-mesa-glx libosmesa6-dev libglu1-mesa-dev

RUN pip install \
    accelerate==0.34.2 \
    cadquery==2.5.2 \
    cadquery-ocp==7.7.2 \
    flash-attn==2.7.2.post1 \
    manifold3d==3.0.0 \
    numpy==2.2.0 \
    pillow==11.3.0 \
    pyvista==0.46.3 \
    safetensors==0.4.5 \
    scipy==1.14.1 \
    tokenizers==0.21.0 \
    transformers==4.50.3 \
    trimesh==4.5.3 \
    qwen-vl-utils==0.0.10