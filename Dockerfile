FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update \
    && apt-get install -y git git-lfs wget libgl1-mesa-glx libosmesa6-dev libglu1-mesa-dev

RUN git clone https://github.com/isl-org/Open3D.git \
    && cd Open3D \
    && git checkout 8e434558a9b1ecacba7854da7601a07e8bdceb26 \
    && mkdir build \
    && cd build \
    && cmake -DENABLE_HEADLESS_RENDERING=ON \
             -DBUILD_GUI=OFF \
             -DUSE_SYSTEM_GLEW=OFF \
             -DUSE_SYSTEM_GLFW=OFF \
             .. \
    && make -j8 \
    && make install-pip-package

RUN pip install --no-deps \
    git+https://github.com/facebookresearch/pytorch3d@06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f \
    git+https://github.com/CadQuery/cadquery.git@e99a15df3cf6a88b69101c405326305b5db8ed94 \
    accelerate==0.34.2 \
    cadquery-ocp==7.7.2 \
    casadi==3.6.7 \
    comm==0.2.2 \
    contourpy==1.3.1 \
    cycler==0.12.1 \
    debugpy==1.8.11 \
    einops==0.8.0 \
    ezdxf==1.3.5 \
    flash-attn==2.7.2.post1 \
    fonttools==4.55.3 \
    huggingface-hub==0.27.0 \
    imageio==2.36.1 \
    ipykernel==6.29.5 \
    ipywidgets==8.1.5 \
    jupyter_client==8.6.3 \
    jupyterlab_widgets==3.0.13 \
    kiwisolver==1.4.7 \
    lazy_loader==0.4 \
    manifold3d==3.0.0 \
    matplotlib==3.10.0 \
    multimethod==1.12 \
    nlopt==2.9.0 \
    numpy==2.2.0 \
    path==17.0.0 \
    pyparsing==3.2.0 \
    python-dateutil==2.9.0.post0 \
    pyzmq==26.2.0 \
    regex==2024.11.6 \
    safetensors==0.4.5 \
    scikit-image==0.25.0 \
    scipy==1.14.1 \
    tifffile==2024.12.12 \
    tokenizers==0.21.0 \
    tornado==6.4.2 \
    transformers==4.50.3 \
    trimesh==4.5.3 \
    typish==1.9.3 \
    qwen-vl-utils==0.0.10 \
    widgetsnbextension==4.0.13
