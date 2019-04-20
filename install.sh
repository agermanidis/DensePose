apt-get update
apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      libgflags-dev \
      cmake \
      wget
pip install --user \
      future \
      numpy \
      protobuf \
      typing \
      Cython \
      hypothesis
git clone https://github.com/pytorch/pytorch.git && cd pytorch && git submodule update --init --recursive && python2 setup.py install && cp build/lib/* /usr/local/lib && cd ..
pip install -U Cython
pip install -r requirements.txt
make
mkdir weights
wget https://s3.amazonaws.com/runway-checkpoints/DensePose_ResNet50_FPN_s1x-e2e.pkl -P weights
