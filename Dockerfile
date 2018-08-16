FROM yoanlin/alpine-boost-dev:edge

RUN apk update \
    && apk add --no-cache git \
    cmake \
    g++ \
    make \
    jpeg jpeg-dev \
    libpng libpng-dev \
    giflib-dev \
    boost-dev \
    openblas-dev  \
    ca-certificates wget \
    && rm -rf /var/cache/apk/*

# && rm -rf /var/cache/apk/*

# FROM yoanlin/dlib:latest

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# install ca-certificates so that HTTPS works consistently
# the other runtime dependencies for Python are installed later

# dependencies for scipy, numpy and Pillow
# Runtime
# RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/main >> /etc/apk/repositories
# RUN apk update

RUN apk add --progress zip zlib jpeg libpng openblas boost libcrypto1.0 libssl1.0 protobuf
RUN apk add --progress libstdc++
RUN apk add --progress --update git make g++ unzip autoconf automake libtool file libressl curl

ARG GRPC_VERSION=v1.12.x
RUN git clone --depth 1 --branch $GRPC_VERSION https://github.com/grpc/grpc /tmp/grpc
# COPY grpc /tmp/grpc
RUN (cd /tmp/grpc \
    && git submodule update --init --recursive \
    && (cd third_party/protobuf && ./autogen.sh && ./configure && make install) \
    && make \
    && make install \
    ) && rm -rf /tmp/grpc

# I should do, but I want to know correct solution
# looks like the version link is incorrect
# might be related to https://github.com/grpc/grpc/pull/13264/files
RUN ln -s /usr/local/lib/libgrpc++.so /usr/local/lib/libgrpc++.so.1

RUN wget -c -q https://github.com/davisking/dlib/archive/master.tar.gz
RUN tar xvf master.tar.gz \
    && mv dlib-master dlib \
    && (mkdir -p dlib/build \
    && cd dlib/build \
    && cmake -DDLIB_PNG_SUPPORT=ON -DDLIB_GIF_SUPPORT=ON -DDLIB_JPEG_SUPPORT=ON .. \
    && cmake --build . --config Release \
    && make install \
    ) && rm master.tar.gz && rm -rf /dlib/build

RUN mkdir /src
COPY *.cc /src/
COPY *.h /src/
COPY cmake/ /src/cmake/
COPY CMakeLists.txt /src/
COPY protos/ /src/protos/
RUN mkdir /build
WORKDIR /build
RUN cmake /src && cmake --build . -- -j2
RUN curl -O https://storage.googleapis.com/dlib-models/shape_predictor_68_face_landmarks.dat
RUN curl -O https://storage.googleapis.com/dlib-models/shape_predictor_5_face_landmarks.dat

FROM alpine:edge
WORKDIR /
RUN apk update \
    && apk add --no-cache giflib \
    jpeg \
    libjpeg-turbo \
    libpng \
    boost-system \
    boost-program_options \
    boost-filesystem \
    openblas
RUN apk add libcrypto1.0 libssl1.0 protobuf libstdc++ libressl
RUN rm -rf /usr/local
COPY --from=0 /usr/local/ /usr/local/
COPY --from=0 /build/*.dat /
COPY --from=0 /build/face-detection-server /
COPY --from=0 /build/face-detection-client /
COPY --from=0 /build/shape-detection-server /
COPY --from=0 /build/train-shape-predictor /
