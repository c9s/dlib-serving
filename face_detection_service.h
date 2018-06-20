#ifndef FACE_DETECTION_SERVICE_H
#define FACE_DETECTION_SERVICE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "inference.pb.h"
#include "inference.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using inference::Point;
using inference::Rectangle;
using inference::ObjectDetection;
using inference::ShapeDetection;
using inference::ShapeDetectionResponse;
using inference::DetectionRequest;

using std::chrono::system_clock;

class DlibFaceDetectionService final : public ObjectDetection::Service {
 public:
  DlibFaceDetectionService();
  grpc::Status DetectObjects(grpc::ServerContext* context, const DetectionRequest* request, grpc::ServerWriter<Rectangle>* writer);
 private:
  dlib::frontal_face_detector detector_;
};

#endif
