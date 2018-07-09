#ifndef SHAPE_DETECTION_SERVICE_H
#define SHAPE_DETECTION_SERVICE_H

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <grpc/grpc.h>
#include <grpcpp/server_context.h>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include "jpeg_loader.h"
#include "serving.pb.h"
#include "serving.grpc.pb.h"
#include "shape_detection_service.h"

using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::Status;

using serving::ObjectDetection;
using serving::DetectionResponse;
using serving::DetectionRequest;

class DlibShapeDetectionService final : public ObjectDetection::Service {
  public:
    explicit DlibShapeDetectionService(const std::string& model_file) {
      model_file_ = model_file;
      sp_ = dlib::shape_predictor();
      dlib::deserialize(model_file_) >> sp_;
    }

    virtual grpc::Status Detect(ServerContext* context, const DetectionRequest *request, DetectionResponse *response);

  private:
    std::string model_file_;
    dlib::shape_predictor sp_;
};

#endif
