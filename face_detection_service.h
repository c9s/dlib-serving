#ifndef FACE_DETECTION_SERVICE_H
#define FACE_DETECTION_SERVICE_H
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "serving.pb.h"
#include "serving.grpc.pb.h"

using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

using serving::Object;
using serving::ObjectDetection;
using serving::DetectionRequest;
using serving::DetectionResponse;

class DlibFaceDetectionService final : public ObjectDetection::Service {
 public:
  DlibFaceDetectionService();

  // using ObjectDetection::Service::Detect;
  virtual grpc::Status Detect(grpc::ServerContext* context, const DetectionRequest* request, DetectionResponse *response);

 private:
  dlib::frontal_face_detector detector_;
};

#endif
