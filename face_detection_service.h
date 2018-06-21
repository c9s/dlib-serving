#ifndef FACE_DETECTION_SERVICE_H
#define FACE_DETECTION_SERVICE_H
#include <grpc/grpc.h>
#include <grpcpp/server_context.h>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "inference.pb.h"
#include "inference.grpc.pb.h"

using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

using inference::Object;
using inference::ObjectDetection;
using inference::DetectionRequest;

using std::chrono::system_clock;

class DlibFaceDetectionService final : public ObjectDetection::Service {
 public:
  DlibFaceDetectionService();
  grpc::Status DetectObjects(grpc::ServerContext* context, const DetectionRequest* request, grpc::ServerWriter<Object>* writer);
 private:
  dlib::frontal_face_detector detector_;
};

#endif
