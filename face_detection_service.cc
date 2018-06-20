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
#include <dlib/image_processing.h>

#include "jpeg_loader.h"
#include "helper.h"
#include "inference.pb.h"
#include "inference.grpc.pb.h"
#include "face_detection_service.h"

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
using inference::DetectionRequest;

using std::chrono::system_clock;

DlibFaceDetectionService::DlibFaceDetectionService() {
  detector_ = dlib::get_frontal_face_detector();
}

Status DlibFaceDetectionService::DetectObjects(ServerContext* context, const DetectionRequest* request, ServerWriter<Rectangle>* writer) {
  if (!request->has_image()) {
    return grpc::Status(grpc::INVALID_ARGUMENT, "image is required.");
  }

  inference::Image image = request->image();
  std::string content = image.content();

  auto loader = JpegLoader();
  loader.ReadImage(content);

  dlib::array2d<dlib::rgb_pixel> img;
  loader.GetImage(img);

  dlib::pyramid_up(img);

  // Now tell the face detector to give us a list of bounding boxes
  // around all the faces in the image.
  std::vector<dlib::rectangle> dets = detector_(img);

  for (auto det : dets) {
    auto rect = Rectangle();
    rect.set_x(det.left());
    rect.set_y(det.top());
    rect.set_width(det.width());
    rect.set_height(det.height());
    writer->Write(rect);
  }

  return Status::OK;
}
