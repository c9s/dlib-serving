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
#include "face_detection_service.h"

using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using inference::Point;
using inference::Object;
using inference::Rectangle;
using inference::ObjectDetection;
using inference::DetectionRequest;

using std::chrono::system_clock;

DlibFaceDetectionService::DlibFaceDetectionService() {
  detector_ = dlib::get_frontal_face_detector();
}

Status DlibFaceDetectionService::DetectObjects(ServerContext* context, const DetectionRequest* request, ServerWriter<Object>* writer) {
  // return grpc::Status(grpc::INVALID_ARGUMENT, "image is required.");
  std::string content = request->image();

  std::cerr << "reading image" << std::endl;
  auto loader = JpegLoader();
  loader.ReadImage(content);

  std::cerr << "loading image" << std::endl;
  dlib::array2d<dlib::rgb_pixel> img;
  loader.GetImage(img);

  std::cerr << "pyramid up image" << std::endl;
  dlib::pyramid_up(img);

  // Now tell the face detector to give us a list of bounding boxes
  // around all the faces in the image.
  std::cerr << "detecting" << std::endl;
  std::vector<dlib::rectangle> dets = detector_(img);

  for (const dlib::rectangle & det : dets) {
    std::cout << "found face at " << det << std::endl;
    Object obj;
    Rectangle *rect = new Rectangle;
    rect->set_x(det.left());
    rect->set_y(det.top());
    rect->set_width(det.width());
    rect->set_height(det.height());
    obj.set_allocated_rect(rect);
    writer->Write(obj);
  }

  return Status::OK;
}
