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

using serving::Point;
using serving::Object;
using serving::Rectangle;
using serving::ObjectDetection;
using serving::DetectionRequest;
using serving::DetectionResponse;

using std::chrono::system_clock;

DlibFaceDetectionService::DlibFaceDetectionService() {
  detector_ = dlib::get_frontal_face_detector();
}

Status DlibFaceDetectionService::DetectStream(ServerContext* context, const DetectionRequest* request, DetectionResponse *response) {
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

  response->set_type("objects");

  for (const dlib::rectangle & det : dets) {
    std::cout << "found face at " << det << std::endl;
    Object *object = response->add_objects();

    Rectangle *box = new Rectangle;
    box->set_x(det.left());
    box->set_y(det.top());
    box->set_width(det.width());
    box->set_height(det.height());
    object->set_allocated_box(box);
  }

  return Status::OK;
}
