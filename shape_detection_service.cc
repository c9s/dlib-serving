#include "shape_detection_service.h"

using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using inference::Point;
using inference::ObjectDetection;
using inference::ShapeDetection;
using inference::DetectionResponse;
using inference::DetectionRequest;

Status DlibShapeDetectionService::Detect(ServerContext* context, DetectionRequest *request, DetectionResponse *response) {
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

  /*
  if (!request->image()) {
    return grpc::Status(grpc::INVALID_ARGUMENT, "image is required.");
  }
  */

  std::string content = request->image();

  auto loader = JpegLoader();
  loader.ReadImage(content);

  dlib::array2d<dlib::rgb_pixel> img;
  loader.GetImage(img);

  dlib::pyramid_up(img);

  // Now tell the face detector to give us a list of bounding boxes
  // around all the faces in the image.
  std::vector<dlib::rectangle> dets = detector(img);

  // Now we will go ask the shape_predictor to tell us the pose of
  // each face we detected.
  std::vector<dlib::full_object_detection> shapes;
  for (unsigned long j = 0; j < dets.size(); ++j)
  {
      dlib::full_object_detection shape = sp_(img, dets[j]);
      std::cout << "number of parts: "<< shape.num_parts() << std::endl;
      std::cout << "pixel position of first part:  " << shape.part(0) << std::endl;
      std::cout << "pixel position of second part: " << shape.part(1) << std::endl;

      for (unsigned long pi = 0; pi < shape.num_parts(); ++pi) {
        dlib::point part = shape.part(pi);

        Point *point = response->add_points();
        point->set_x(part.x());
        point->set_y(part.y());
      }

      // You get the idea, you can get all the face part locations if
      // you want them.  Here we just store them in shapes so we can
      // put them on the screen.
      shapes.push_back(shape);
  }

  return Status::OK;
}

