/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <cstdio>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>


#include <jpeglib.h>

#include <dlib/config.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include "helper.h"
#include "inference.pb.h"
#include "inference.grpc.pb.h"
#include "face_landmark_detection.h"



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


class JpegLoader {

public:

  explicit JpegLoader() : width_(0), height_(0), output_components_(0) { }


  JpegLoader(const std::vector<unsigned char>& data, int height, int width, int output_components)
      : data(data), height_(height), width_(width), output_components_(output_components)
  {
      // read_image(filename);
  }
  
  bool is_gray() const
  {
      return (output_components_ == 1);
  }

  bool is_rgb() const
  {
      return (output_components_ == 3);
  }

  bool is_rgba() const
  {
      return (output_components_ == 4);
  }

  // decode jpeg image from memory
  // https://gist.github.com/PhirePhly/3080633
  void ReadImage(const std::string& content) {
    const char * chrs = content.c_str();

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    auto uchrs = reinterpret_cast<unsigned char*>(const_cast<char*>(chrs));
    auto jpg_size = content.length();
    unsigned char *jpg_buffer = uchrs;
    jpeg_mem_src(&cinfo, jpg_buffer, jpg_size);

    int rc = jpeg_read_header(&cinfo, TRUE);
    if (rc != 1) {
      throw std::runtime_error("jpeg: decode failed");
      // return grpc::Status(grpc::INVALID_ARGUMENT, "jpeg decode failed");
    }
    jpeg_start_decompress(&cinfo);

    width_ = cinfo.output_width;
    height_ = cinfo.output_height;

    if (output_components_ != 1 && 
        output_components_ != 3 &&
        output_components_ != 4)
    {
      jpeg_destroy_decompress(&cinfo);
      throw std::runtime_error("jpeg: unsupported number of colors");
      // return grpc::Status(grpc::INVALID_ARGUMENT, "jpeg: unsupported number of colors");
      // std::ostringstream sout;
      // sout << "jpeg_loader: Unsupported number of colors (" << output_components_ << ") in file " << filename;
      // throw image_load_error(sout.str());
    }

    std::vector<unsigned char*> rows;
    rows.resize(height_);

    // size the image buffer
    data.resize(height_ * width_ * output_components_);

    // setup pointers to each row
    for (unsigned long i = 0; i < rows.size(); ++i)
        rows[i] = &data[i * width_ * output_components_];

    // read the data into the buffer
    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, &rows[cinfo.output_scanline], 100);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
  }

  template<typename T>
  void GetImage(T &t_) const
  {

    dlib::image_view<T> t(t_);
    t.set_size(height_, width_);
    for (unsigned n = 0; n < height_; n++)
    {
        const unsigned char* v = get_row(n);
        for (unsigned m = 0; m < width_;m++)
        {
            if (is_gray())
            {
                unsigned char p = v[m];
                dlib::assign_pixel( t[n][m], p );
            }
            else if (is_rgba()) {
                dlib::rgb_alpha_pixel p;
                p.red = v[m*4];
                p.green = v[m*4+1];
                p.blue = v[m*4+2];
                p.alpha = v[m*4+3];
                dlib::assign_pixel( t[n][m], p );
            }
            else // if ( is_rgb() )
            {
                dlib::rgb_pixel p;
                p.red = v[m*3];
                p.green = v[m*3+1];
                p.blue = v[m*3+2];
                dlib::assign_pixel( t[n][m], p );
            }
        }
    }
  }

private:


  int width_;

  int height_;

  int output_components_;

  std::vector<unsigned char> data;

  const unsigned char* get_row(unsigned long i) const
  {
      return &data[i*width_*output_components_];
  }
};






/*
float ConvertToRadians(float num) {
  return num * 3.1415926 /180;
}
*/

// The formula is based on http://mathforum.org/library/drmath/view/51879.html
/*
float GetDistance(const Point& start, const Point& end) {
  const float kCoordFactor = 10000000.0;
  float lat_1 = start.latitude() / kCoordFactor;
  float lat_2 = end.latitude() / kCoordFactor;
  float lon_1 = start.longitude() / kCoordFactor;
  float lon_2 = end.longitude() / kCoordFactor;
  float lat_rad_1 = ConvertToRadians(lat_1);
  float lat_rad_2 = ConvertToRadians(lat_2);
  float delta_lat_rad = ConvertToRadians(lat_2-lat_1);
  float delta_lon_rad = ConvertToRadians(lon_2-lon_1);

  float a = pow(sin(delta_lat_rad/2), 2) + cos(lat_rad_1) * cos(lat_rad_2) *
            pow(sin(delta_lon_rad/2), 2);
  float c = 2 * atan2(sqrt(a), sqrt(1-a));
  int R = 6371000; // metres

  return R * c;
}
*/

/*
std::string GetFeatureName(const Point& point,
                           const std::vector<Feature>& feature_list) {
  for (const Feature& f : feature_list) {
    if (f.location().latitude() == point.latitude() &&
        f.location().longitude() == point.longitude()) {
      return f.name();
    }
  }
  return "";
}
*/


class ShapeDetectionService final : public ShapeDetection::Service {

  public:
    explicit ShapeDetectionService(const std::string& model_file) {
      model_file_ = model_file;
      sp_ = dlib::shape_predictor();
      dlib::deserialize(model_file_) >> sp_;
    }

    Status DetectShape(ServerContext* context, DetectionRequest *request, ShapeDetectionResponse *response) {

      dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

      Point *point = response->add_points();
      
      if (request->has_image()) {
        inference::Image image = request->image();
        std::string content = image.content();

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
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            shapes.push_back(shape);
        }

      }

      return Status::OK;
    }

  private:
    std::string model_file_;
    dlib::shape_predictor sp_;
};




class ObjectDetectionService final : public ObjectDetection::Service {

 public:
  explicit ObjectDetectionService() {
    detector_ = dlib::get_frontal_face_detector();
  }

  /*
  Status GetFeature(ServerContext* context, const Point* point,
                    Feature* feature) override {

    feature->set_name(GetFeatureName(*point, feature_list_));
    feature->mutable_location()->CopyFrom(*point);
    return Status::OK;
  }
  */

  /*
  Status ListFeatures(ServerContext* context,
                      const inference::Rectangle* rectangle,
                      ServerWriter<Feature>* writer) override {

    auto lo = rectangle->lo();
    auto hi = rectangle->hi();
    long left = (std::min)(lo.longitude(), hi.longitude());
    long right = (std::max)(lo.longitude(), hi.longitude());
    long top = (std::max)(lo.latitude(), hi.latitude());
    long bottom = (std::min)(lo.latitude(), hi.latitude());
    for (const Feature& f : feature_list_) {
      if (f.location().longitude() >= left &&
          f.location().longitude() <= right &&
          f.location().latitude() >= bottom &&
          f.location().latitude() <= top) {
        writer->Write(f);
      }
    }
    return Status::OK;
  }
  */

  /*
  Status RecordRoute(ServerContext* context, ServerReader<Point>* reader,
                     RouteSummary* summary) override {
    Point point;
    int point_count = 0;
    int feature_count = 0;
    float distance = 0.0;
    Point previous;

    system_clock::time_point start_time = system_clock::now();
    while (reader->Read(&point)) {
      point_count++;
      if (!GetFeatureName(point, feature_list_).empty()) {
        feature_count++;
      }
      if (point_count != 1) {
        distance += GetDistance(previous, point);
      }
      previous = point;
    }
    system_clock::time_point end_time = system_clock::now();
    summary->set_point_count(point_count);
    summary->set_feature_count(feature_count);
    summary->set_distance(static_cast<long>(distance));
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    summary->set_elapsed_time(secs.count());
    return Status::OK;
  }
  */

  /*
  Status RouteChat(ServerContext* context,
                   ServerReaderWriter<RouteNote, RouteNote>* stream) override {

    std::vector<RouteNote> received_notes;
    RouteNote note;
    while (stream->Read(&note)) {
      for (const RouteNote& n : received_notes) {
        if (n.location().latitude() == note.location().latitude() &&
            n.location().longitude() == note.location().longitude()) {
          stream->Write(n);
        }
      }
      received_notes.push_back(note);
    }

    return Status::OK;
  }
  */

 private:
  dlib::frontal_face_detector detector_;
};

void RunServer(const std::string& db_path) {
  std::string server_address("0.0.0.0:50051");

  ShapeDetectionService service("shape_predictor_68_face_landmarks.dat.bz2");
  // ObjectDetectionService service();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

/*
int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  // std::string db = inference::GetDbFileContent(argc, argv);
  std::string db = "test";
  RunServer(db);
  return 0;
}
*/
