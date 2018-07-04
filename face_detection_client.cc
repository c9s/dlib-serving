#include <chrono>

#include <iostream>
#include <fstream>
#include <sstream>

#include <memory>
#include <random>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "serving.pb.h"
#include "serving.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

using serving::ObjectDetection;
using serving::DetectionRequest;
using serving::Object;
using serving::Point;

class FaceDetectionClient {

  public:
    FaceDetectionClient(std::shared_ptr<Channel> channel)
      : stub_(serving::ObjectDetection::NewStub(channel)) { }

  bool DetectImageFile(const std::string& image_file) {
    std::ifstream file(image_file, std::ios::binary);
    std::ostringstream ostrm;
    ostrm << file.rdbuf();
    std::string bytes = ostrm.str();
    return DetectImage(bytes);
  }

  bool DetectImage(const std::string& image) {
    DetectionRequest request;
    request.set_allocated_image(new std::string(image));
    request.clear_region();

    ClientContext context;
    std::unique_ptr<ClientReader<Object> > reader(
        stub_->DetectStream(&context, request));

    Object obj;
    while (reader->Read(&obj)) {
      std::cerr << "Found rect: " << obj.box().x() << "x" << obj.box().y() << " at " << std::endl;
      std::cerr.flush();
    }

    Status status = reader->Finish();
    if (status.ok()) {
      std::cerr << "Detect rpc succeeded." << std::endl;
    } else {
      std::cerr << "Detect rpc failed." << std::endl;
    }
    std::cout << status.error_details() << std::endl;
    std::cout << status.error_message() << std::endl;

    return true;
  }

  private:
  std::unique_ptr<ObjectDetection::Stub> stub_;
};

int main(int argc, char** argv) {
  auto channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
  auto client = FaceDetectionClient(channel);

  client.DetectImageFile("dlib/examples/faces/2007_007763.jpg");
  // client.DetectImage(bytes);
  return 0;
}
