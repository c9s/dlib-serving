#include <iostream>
#include <memory>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/security/server_credentials.h>

#include "serving.pb.h"
#include "serving.grpc.pb.h"
#include "shape_detection_service.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using std::chrono::system_clock;

void RunServer(const std::string& server_address, const std::string& model_file) {

  DlibShapeDetectionService service(model_file);

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(8 * 1024 * 1024);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  const std::string default_shape_model = "shape_predictor_68_face_landmarks.dat";
  const char * env_model_file = std::getenv("MODEL_FILE");

  std::string model_file;
  if (env_model_file != NULL) {
    model_file.assign(env_model_file);
  } else {
    std::cout << "env MODEL_FILE is not defined." << std::endl;
    std::cout << "fallback to default shape landmarks model:" << default_shape_model << std::endl;
    model_file = default_shape_model;
  }
  std::string server_address("0.0.0.0:50051");
  RunServer(server_address, model_file);
  return 0;
}
