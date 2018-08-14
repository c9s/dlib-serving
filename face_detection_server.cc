#include <iostream>
#include <memory>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "serving.pb.h"
#include "serving.grpc.pb.h"

#include "face_detection_service.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;

using std::chrono::system_clock;

const int DEFAULT_TCP_PORT = 50001;

void RunServer(const std::string& server_address) {
  DlibFaceDetectionService service;
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(8 * 1024 * 1024);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
    std::string bind;
    bool verbose;
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("help", "Options related to the program.")
        ("bind,b", boost::program_options::value<std::string>(&bind)->default_value("0.0.0.0:50051"),"Address to bind")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(false), "Print to stdout information as job is processed.")
        ;

  // parse command line options
  boost::program_options::variables_map vm;

  try
  {
      boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
      boost::program_options::notify(vm);
  }
  catch(std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // const char * env_model_file = std::getenv("MODEL_FILE");
  RunServer(bind);
  return 0;
}
