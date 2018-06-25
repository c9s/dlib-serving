#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <dlib/config.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

using namespace dlib;
using namespace std;
namespace fs = boost::filesystem;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);

/*!
    ensures
        - returns an object D such that:    
            - D[i][j] == the distance, in pixels, between the eyes for the face represented
              by objects[i][j].
!*/

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string output;
    std::string data_directory;

    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("help", "Options related to the program.")
        ("output,o", boost::program_options::value<std::string>(&output)->default_value("sp.dat"),"output file")
        ("data-dir", boost::program_options::value<std::string>(&data_directory)->default_value("faces"),"data directory")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_environment(desc, [](const std::string& env_var) {
        return env_var == "DATA_DIR" ? "data-dir" : "";
    }), vm);

    try
    {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);
    }
    catch(std::exception &e)
    { 
        std::cout << e.what() << std::endl;
        return 1;
    }

    if (data_directory.empty())
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./train_shape_predictor [data direcotry]" << endl;
        cout << endl;
        return 1;
    }

    fs::path p = fs::path(data_directory);
    if (!fs::is_directory(p)) {
      cerr << "directory " << data_directory << " does not exist" << endl;
      return 1;
    }

    try
    {
        // The faces directory contains a training dataset and a separate
        // testing dataset.  The training data consists of 4 images, each
        // annotated with rectangles that bound each human face along with 68
        // face landmarks on each face.  The idea is to use this training data
        // to learn to identify the position of landmarks on human faces in new
        // images. 
        // 
        // Once you have trained a shape_predictor it is always important to
        // test it on data it wasn't trained on.  Therefore, we will also load
        // a separate testing set of 5 images.  Once we have a shape_predictor 
        // created from the training data we will see how well it works by
        // running it on the testing images. 
        // 
        // So here we create the variables that will hold our dataset.
        // images_train will hold the 4 training images and faces_train holds
        // the locations and poses of each face in the training images.  So for
        // example, the image images_train[0] has the faces given by the
        // full_object_detections in faces_train[0].
        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;

        // Now we load the data.  These XML files list the images in each
        // dataset and also contain the positions of the face boxes and
        // landmarks (called parts in the XML file).  Obviously you can use any
        // kind of input format you like so long as you store the data into
        // images_train and faces_train.  But for convenience dlib comes with
        // tools for creating and loading XML image dataset files.  Here you see
        // how to load the data.  To create the XML files you can use the imglab
        // tool which can be found in the tools/imglab folder.  It is a simple
        // graphical tool for labeling objects in images.  To see how to use it
        // read the tools/imglab/README.txt file.
        cout << "loading training image dataset" << endl;
        load_image_dataset(images_train, faces_train, data_directory+"/training.xml");

        cout << "loading testing image dataset" << endl;
        load_image_dataset(images_test, faces_test, data_directory+"/testing.xml");


        // Now make the object responsible for training the model.  
        shape_predictor_trainer trainer;
        // This algorithm has a bunch of parameters you can mess with.  The
        // documentation for the shape_predictor_trainer explains all of them.
        // You should also read Kazemi's paper which explains all the parameters
        // in great detail.  However, here I'm just setting three of them
        // differently than their default values.  I'm doing this because we
        // have a very small dataset.  In particular, setting the oversampling
        // to a high amount (300) effectively boosts the training set size, so
        // that helps this example.
        trainer.set_oversampling_amount(300);
        // I'm also reducing the capacity of the model by explicitly increasing
        // the regularization (making nu smaller) and by using trees with
        // smaller depths.  
        trainer.set_nu(0.05);
        trainer.set_tree_depth(2);

        // some parts of training process can be parallelized.
        // Trainer will use this count of threads when possible
        trainer.set_num_threads(2);

        // Tell the trainer to print status messages to the console so we can
        // see how long the training will take.
        trainer.be_verbose();

        // Now finally generate the shape model
        dlib::shape_predictor sp = trainer.train(images_train, faces_train);


        // Now that we have a model we can test it.  This function measures the
        // average distance between a face landmark output by the
        // shape_predictor and where it should be according to the truth data.
        // Note that there is an optional 4th argument that lets us rescale the
        // distances.  Here we are causing the output to scale each face's
        // distances by the interocular distance, as is customary when
        // evaluating face landmarking systems.
        cout << "mean training error: "<< 
            test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

        // The real test is to see how well it does on data it wasn't trained
        // on.  We trained it on a very small dataset so the accuracy is not
        // extremely high, but it's still doing quite good.  Moreover, if you
        // train it on one of the large face landmarking datasets you will
        // obtain state-of-the-art results, as shown in the Kazemi paper.
        cout << "mean testing error:  " << 
            test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;


        std::string output_file = "sp.dat";

        cout << "writing model file at " << output_file << endl;

        // Finally, we save the model to disk so we can use it later.
        dlib::serialize(output_file) << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around 
    // the eye.
    for (unsigned long i = 36; i <= 41; ++i) 
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the right eye by averaging the points around 
    // the eye.
    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
    {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}
