#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/smart_ptr/shared_ptr.hpp"

// Caffe's required library
#pragma comment(lib, "caffeout.lib")
#include "include/caffe/sgd_solvers.hpp"
#include <string>


// enable namespace
using namespace std;
using namespace cv;
using namespace boost;
using namespace caffe;

// caffe training
int gotime()
{
    // parse solver parameters
    string solver_prototxt = "examples/mnist/lenet_solver-leveldb.prototxt";
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(solver_prototxt, &solver_param);

    // set device id and mode
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);

    // solver handler
	boost::shared_ptr<caffe::Solver<float> > 
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    //AdamSolver<float> solver(solver_param);

    //// resume training
    //string snapshot_status = "lenet_iter_10000.solverstate";
    //solver->Solve(snapshot_status);

    //// finetune
    //string weights_model = "models/bvlc_reference_caffenet.caffemodel";
    //solver->net()->CopyTrainedLayersFrom(weights_model);

    // start solver
    solver->Solve();
    LOG(INFO) << "Optimization Done.";

    return 0;
}

int main() {
	//int x = 1;
    gotime();
}