#pragma once
#ifndef DQN_NEURAL_Q_LEANER_H
#define DQN_NEURAL_Q_LEANER_H

#define GLOG_NO_ABBREVIATED_SEVERITIES


#include <string>
#include "TransitionQueue.h"
#include <atomic>
#include <thread>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/net.hpp>
#include <random>
#include <caffe/util/io.hpp>
#include <caffe/solver.hpp>
#include <include/caffe/layers/memory_data_layer.hpp>
#include <include/caffe/layers/eltwise_layer.hpp>
#include <include/caffe/util/upgrade_proto.hpp>
#include <include/caffe/layers/reshape_layer.hpp>

// set caffe root path manually
const std::string CAFFE_ROOT = "../caffe";

namespace dqn
{

// void train_minibatch_thread(void* neural_q_learner);

class NeuralQLearner
{
	TransitionQueue* transitions_;
	int minibatch_size_;  // Needs to be fast enough to act between batches.
	int replay_memory_;
	std::atomic<bool> queue_lock_;
	boost::shared_ptr<caffe::Net<float>> net_;
	boost::shared_ptr<caffe::Net<float>> clone_net_;
	caffe::Solver<float>* solver_;
	boost::shared_ptr<caffe::Blob<float>> frames_input_blob_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> frames_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> target_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> action_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> reshape_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_frames_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_target_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_action_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_reshape_layer_;
	std::vector<boost::shared_ptr<caffe::MemoryDataLayer<float>>> input_layers_;
	std::vector<boost::shared_ptr<caffe::MemoryDataLayer<float>>> clone_input_layers_;
	int num_actions_;
	std::vector<int> actions_;
	cv::Mat* last_state_;
	int      last_action_;
	float    last_reward_;
	bool     last_terminal_;
	int raw_frame_width_ = 771;
	int frame_area_ = raw_frame_width_ * raw_frame_width_;
	int sample_frame_count_ = 4;
	int sample_data_size_ = frame_area_ * sample_frame_count_;
	int minibatch_data_size_ = sample_data_size_ * minibatch_size_;
	std::vector<float> last_q_values_;
	bool should_train_;
	bool should_train_async_;
	int train_iter_;
	int clone_iter_;
	long iter_ = 0;
	double discount_;

	public:
	NeuralQLearner(int state_dim, int replay_memory, int minibatch_size,
		int n_actions, double discount, double q_learning_rate,
		std::string solver_path, bool is_training, int train_iter,
		bool should_train_async, int clone_iter)
	{
		transitions_ = new TransitionQueue(state_dim, replay_memory);
		minibatch_size_ = minibatch_size;
		replay_memory_ = replay_memory;
		num_actions_ = n_actions;
		discount_ = discount;
		should_train_ = is_training;
		should_train_async_ = should_train_async;
		train_iter_ = train_iter;
		clone_iter_ = clone_iter;

		for(auto i = 0; i < n_actions; i++)
		{
			actions_.push_back(i);
		}
		queue_lock_ = false;

		// parse solver parameters
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_path, &solver_param);

		// set device id and mode
		caffe::Caffe::SetDevice(0);
		caffe::Caffe::set_mode(caffe::Caffe::CPU);

		// solver handler
		solver_ = caffe::SolverRegistry<float>::CreateSolver(solver_param);

		//net_ = new caffe::Net<float>(CAFFE_ROOT + "/" + proto_path);

		net_ = boost::dynamic_pointer_cast<caffe::Net<float>>(solver_->net());
		
		frames_input_blob_ = net_->blob_by_name("gta_frames_input_layer");

		frames_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net_->layer_by_name("gta_frames_input_layer"));
		target_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net_->layer_by_name("target_input_layer"));
		action_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net_->layer_by_name("action_input_layer"));
//		reshape_layer_ =
//			boost::dynamic_pointer_cast<caffe::ReshapeLayer<float>>(
//				net_->layer_by_name("reshape"));


 		assert(frames_input_layer_);
		assert(target_input_layer_);
		assert(action_input_layer_);
//		assert(reshape_layer_);

		input_layers_.push_back(frames_input_layer_);
		input_layers_.push_back(target_input_layer_);
		input_layers_.push_back(action_input_layer_);

		caffe::NetParameter net_param;
		ReadNetParamsFromTextFileOrDie("examples/dqn/dqn_model.prototxt", &net_param);
		clone_net_.reset(new caffe::Net<float>(net_param));
		reset_clone_net();

		net_->set_debug_info(true);
		clone_net_->set_debug_info(true);

		// solver_->OnlineUpdateSetup(nullptr);
	}

	void reset_clone_net()
	{
		caffe::NetParameter net_param;
		net_->ToProto(&net_param);
		net_param.mutable_state()->set_phase(caffe::TEST);
		clone_net_->CopyTrainedLayersFrom(net_param);
		
		clone_frames_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				clone_net_->layer_by_name("gta_frames_input_layer"));
		clone_target_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				clone_net_->layer_by_name("target_input_layer"));
		clone_action_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				clone_net_->layer_by_name("action_input_layer"));
//		clone_reshape_layer_ =
//			boost::dynamic_pointer_cast<caffe::ReshapeLayer<float>>(
//				net_->layer_by_name("reshape"));

 		assert(clone_frames_input_layer_);
		assert(clone_target_input_layer_);
		assert(clone_action_input_layer_);
//		assert(clone_reshape_layer_);

		clone_input_layers_.push_back(clone_frames_input_layer_);
		clone_input_layers_.push_back(clone_target_input_layer_);
		clone_input_layers_.push_back(clone_action_input_layer_);
	}

	void load_weights(std::string model_file)
	{
		net_->CopyTrainedLayersFrom(CAFFE_ROOT + "/" + model_file);
	}

	~NeuralQLearner()
	{
		delete transitions_;
		net_.reset();
		delete solver_;
		for(int i = 0; i < input_layers_.size(); i++)
		{
			input_layers_[i].reset();
		}
		for(int i = 0; i < clone_input_layers_.size(); i++)
		{
			clone_input_layers_[i].reset();
		}
	}

	int select_action_e_greedy(double epsilon);

	int perceive(float reward, cv::Mat* raw_state, bool terminal, 
		bool testing, double epsilon);

	bool get_queue_lock()
	{
		// TODO: Use regular lock instead of an atomic bool.
		auto expect_false = false;
		return queue_lock_.compare_exchange_strong(expect_false, true);
	}

	void release_queue_lock()
	{
		queue_lock_.store(false);
	}

	std::vector<float> array_to_vec(const float* out_array, const int size)
	{
		std::vector<float> output(out_array, out_array + size);
		return output;
	}

	void set_action_vector(int action_index, std::vector<float> & actions)
	{
		for(int i = 0; i < num_actions_; i++)
		{
			if(i == action_index)
			{
				actions.push_back(1.0);
			}
			else
			{
				actions.push_back(0.0);
			}
		}		
	}

	std::vector<float> feed_net(std::vector<Transition>& transistion_sample, 
		boost::shared_ptr<caffe::Net<float>> net, bool is_s1, std::vector<float> &targets)
	{
		std::vector<cv::Mat> frames;
		std::vector<float> actions;

		for (auto i = 0; i < minibatch_size_; i++)
		{
			Transition transition = transistion_sample[i];
			if(is_s1) frames.push_back(*(transition.s));
			else      frames.push_back(*(transition.s2));
			set_action_vector(transition.a, actions);
			// Frame input size is minibatch * frames_per_sample * sizeof(cv::Mat == w * h)
			// TODO: Set input channels with four consecutive frames.
		}

		std::vector<int> labels(frames.size());
		std::vector<float> labels2(frames.size());

		std::fill(labels.begin(), labels.end(), 0);
		std::fill(labels2.begin(), labels2.end(), 0.0f);

		// TODO: Create new memory layer add mat vector without transform (random crop, scale, mirror)
		// that way we can take advantage of a fixed perspective and input time based channels.
		// Do this after you have stepped through with one frame as input and can see the 
		// dimensions and format that Transform spits out. (Should just be flattened array of batch * channels * w * h)
		// First see if transform is too slow though, since data augementation could still be useful.
		// Could just do Reset()...

		const float* out_array;
		if(is_s1)
		{
			std::vector<float> dummy_input;
			// Get actuals - Q1
			// Forward to train net
			frames_input_layer_->AddMatVector(frames, labels);
			action_input_layer_->Reset(&actions[0], &labels2[0], minibatch_size_);
			target_input_layer_->Reset(const_cast<float*>(targets.data()), &labels2[0], minibatch_size_);

			// TODO: Remove this redundant forward pass, just used for checking
//			auto loss = net->ForwardPrefilled();

			solver_->Step(1);

//			auto check_target = array_to_vec(net->blob_by_name("target")->cpu_data(), minibatch_size_ * num_actions_);
//			auto check_q_values = array_to_vec(net->blob_by_name("gtanet_q_values")->cpu_data(), minibatch_size_ * num_actions_);
//			auto check_reshape = array_to_vec(net->blob_by_name("gtanet_q_values_reshape")->cpu_data(), minibatch_size_ * num_actions_);
//			auto check_eltwise = array_to_vec(net->blob_by_name("action_q_value")->cpu_data(), minibatch_size_ * num_actions_);
			std::vector<float> blank;
			return blank;
			auto results = net->output_blobs();
			out_array = results[0]->cpu_data();
			return array_to_vec(out_array, 1);
		}
		else
		{
			// Get targets - Q2
			// Forward to target/clone net (could replace net with memoized hash(input) -> output)
			// Analogous to memory of reward stored separately from experience processing. Cortex / Hippocampus
			clone_frames_input_layer_->AddMatVector(frames, labels);
			clone_action_input_layer_->Reset(&actions[0], &labels2[0], minibatch_size_);
			clone_target_input_layer_->Reset(&actions[0], &labels2[0], minibatch_size_); // Placeholder, we get layer before loss for targets
			net->ForwardPrefilled(nullptr);
			out_array = net->blob_by_name("gtanet_q_values")->cpu_data(); // TODO store blob object and reuse pointer
			auto check_q_values = array_to_vec(net->blob_by_name("gtanet_q_values")->cpu_data(), minibatch_size_ * num_actions_);
			auto check_reshape = array_to_vec(net->blob_by_name("gtanet_q_values_reshape")->cpu_data(), minibatch_size_ * num_actions_);
			auto check_eltwise = array_to_vec(net->blob_by_name("action_q_value")->cpu_data(), minibatch_size_ * num_actions_);
			return array_to_vec(out_array, minibatch_size_ * num_actions_);
		}
	}

	void purge_old_transitions()
	{
		if (transitions_->size() > replay_memory_)
		{
			// Make a best effort to keep transitions_ at hist_size 
			// while holding lock for as short a time possible.
			auto amount = transitions_->size() - replay_memory_;
			for (auto i = 0; i < amount; i++)
			{
				transitions_->release();
			}
		}
	}

	bool ready_to_learn()
	{
		return (
			(! should_train_async_ || get_queue_lock()) && 
			(transitions_->size() > minibatch_size_)
		);
	}

	void set_batch_size(int batch_size)
	{
		for(int i = 0; i < input_layers_.size(); i++)
		{
			(input_layers_[i])->set_batch_size(batch_size);
			(clone_input_layers_[i])->set_batch_size(batch_size);
		}
	}

	void QLearnMinibatch()
	{
		// Perform a minibatch Q-learning update:
		// w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
		if( ! ready_to_learn()) { return; }
		//while(true)
		//{
			set_batch_size(minibatch_size_);
			purge_old_transitions();
			auto transistion_sample = transitions_->sample(minibatch_size_);
			
			// Get target tensors for minibatch - r + gamma *  max_a2( Q(s2,a2) )
			if(iter_ % clone_iter_ == 0)
			{
				reset_clone_net();
			}

			// Forward s2
			auto is_s1 = false;
			std::vector<float> targets_dummy(num_actions_ * minibatch_size_);
			auto q2_all = feed_net(transistion_sample, clone_net_, is_s1, targets_dummy);

			// Get targets
			std::vector<float> targets(num_actions_ * minibatch_size_);
			std::fill(targets.begin(), targets.end(), 0.0f);
			for(int i = 0; i < minibatch_size_; i++)
			{
				Transition transition = transistion_sample[i];
				std::vector<float> q_2_sample;
				for(int j = i * num_actions_; j < ((i+1) * num_actions_); j++)
				{
					q_2_sample.push_back(q2_all[j]);
				}
				float q2_max = *std::max_element(q_2_sample.begin(), q_2_sample.end());
				float target = abs(transition.r + q2_max * discount_);
				targets[i * num_actions_ + transition.a] = target;
			}

	//		// TODO: Delete after figuring out why loss is zero on Q1 pass
	//		auto q1_all = net_->blob_by_name("gtanet_q_values")->cpu_data();
	//		auto q1_out = net_->blob_by_name("action_q_value")->cpu_data();

			// Set s1
			is_s1 = true;
			feed_net(transistion_sample, net_, is_s1, targets);

			//solver_->OnlineUpdate();

			// TODO: Backprop avg. delta as top gradient for all actions
			// TODO: Apply regularizer (weight cost?) by picking an SGD type like Adagrad
			// TODO: Step the learning rate down when loss goes down, or use Caffe's built-in stepsize by setting loss appropriately somewhere.
			// 
			set_batch_size(1);
			if(should_train_async_)
			{
				release_queue_lock();
			}			
		//}
	}
};

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

inline double get_random_double(float start, float end)
{
	static std::random_device rd;
	static std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, 1);
	return dist(e2);
}

// The function we want to execute on the new thread.
inline void train_minibatch_thread(NeuralQLearner* self)
{
	self->QLearnMinibatch();
}


inline int NeuralQLearner::select_action_e_greedy(double epsilon)
{
	int action;
	if(last_q_values_.size() == 0 || get_random_double(0, 1) < epsilon)
	{
		action = *select_randomly(actions_.begin(), actions_.end());
	}
	else
	{
		std::vector<int> best_actions;

		float max_q = std::numeric_limits<double>::min();
		int max_q_i = -1;
		for (auto i = 0; i < last_q_values_.size(); i++) {
			if(last_q_values_[i] > max_q)
			{
				max_q = last_q_values_[i];
				best_actions.clear();
				best_actions.push_back(i);
			}
			else if(last_q_values_[i] == max_q)
			{
				best_actions.push_back(i);
			}
		}

		if(best_actions.size() > 1)
		{
			action = *select_randomly(best_actions.begin(), best_actions.end());					
		} 
		else
		{
			action = best_actions[0];
		}
	}

	return action;
}

inline int NeuralQLearner::perceive(float reward, cv::Mat* raw_state, 
		bool terminal, bool testing, double epsilon)
{		
	// TODO: Store entire transition: s, a, r, s'
	if(last_state_ != nullptr && should_train_)
	{
		transitions_->add(last_state_, last_action_, reward, last_terminal_);
	}

	// TODO: Compute some validation statistics to judge performance

	// set the patch for testing
	std::vector<cv::Mat> frames;
	frames.push_back(*raw_state);
	std::vector<int> labels(frames.size());
	frames_input_layer_->AddMatVector(frames, labels);
	
	std::vector<float> target_input(num_actions_);
	std::fill(target_input.begin(), target_input.end(), 0.0f);
	//action_input_layer_->AddDatumVector()???

	// TODO: Select action greedily using prev_results - if prev_results is null, then select random.
	int action = select_action_e_greedy(epsilon);
	std::vector<float> actions;
	set_action_vector(action, actions);
	std::vector<float> labels2(num_actions_);
	std::fill(labels2.begin(), labels2.end(), 0.0f);
	action_input_layer_->Reset(&actions[0], &labels2[0], minibatch_size_);
	target_input_layer_->Reset(&target_input[0], &labels2[0], minibatch_size_);

	try
	{
		// Net forward
		net_->ForwardPrefilled(nullptr);
		const float * out_array = net_->blob_by_name("gtanet_q_values")->cpu_data(); // TODO store blob object and reuse pointer
//		const float* out_array = results[0]->cpu_data();
		// Store results in prev_results
		last_q_values_.erase(last_q_values_.begin(), last_q_values_.end());
		for(int i = 0; i < num_actions_; i++)
		{
			last_q_values_.push_back(out_array[i]);
		}
		action = select_action_e_greedy(epsilon);
	}
	catch(...)
	{
		LOG(INFO) << "Problem forwarding, most likely Eltwise product memory violation";
		// TODO: Figure out why this is happening, where exactly (reshape or action), how often it happens and fix it.
		// TODO: If this gets ported off windows, may need to do this for handling memory exceptions: http://stackoverflow.com/a/918891/134077
		action = last_action_;
	}

	if(should_train_ && iter_ % train_iter_ == 0)
	{
		if(should_train_async_)
		{
			std::thread(train_minibatch_thread, this).detach();	
		}
		else
		{
			try
			{
				QLearnMinibatch();
			}
			catch(const std::exception &exc)
			{
				LOG(INFO) << exc.what();
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				QLearnMinibatch();
			}
			catch(...)
			{
				LOG(INFO) << "error training, trying again";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				QLearnMinibatch();
			}
		}
	}

	if(iter_  % 1000 == 0)
	{
		LOG(INFO) << "iteration: " << iter_;
	}


	//deleteme//std::thread t1(train_minibatch_thread, this);
	//deleteme//t1.join();

	// TODO: Return the action selected
	
	last_state_ = raw_state;
	last_action_ = action;
	last_terminal_ = false;

	if(!should_train_)
	{
		// Image history gets deleted in QLearnMinibatch() during training
		(*raw_state).release();
		delete raw_state;
	}

	iter_ += 1;

	return action;
}


//inline void train_minibatch_thread(NeuralQLearner* neural_q_learner)
//{
////	NeuralQLearner x = static_cast<NeuralQLearner>(neural_q_learner);
//}

//inline void NeuralQLearner::perceive(cv::Mat* m)
//{
//	transition_table_->add_recent_state(m);
//	if(transition_table_->size() > minibatch_size_ && ! currently_learning_)
//	{
//		// TODO: Learn minibatch in another thread
//
////			NeuralQLearner* self = &this;
//		std::thread minibatch_thread (train_minibatch_thread);
//	}
//}

}

#endif // DQN_NEURAL_Q_LEANER_H

//NeuralQLearner::NeuralQLearner()
//{
//}
//
//NeuralQLearner::~NeuralQLearner()
//{
//}
//
//}
//
//int main_disabled(int argc, char** argv) {
//	cuda::GpuMat;
//	Mat img = imread(CAFFE_ROOT + "/examples/images/mnist_5.png");
//
//	cvtColor(img, img, CV_BGR2GRAY);
//    imshow("img", img);
//	cv::waitKey(1);
//
//	// Set up Caffe
//	Caffe::set_mode(Caffe::CPU);
//
//	// Load net
//    Net<float> net(CAFFE_ROOT + "/examples/mnist/lenet_test-memory-1.prototxt");
//    string model_file = CAFFE_ROOT + "/examples/mnist/lenet_iter_10000.caffemodel";
//    net.CopyTrainedLayersFrom(model_file);
//
//    // set the patch for testing
//    vector<Mat> patches;
//    patches.push_back(img);
//
//	// push vector<Mat> to data layer
//    float loss = 0.0;
//    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
//    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net.layer_by_name("data"));
//    
//    vector<int> labels(patches.size());
//    memory_data_layer->AddMatVector(patches, labels);
//
//    // Net forward
//    const vector<Blob<float>*> & results = net.ForwardPrefilled(&loss);
//    float *output = results[1]->mutable_cpu_data();
//
//    // Display the output
//    for (int i = 0; i < 10; i++) {
//        printf("Probability to be Number %d is %.3f\n", i, output[i]);
//    }
//    waitKey(0);
//
//
//
//	string filename_str = "C:\\Users\\a\\test.txt";
//
//	const char* filename2 = filename_str.c_str();
//	std::ifstream hFile(filename2);
//
//	std::vector<std::string> lines;
//	std::string line;
//	while(std::getline(hFile, line)) {
//		lines.push_back(line);
//	}
//
//
//	return 0;
//}