#pragma once
#ifndef AGENT_NET_H
#define AGENT_NET_H

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
#include "deep_drive.h"

// set caffe root path manually
const std::string CAFFE_ROOT = "../caffe";

namespace deep_drive
{

// void train_minibatch_thread(void* neural_q_learner);

class Agent
{
	TransitionQueue* transitions_;
	int minibatch_size_ = 1; 
	int learn_minibatch_size_;  
	int replay_size_;
	std::atomic<bool> queue_lock_;
	caffe::Net<float>* net_;
//	boost::shared_ptr<caffe::Net<float>> clone_net_;
	caffe::Solver<float>* solver_;
	boost::shared_ptr<caffe::Blob<float>> frames_input_blob_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> frames_input_layer_;
	boost::shared_ptr<caffe::MemoryDataLayer<float>> target_input_layer_;
//	boost::shared_ptr<caffe::MemoryDataLayer<float>> vehicle_states_input_layer_;
//	boost::shared_ptr<caffe::MemoryDataLayer<float>> reshape_layer_;
//	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_frames_input_layer_;
//	boost::shared_ptr<caffe::MemoryDataLayer<float>> clone_target_input_layer_;
	std::vector<boost::shared_ptr<caffe::MemoryDataLayer<float>>> input_layers_;
//	std::vector<boost::shared_ptr<caffe::MemoryDataLayer<float>>> clone_input_layers_;
	int num_output_;
	std::vector<int> actions_;
	cv::Mat* last_state_;
	Action   last_action_;
//	int raw_frame_width_ = 771;
//	int frame_area_ = raw_frame_width_ * raw_frame_width_;
//	int sample_frame_count_ = 4;
//	int sample_data_size_ = frame_area_ * sample_frame_count_;
//	int minibatch_data_size_ = sample_data_size_ * minibatch_size_;
	std::vector<float> last_action_values_;
	bool should_train_;
	bool should_train_async_;
	bool should_manually_set_acceleration_;
	int train_iter_;
//	int clone_iter_;
	long iter_ = 0;
	bool debug_info_;
	SharedAgentControlData* shared_agent_control_;
	SharedRewardData* shared_reward_;
	double replay_chance_;
	int purge_every_;
	bool should_skip_update_;
	bool should_fill_replay_memory_;

	public:
	Agent(int replay_size, int learn_minibatch_size,
		int num_output, std::string solver_path, std::string model_path,
		bool should_train, int train_iter,
		bool should_train_async,
//		int clone_iter, 
		SharedAgentControlData* shared_agent_control,
		SharedRewardData* shared_reward, std::string resume_solver_path, bool debug_info,
		double replay_chance, int purge_every, bool should_skip_update, bool should_fill_replay_memory)
	{
		transitions_ = new TransitionQueue();
		learn_minibatch_size_ = learn_minibatch_size;
		replay_size_ = replay_size;
		num_output_ = num_output;
		should_train_ = should_train; // TODO: Change to should_train
		should_train_async_ = should_train_async;
		train_iter_ = train_iter;
//		clone_iter_ = clone_iter;
		shared_agent_control_ = shared_agent_control;
		shared_reward_ = shared_reward;
		debug_info_ = debug_info;
		replay_chance_ = replay_chance;
		purge_every_ = purge_every;
		should_skip_update_ = should_skip_update;
		should_fill_replay_memory_ = should_fill_replay_memory;

		for(auto i = 0; i < num_output; i++)
		{
			actions_.push_back(i);
		}
		queue_lock_ = false;

		// parse solver parameters
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_path, &solver_param);

		// set device id and mode 
		caffe::Caffe::SetDevice(0);
		caffe::Caffe::set_mode(caffe::Caffe::GPU);

		if(should_train)
		{
			// solver handler
			solver_ = caffe::SolverRegistry<float>::CreateSolver(solver_param);
			if(resume_solver_path != "")
			{
				solver_->Restore(resume_solver_path.c_str());
			}
			net_ = boost::dynamic_pointer_cast<caffe::Net<float>>(solver_->net()).get();
		}
		else
		{
			solver_ = nullptr;
			net_ = new caffe::Net<float>(CAFFE_ROOT + "/" + model_path, caffe::TEST);
		}

		
		frames_input_blob_ = net_->blob_by_name("gta_frames_input_layer");

		frames_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net_->layer_by_name("gta_frames_input_layer"));
		target_input_layer_ =
			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
				net_->layer_by_name("target_input_layer"));
//		vehicle_states_input_layer_ =
//			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
//				net_->layer_by_name("vehicle_states_input_layer"));
//		reshape_layer_ =
//			boost::dynamic_pointer_cast<caffe::ReshapeLayer<float>>(
//				net_->layer_by_name("reshape"));


 		assert(frames_input_layer_);
		assert(target_input_layer_);
//		assert(vehicle_states_input_layer_);
//		assert(reshape_layer_);

		input_layers_.push_back(frames_input_layer_);
		input_layers_.push_back(target_input_layer_);

//		caffe::NetParameter net_param;
//		ReadNetParamsFromTextFileOrDie("examples/deep_drive/deep_drive_model.prototxt", &net_param);
//		clone_net_.reset(new caffe::Net<float>(net_param));
//		reset_clone_net();

		net_->set_debug_info(debug_info_);
//		clone_net_->set_debug_info(false);

		// solver_->OnlineUpdateSetup(nullptr);
	}

//	void reset_clone_net()
//	{
//		caffe::NetParameter net_param;
//		net_->ToProto(&net_param);
//		net_param.mutable_state()->set_phase(caffe::TEST);
//		clone_net_->CopyTrainedLayersFrom(net_param);
//
//		clone_frames_input_layer_ =
//			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
//				clone_net_->layer_by_name("gta_frames_input_layer"));
//		clone_target_input_layer_ =
//			boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
//				clone_net_->layer_by_name("target_input_layer"));
////		clone_reshape_layer_ =
////			boost::dynamic_pointer_cast<caffe::ReshapeLayer<float>>(
////				net_->layer_by_name("reshape"));
//
// 		assert(clone_frames_input_layer_);
//		assert(clone_target_input_layer_);
////		assert(clone_reshape_layer_);
//
//		clone_input_layers_.push_back(clone_frames_input_layer_);
//		clone_input_layers_.push_back(clone_target_input_layer_);
//		output("Cloned net");
//	}

	bool get_should_save_experiences()
	{
		return should_fill_replay_memory_;
	}

	void load_weights(const std::string& weight_file)
	{
		net_->CopyTrainedLayersFrom(CAFFE_ROOT + "/" + weight_file);
	}

	bool get_should_train()
	{
		return this->should_train_;
	}

	~Agent()
	{
		delete transitions_;
		delete net_;
//		net_->reset();
		if(solver_ != nullptr)
		{
			delete solver_;
		}

		for(int i = 0; i < input_layers_.size(); i++)
		{
			input_layers_[i].reset();
		}
//		for(int i = 0; i < clone_input_layers_.size(); i++)
//		{
//			clone_input_layers_[i].reset();
//		}
	}

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
		for(int i = 0; i < num_output_; i++)
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

	void log_targets_actuals(std::vector<float> check_target, std::vector<float> check_fctop)
	{
		int j = 0;
		for(int i = 0; i < check_target.size(); i++)
		{
			if(i % num_output_ == 0)
			{
				LOG(INFO) << "\n target " << j << "---------------";
				j++;
			}
			LOG(INFO) << "targets[" << i << "]: " << check_target[i];			
			LOG(INFO) << "actuals[" << i << "]: " << check_fctop[i] << "\n";
		}
	}

	void FeedNet(std::vector<Transition>& transistion_sample, 
		caffe::Net<float>* net, std::vector<float> &targets)
	{
		std::vector<cv::Mat> frames;
		std::vector<float> actions;

		for (auto i = 0; i < minibatch_size_; i++)
		{
			Transition transition = transistion_sample[i];
			frames.push_back(*(transition.image));

			// Frame input size is minibatch * frames_per_sample * sizeof(cv::Mat == w * h)
			// TODO: Set input channels with four consecutive frames for time inference or use frame difference for motion.
		}

		std::vector<int> labels(frames.size());
		std::vector<float> labels2(frames.size());

		std::fill(labels.begin(), labels.end(), 0);
		std::fill(labels2.begin(), labels2.end(), 0.0f);

		const float* out_array;

		std::vector<float> dummy_input;
		frames_input_layer_->AddMatVector(frames, labels);
		target_input_layer_->Reset(const_cast<float*>(targets.data()), &labels2[0], minibatch_size_);
//		vehicle_states_input_layer_->Reset(const_cast<float*>(vehicle_states.data()), &labels2[0], minibatch_size_);
		solver_->Step(1);
		auto check_target = array_to_vec(net->blob_by_name("target")->cpu_data(), minibatch_size_ * num_output_);
		auto check_fctop = array_to_vec(net->blob_by_name("gtanet_fctop")->cpu_data(), minibatch_size_ * num_output_);

		log_targets_actuals(check_target, check_fctop);

		return;
//		auto results = net->output_blobs();
//		out_array = results[0]->cpu_data();
//		return array_to_vec(out_array, 1);

	}

	void purge_old_transitions(int keep_count)
	{
		if (transitions_->size() > keep_count)
		{
			auto num_to_purge = transitions_->size() - keep_count;
			for (auto i = 0; i < num_to_purge; i++)
			{
				transitions_->release();
			}
		}
	}

	bool ready_to_learn(int replay_size)
	{
		return (
			(! should_train_async_ || get_queue_lock()) && 
			(transitions_->size() >= replay_size)
		);
	}

	void set_batch_size(int batch_size)
	{
		minibatch_size_ = batch_size;
		for(int i = 0; i < input_layers_.size(); i++)
		{
			(input_layers_[i])->set_batch_size(batch_size);
//			(clone_input_layers_[i])->set_batch_size(batch_size);
		}
	}

	void normalize_metrics(double spin, double& speed, double& direction)
	{
		speed = kSpeedCoefficient * speed;

		if(spin <= -0.01)
		{
			direction = -1;
		}
		else if(spin >= 0.01)
		{
			direction = 1;
		}
		else
		{
			direction = 0;
		}
	}

	void Learn()
	{
		int train_count = 0;
		purge_old_transitions(replay_size_);

		// Get target tensors for minibatch - r + gamma *  max_a2( Q(s2,a2) )

//		if(iter_ % clone_iter_ == 0)
//		{
//			reset_clone_net();
//		}

		while(train_count < train_iter_) // Set while(true) here to overfit on one batch
		{
			set_batch_size(learn_minibatch_size_);
			auto transistion_sample = transitions_->sample(minibatch_size_);

			// Get targets / vehicle states
			std::vector<float> targets       (num_output_ * minibatch_size_);

			// Set targets / vehicle states
			std::fill(targets.begin(), targets.end(), 0.0f);
//			std::fill(targets.begin(), vehicle_states.end(), 0.0f);

			for(int i = 0; i < minibatch_size_; i = i++)
			{
				Transition transition = transistion_sample[i];
				int j = i * num_output_;

				double speed = transition.speed;
				double spin = transition.spin;

				// Normalize speed in the loss function as its raw value is
				// an order of magnitude larger than spin and speed change and 
				// we want them to all contributing equally to the loss.
				double direction;
				normalize_metrics(spin, speed, direction);

//				targets[j + 0] = steer;
//				targets[j + 1] = throttle;
//				targets[j + 2] = transition.speed_change;
//				targets[j + 3] = direction;

//				vehicle_states[j + 0] = prev_spin;
//				vehicle_states[j + 1] = throttle;
//				vehicle_states[j + 2] = transition.speed_change;
//				vehicle_states[j + 3] = direction;
			}

			//		// TODO: Delete after figuring out why loss is zero on Q1 pass
			//		auto q1_all = net_->blob_by_name("gtanet_q_values")->cpu_data();
			//		auto q1_out = net_->blob_by_name("action_q_value")->cpu_data();

			FeedNet(transistion_sample, net_, targets);

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
			train_count++;
		}
	}

	void LearnIfExperienced()
	{
		// Perform a minibatch Q-learning update:
		// w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
//		(*shared_agent_control_).should_reload_game = true; // Reload while training.
		bool learned = false;
		if(transitions_->size() >= replay_size_)
		{
			wait_to_reset_game_mod_options(shared_reward_); // Getting stuck on pause waiting for this, not sure why but make sure reset happens before pause.
			wait_to_toggle_pause_game(shared_agent_control_);
			Learn();
			learned = true;
		}
		purge_old_transitions(replay_size_ - train_iter_);
//		reset_agent();
		if(learned)
		{
			wait_to_toggle_pause_game(shared_agent_control_);
		}
	}

	int select_action(std::vector<float> action_values)
	{
		// Choose the max value with random tie breaking
		int action;
		std::vector<int> best_actions;

		float max_q = -std::numeric_limits<double>::max();
		int max_q_i = -1;
		for (auto i = 0; i < action_values.size(); i++) {
			if(action_values[i] > max_q)
			{
				max_q = action_values[i];
				best_actions.clear();
				best_actions.push_back(i);
			}
			else if(action_values[i] == max_q)
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
		return action;
	}

	void AddToReplayMemory(double spin, double speed, double speed_change, double steer, double throttle)
	{
		if(last_state_ != nullptr && should_fill_replay_memory_)
		{
			if(iter_ != 0 && iter_ % purge_every_ == 0)
			{
				purge_old_transitions(replay_size_);
			}
			if(get_random_double(0.0, 1.0) <= replay_chance_)
			{
				transitions_->add(last_state_, spin, speed, 
				                  speed_change, steer, throttle);
				//				saveInput(shared_reward_, iter_, true, 1, raw_state);
			}
		}
	}

	Action Forward(cv::Mat* raw_state, double spin, double speed, double speed_change, double steer, double throttle)
	{
		Action next_action;
		std::vector<cv::Mat> frames;
		frames.push_back(*raw_state);
		std::vector<int> labels(frames.size());
		frames_input_layer_->AddMatVector(frames, labels);
	
		// Targets are inneffectual here, just setting them to check against actuals.
		double direction;
		normalize_metrics(spin, speed, direction);
		std::vector<float> targets;
//		std::vector<float> vehicle_states;
		
//		for(int i = 0; i < num_output_; i++)
//		{
//			targets.push_back(spin);
//		}

		double prev_direction;
		auto prev = transitions_->previous();

		if(prev.image == nullptr)
		{
//			next_action.spin = 0;
//			next_action.direction = 0;
//			next_action.speed = 0;
//			next_action.speed_change = 0;
//			next_action.steer = 0;
//			next_action.throttle = 0;
//			return next_action;

//			vehicle_states.push_back(0);
//			vehicle_states.push_back(0);
//			vehicle_states.push_back(0);
//			vehicle_states.push_back(0);
//			vehicle_states.push_back(0);
//			vehicle_states.push_back(0);
		}
		else
		{
			normalize_metrics(prev.spin, prev.speed, prev_direction);
//			vehicle_states.push_back(prev.spin);
//			vehicle_states.push_back(prev_direction);
//			vehicle_states.push_back(prev.speed);
//			vehicle_states.push_back(prev.speed_change);
//			vehicle_states.push_back(prev.steer);
// 			vehicle_states.push_back(prev.throttle);			
		}



		targets.push_back(0);
		targets.push_back(0);
		targets.push_back(0);
		targets.push_back(0);
		targets.push_back(0);  // These are not normalized so are completely incorrect. Just using as placeholder. Targets get checked elsewhere now (during offline training and in control AutoItx while online).
		targets.push_back(0);  // These are not normalized so are completely incorrect. Just using as placeholder. Targets get checked elsewhere now (during offline training and in control AutoItx while online).


//		std::fill(target_input.begin(), target_input.end(), 0.0f);
		//	set_action_vector(action, actions);
		std::vector<float> labels2(num_output_);
		std::fill(labels2.begin(), labels2.end(), 0.0f);
		target_input_layer_->Reset(&targets[0], &labels2[0], minibatch_size_);
//		vehicle_states_input_layer_->Reset(&vehicle_states[0], &labels2[0], minibatch_size_);
		try
		{
			// Net forward
			net_->ForwardPrefilled(nullptr);
			const float * out_array = net_->blob_by_name("gtanet_fctop")->cpu_data(); // TODO store blob object and reuse pointer
//			auto check_fctop = array_to_vec(out_array, num_output_);
//			log_targets_actuals(targets, check_fctop);

			//		const float* out_array = results[0]->cpu_data();
			// Store results in prev_results
			last_action_values_.erase(last_action_values_.begin(), last_action_values_.end());
			for(int i = 0; i < num_output_; i++)
			{
				last_action_values_.push_back(out_array[i]);
//				if(iter_ % 40 == 0)
//				{
//					LOG(INFO) << "action values " << i << " = " << out_array[i];
//				}
			}
			next_action.spin         = last_action_values_[0];
			next_action.direction    = last_action_values_[1];
			next_action.speed        = last_action_values_[2];
			next_action.speed_change = last_action_values_[3];
			next_action.steer        = last_action_values_[4];
			next_action.throttle     = last_action_values_[5];
		}
		catch(...)
		{
			LOG(INFO) << "Problem forwarding, most likely Eltwise product memory violation";
			// TODO: Figure out why this is happening, where exactly (reshape or action), how often it happens and fix it.
			// TODO: If this gets ported off windows, may need to do this for handling memory exceptions: http://stackoverflow.com/a/918891/134077
			next_action = last_action_;
		}
		return next_action;
	}

	void PossiblyLearn(Action& next_action)
	{
		if(should_train_ && iter_ % train_iter_ == (train_iter_ - 1))
		{
			// Next iteration will be training, go to sleep by setting action to no-op
			next_action.spin = 0;
			next_action.speed = last_action_.speed;			
			next_action.speed_change = 0;
		}

		if(should_skip_update_)
		{
			// Temporary hack to test training net
			return;
		}

		if(should_train_ && iter_ % train_iter_ == 0 && iter_ != 0)
		{
			if(should_train_async_)
			{
				// Need to inline this method after class declaration to have access to train_minibatch_thread
				//				std::thread(train_minibatch_thread, this).detach();	
			}
			else
			{
				try
				{
					LearnIfExperienced();
				}
				catch(const std::exception &exc)
				{
					LOG(INFO) << exc.what();
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					LearnIfExperienced();
				}
				catch(...)
				{
					LOG(INFO) << "error training, trying again";
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					LearnIfExperienced();
				}
			}
		}
	}

	Action Perceive(cv::Mat* screen, double spin, double speed, double speed_change, float steer, float throttle)
	{
		AddToReplayMemory(spin, speed, speed_change, steer, throttle);
		Action next_action = Forward(screen, spin, speed, speed_change, steer, throttle);
		PossiblyLearn(next_action);

		if(iter_  % 100 == 0)
		{
			LOG(INFO) << "iteration: " << iter_;
		}
	
		last_state_ = screen;
		last_action_ = next_action;

		iter_ += 1;

		return next_action;
	}

	int infer_action(double& accumulated_spin, double desired_speed, double current_speed)
	{
		bool spin_is_small = (accumulated_spin <= kAccumulatedSpinThreshold) && (accumulated_spin >= -kAccumulatedSpinThreshold);
		float speed_diff = desired_speed - current_speed;
		float speed_span = abs(speed_diff);
		
		if(spin_is_small && speed_span <= kSpeedThreshold)
		{
			// None
			return 0;
		}
		else if(accumulated_spin > kAccumulatedSpinThreshold && speed_span <= kSpeedThreshold)
		{
			// Left
			accumulated_spin = 0;
			return 1;
		}
		else if(accumulated_spin < -kAccumulatedSpinThreshold && speed_span <= kSpeedThreshold)
		{
			// Right
			accumulated_spin = 0;
			return 2;
		}
		else if(spin_is_small && speed_diff > kSpeedThreshold)
		{
			// Forward
			return 3;
		}
		else if(spin_is_small && speed_diff < kSpeedThreshold)
		{
			// Backward
			return 4;
		}
		else if(accumulated_spin > kAccumulatedSpinThreshold && speed_diff > kSpeedThreshold)
		{
			// Left forward
			accumulated_spin = 0;
			return 5;
		}
		else if(accumulated_spin > kAccumulatedSpinThreshold && speed_diff < kSpeedThreshold)
		{
			// Left backward
			accumulated_spin = 0;
			return 6;
		}
		else if(accumulated_spin < -kAccumulatedSpinThreshold && speed_diff > kSpeedThreshold)
		{
			// Right forward
			accumulated_spin = 0;
			return 7;
		}
		else if(accumulated_spin < -kAccumulatedSpinThreshold && speed_diff < kSpeedThreshold)
		{
			// Right backward
			accumulated_spin = 0;
			return 8;
		} 
		else
		{
			throw std::invalid_argument( "Action could not be inferred." );
		}
	}
};

// The function we want to execute on the new thread.
inline void train_minibatch_thread(Agent* self)
{
	self->LearnIfExperienced();
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

#endif // AGENT_NET_H

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