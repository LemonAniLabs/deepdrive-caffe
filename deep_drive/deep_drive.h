#ifndef DEEP_DRIVE_H
#define DEEP_DRIVE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <random>
#include <caffe/util/io.hpp>
#include <chrono>
#include <thread>
#include <direct.h>

namespace deep_drive{
	const int kSaveDataStep = 0; // 88986; // 63361; // Using folder now, first dataset has different sessions at these frames though.


	// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
	inline std::string current_date_time() {
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
		// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
		// for more information about date/time format
		strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", &tstruct);

		return buf;
	}

	std::string save_input_folder_name;
	bool should_create_saved_input_folder = true;

	const double kSpeedCoefficient = 0.05;
	const double kAccumulatedSpinThreshold = 0.875;
	const double kSpeedThreshold = 2;


	#define AGENT_CONTROL_SHARED_MEMORY TEXT("Local\\AgentControl")
	struct SharedAgentControlData
	{
		INT32 action;
		bool should_agent_wait;
		LONGLONG step;
		bool should_reload_game;
		bool should_toggle_pause_game;
		double desired_spin;
		double desired_speed;
		double desired_speed_change;
		double desired_direction;
		double actual_spin;
		double actual_speed;
		double actual_speed_change;
		bool heading_achieved;
		bool speed_achieved;
		float steer;
		float throttle;
	};

	#define REWARD_SHARED_MEMORY TEXT("Local\\AgentReward")

	struct SharedRewardData
	{
		double distance;
		bool on_road;
		bool should_reset_agent;
		double heading;
		double speed;
		double desired_spin; // for directly setting spin, intermediate step to real control
		double desired_speed; // for directly setting speed, intermediate step to real control
		double desired_speed_change; // for directly setting speed change, intermediate step to real control
		double desired_direction;
		double spin;
	};


	void output(const char* out_string)
	{
		OutputDebugStringA(out_string);
		LOG(INFO) << out_string;
	}

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

	inline int get_random_int(int start, int end)
	{
		std::random_device rd;     // only used once to initialise (seed) engine
		std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_int_distribution<int> uni(start, end); // guaranteed unbiased

		auto random_integer = uni(rng);

		return random_integer;
	}

	inline double get_random_double(double start, double end)
	{
		std::random_device rd;     // only used once to initialise (seed) engine
		std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_real_distribution<double> uni(start, end); // guaranteed unbiased

		auto random_double = uni(rng);

		return random_double;
	}

	struct Action
	{
		double spin;
		double speed; // Norm of 3D speed
		double speed_change;
		double direction;
		float  steer;
		float  throttle;
	};

	inline void wait_to_reset_game_mod_options(SharedRewardData* shared_reward_memory)
	{
		while ((*shared_reward_memory).should_reset_agent == true)
		{
			output("Waiting to reset game mod options...");
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		} 
	}

	inline void reset_game_mod_options(SharedRewardData* shared_reward_memory)
	{
		(*shared_reward_memory).should_reset_agent = true;
		wait_to_reset_game_mod_options(shared_reward_memory);
	}

	inline void saveImage(cv::Mat img, int iter, std::string folder_name)
	{
		int step = iter + kSaveDataStep;
		if (! img.data) // Check for invalid input
		{
			std::cout << "No image data" << std::endl ;
			return;
		}
		// Save the frame into a file
		cv::imwrite(folder_name + "img_" + std::to_string(step) + ".bmp", img);
	}

	inline void saveMeta(SharedRewardData* shared_reward_data, SharedAgentControlData* shared_agent_data,
		int step_in, std::string folder_name)
	{
		int step = step_in + kSaveDataStep; // 63361; // 88986
		std::ofstream myfile;
		myfile.open (folder_name + "dat_" + std::to_string(step) + ".txt");
		myfile << "step: " <<  std::to_string(step) << 
			", spin: "     << std::to_string(shared_reward_data->spin)  <<
			", speed: "    << std::to_string(shared_reward_data->speed) <<
			", steer: "    << std::to_string(shared_agent_data->steer)  << 
			", throttle: " << std::to_string(shared_agent_data->throttle);
		myfile.close();
	}

	inline void saveInput(SharedRewardData* shared_reward_data, SharedAgentControlData* shared_agent_data,
		int step, bool should_save_data, 
		int save_input_every, cv::Mat* screen)
	{
		if(should_save_data)
		{
			if(should_create_saved_input_folder)
			{
				 save_input_folder_name = "D:\\data\\gtav\\4hz_spin_speed_001\\" + current_date_time() + "\\";
				_mkdir(save_input_folder_name.data());
				should_create_saved_input_folder = false;
			}

			if(step % save_input_every == 0)
			{
				saveImage(*screen, step, save_input_folder_name);
			}

			if(step % save_input_every == 0)
			{
				saveMeta(shared_reward_data, shared_agent_data, step, save_input_folder_name);
			}				
		}
	}

	inline void wait_to_toggle_pause_game(SharedAgentControlData* shared_agent_control)
	{
		(*shared_agent_control).should_toggle_pause_game = true;
		do
		{
			output("Waiting to pause game...");
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		} while ((*shared_agent_control).should_toggle_pause_game == true);
	}

	inline void wait_to_reload_game(SharedAgentControlData* shared_agent_control)
	{
		do
		{
			output("Waiting to reload game...");
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		} while ((*shared_agent_control).should_reload_game == true);
	}
}

#endif  // DEEP_DRIVE_H