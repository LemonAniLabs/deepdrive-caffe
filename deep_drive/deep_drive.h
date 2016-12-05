#ifndef DEEP_DRIVE_H
#define DEEP_DRIVE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <random>
#include <caffe/util/io.hpp>
#include <chrono>
#include <thread>
#include <direct.h>

namespace deep_drive {

	inline std::string get_env_var( std::string const & key )
	{
		char * val = getenv( key.c_str() );
		return val == NULL ? std::string("") : std::string(val);
	}

	const bool kShouldPrintTimeSince = get_env_var("DEEPDRIVE_PRINT_TIME_SINCE") == "true";

	const int kSaveDataStep = 0; // 88986; // 63361; // Using folder now, first dataset has different sessions at these frames though.
	auto kMsInStep = 125;
	auto kStepDuration = std::chrono::milliseconds(kMsInStep); // Used to keep training data size down. With GeForce GTX 980, we can get to 50Hz or 20ms.


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

	const double kSpeedCoefficient = 0.05; // Also in clean_deep_drive_data.py in deep_drive branch
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
		double desired_steer;
		double desired_throttle;
	};

	#define REWARD_SHARED_MEMORY TEXT("Global\\AgentReward")

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
		bool should_game_drive;
		bool should_perform_random_action;
		bool is_game_driving;
		int temp_action;
	};

	#define SCREEN_IMAGE_SHARED_MEMORY TEXT("Global\\ScreenImage")
	struct SharedScreenData
	{
		int width;
		int height;
		int stride;
		bool should_toggle_pause_game;
		bool should_agent_wait;
		int sampleCount;
		BYTE imageData[684 * 227];
	};
	// Screen shared memory


	inline void output(const char* out_string)
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

	std::random_device random_seed;     // only used once to initialise (seed) engine
	std::mt19937 random_generator(random_seed());    // random-number engine used (Mersenne-Twister in this case)

	template<typename Iter>
	Iter select_randomly(Iter start, Iter end) {
		return select_randomly(start, end, random_generator);
	}

	inline int get_random_int(int start, int end)
	{
		std::uniform_int_distribution<int> uni(start, end); // guaranteed unbiased
		auto random_integer = uni(random_generator);
		return random_integer;
	}

	inline double get_random_double(double start, double end)
	{
		std::uniform_real_distribution<double> uni(start, end); // guaranteed unbiased
		auto random_double = uni(random_generator);
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
//		wait_to_reset_game_mod_options(shared_reward_memory);
	}

	inline void print_time_since(std::chrono::system_clock::time_point start_time, std::string name)
	{
		if(kShouldPrintTimeSince)
		{
			typedef std::chrono::milliseconds ms;
			auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
			auto elapsed_ms = std::chrono::duration_cast<ms>(elapsed);
			output((name + " duration: " + std::to_string(elapsed_ms.count()) + "\n").c_str());			
		}
	}

	inline bool time_left_in_step(std::chrono::system_clock::time_point start_time)
	{
		typedef std::chrono::milliseconds ms;
		auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
		auto elapsed_ms = std::chrono::duration_cast<ms>(elapsed);
		auto extra_wait = kStepDuration - elapsed_ms;
		LOG(INFO) << "Time left in step: " << extra_wait.count();
		if(extra_wait.count() > 5)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return true;
		}
		else
		{
			return false;
		}
	}

	inline void enforce_period(std::chrono::system_clock::time_point start_time)
	{
		// Used to keep training data size down. With GeForce GTX 980, we can get to 50Hz.
		typedef std::chrono::milliseconds ms;
		auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
		auto elapsed_ms = std::chrono::duration_cast<ms>(elapsed);
		auto extra_wait = kStepDuration - elapsed_ms;
		LOG(INFO) << "elapsed ms " << elapsed_ms.count();
		std::this_thread::sleep_for(extra_wait);
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
		int step_in, std::string folder_name, std::chrono::system_clock::time_point screen_cap_start)
	{

		int step = step_in + kSaveDataStep; // 63361; // 88986
		std::ofstream myfile;
		myfile.open (folder_name + "dat_" + std::to_string(step) + ".txt");

		myfile << "step: " << std::to_string(step)                                                           << 
			", spin: "         << std::to_string(shared_reward_data->spin)                                    <<
			", speed: "        << std::to_string(shared_reward_data->speed)                                   <<
			", steer: "        << std::to_string(shared_agent_data->steer)                                    << 
			", throttle: "     << std::to_string(shared_agent_data->throttle)                                 << 
			", img_time: "     << std::to_string(screen_cap_start.time_since_epoch().count())                 << // B,Mse,kse,sec,mil,mic,n
			", metric_time: "  << std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) << // B,Mse,kse,sec,mil,mic,n
			", random_action: " << std::to_string( ! shared_reward_data->is_game_driving);
		myfile.close();
	}

	inline void saveInput(SharedRewardData* shared_reward_data, SharedAgentControlData* shared_agent_data,
		int step, bool should_save_data, 
		int save_input_every, cv::Mat* screen, std::chrono::system_clock::time_point screen_cap_start)
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
				auto save_meta_start = std::chrono::system_clock::now();
				saveMeta(shared_reward_data, shared_agent_data, step, save_input_folder_name, screen_cap_start);
				saveImage(*screen, step, save_input_folder_name);

				// Writing a file every frame takes about 1ms.
//				print_time_since(save_meta_start,  "save meta");
//				print_time_since(screen_cap_start, "screen cap to save meta");
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

	inline std::chrono::system_clock::time_point wait_to_reload_game(SharedAgentControlData* shared_agent_data, SharedRewardData* shared_reward_data, bool should_save_data)
	{
		do
		{
			output("Waiting to reload game...");
			shared_reward_data->should_game_drive = false;
//			shared_reward_data->temp_action = 1; // Brake
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		} while ((*shared_agent_data).should_reload_game == true);


		if(should_save_data)
		{
			shared_reward_data->should_game_drive = true;	
		}
		else
		{
			shared_reward_data->should_game_drive = false;
		}
		auto game_start_time = std::chrono::high_resolution_clock::now();
		return game_start_time;
	}

	inline std::chrono::system_clock::time_point reload_game(SharedAgentControlData* shared_agent_data, SharedRewardData* shared_reward_data, bool should_save_data)
	{
		(*shared_agent_data).should_reload_game = true;
		return wait_to_reload_game(shared_agent_data, shared_reward_data, should_save_data);		
	}

	inline std::chrono::system_clock::time_point reload_game_based_on_duration(std::chrono::minutes game_duration, std::chrono::system_clock::time_point game_start_time, SharedAgentControlData* shared_agent_data, SharedRewardData* shared_reward_data, bool should_save_data, int &step)
	{
		typedef std::chrono::milliseconds ms;
		auto elapsed = std::chrono::high_resolution_clock::now() - game_start_time;
		auto elapsed_ms = std::chrono::duration_cast<ms>(elapsed);
		if(elapsed_ms >= game_duration)
		{
			step = 0;
			should_create_saved_input_folder = true;
			return reload_game(shared_agent_data, shared_reward_data, should_save_data);
		}
		else
		{
			step++;
			return game_start_time;
		}
	}

}

#endif  // DEEP_DRIVE_H