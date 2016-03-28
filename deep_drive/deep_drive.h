#ifndef DEEP_DRIVE_H
#define DEEP_DRIVE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <random>
#include <caffe/util/io.hpp>
#include <chrono>
#include <thread>

namespace deep_drive{
	 int kSaveDataStep = 0; // 88986; // 63361; // Using folder now, first dataset has different sessions at these frames though.

	 double kSpeedCoefficient = 1 / 20;

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
		double speed_change;
		bool heading_achieved;
		bool speed_achieved;
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

	struct Action
	{
		double spin;
		double speed; // Norm of 3D speed
		double speed_change;
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

	// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
	inline std::string current_date_time() {
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
		// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
		// for more information about date/time format
		strftime(buf, sizeof(buf), "%Y-%m-%d.%S", &tstruct);

		return buf;
	}
}

#endif  // DEEP_DRIVE_H