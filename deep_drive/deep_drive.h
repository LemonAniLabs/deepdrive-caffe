#ifndef DEEP_DRIVE_H
#define DEEP_DRIVE_H

#include <deque>
#include <opencv2/core/mat.hpp>

//class MyClass
//{
//public:
//	MyClass();
//	~MyClass();
//private:
//
//};
//
//inline MyClass::MyClass()
//{
//	int y = 1;
//}
//
//inline MyClass::~MyClass()
//{
//	int x = 1;
//}

namespace deep_drive{
	#define AGENT_CONTROL_SHARED_MEMORY TEXT("Local\\AgentControl")
	struct SharedAgentControlData
	{
		INT32 action;
		bool should_agent_wait;
		LONGLONG step;
		bool should_reload_game;
		bool should_toggle_pause_game;
		double heading_change;
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
		double desired_heading; // for directly setting heading, intermediate step to real control
		double desired_speed; // for directly setting speed, intermediate step to real control
		double rotational_velocity;
	};


	void output(const char* out_string)
	{
		OutputDebugStringA(out_string);
		LOG(INFO) << out_string;
	}
}

#endif  // DEEP_DRIVE_H