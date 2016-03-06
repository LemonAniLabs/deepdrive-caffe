#ifndef DQN_H
#define DQN_H

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

namespace dqn{
	#define AGENT_CONTROL_SHARED_MEMORY TEXT("Local\\AgentControl")
	struct SharedAgentControlData
	{
		INT32 action;
		bool paused;
		LONGLONG step;
		bool should_reload_game;
	};

	#define REWARD_SHARED_MEMORY TEXT("Local\\AgentReward")

	struct SharedRewardData
	{
		double distance;
		bool on_road;
		bool should_reset_agent;
	};


	void output(const char* out_string)
	{
		OutputDebugStringA(out_string);
		LOG(INFO) << out_string;
	}
}

#endif  // DQN_H