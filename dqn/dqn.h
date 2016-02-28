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
		BOOLEAN paused;
		LONGLONG step;
	};

	#define REWARD_SHARED_MEMORY TEXT("Local\\AgentReward")

	struct SharedRewardData
	{
		INT32 distance;
		bool on_road;
		bool reset_agent_position;
	};


	void output(const char* out_string)
	{
		OutputDebugStringA(out_string);
		LOG(INFO) << out_string;
	}
}

#endif  // DQN_H