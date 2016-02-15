#pragma once
#ifndef DQN_TRANSITION_TABLE_H
#define DQN_TRANSITION_TABLE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <array>

namespace dqn
{

struct Transition
{
	cv::Mat* s;
	int      a;
	float    r;
	cv::Mat* s2;
	bool     t;

};

class TransitionQueue
{		
	int state_dim_;
	int replay_memory_;
	std::deque<cv::Mat*> s_;
	std::deque<int>      a_;
	std::deque<float>    r_;
	std::deque<bool>     t_;

	public:
	TransitionQueue(int state_dim, int replay_memory)
	{
		state_dim_ = state_dim;
		replay_memory_ = replay_memory;

		// State queue
		s_ = std::deque<cv::Mat*>(); // No size param, let it grow lazily.
	}

	~TransitionQueue() {}

	void add(cv::Mat* m, int action, float reward, bool terminal) 
	{
		s_.push_back(m);
		a_.push_back(action);
		r_.push_back(reward);
		t_.push_back(terminal);
	}

	int size() const
	{
		return s_.size();
	}

	void release()
	{
		release_front(s_);
		a_.pop_front();
		r_.pop_front();
		t_.pop_front();
	}

	template <typename Dtype>
	void release_front(std::deque<Dtype>& c)
	{
		delete c[0];
		c.pop_front();
	}

	Transition sample_one()
	{
		auto valid = false;
		// Get some random starting point and make sure the size left is okay.
		auto start = 0;
		while (!valid)
		{
			// TODO: Use better random here.
			start = 1 + (rand() % s_.size() - 1); // start at 1 because of previous action

			// TODO: Make sure we are not at a terminal state
			// TODO: Discard non-terminal *next*-states (we know first state of transition is non-terminal)
			//       with prob nonTermProb.
			// TODO: Discard non-terminal, non-reward *next*-states with prob 1 - nonTermProb

			if(start + 1 < s_.size())
			{
				// Make sure s2 is available for transition
				valid = true;
			}
		}

		// TODO: Return state, action, reward, state2, term
		Transition ret;
		ret.s  = s_[start];
		ret.a  = a_[start];
		ret.r  = r_[start];
		ret.s2 = s_[start + 1];
		ret.t  = t_[start];

		return ret;
	}

	std::vector<Transition> sample(int batch_size)
	{
		// They batch things out into a larger buffer (1000 vs 32) presumably for efficiency.
		// Let's see if we can get away with drawing directly from the state dequeue.

		// TODO: They draw transitions_ at random from the replay memory rather than seqeuences
		// of transitions_ like I initially did. I wonder if this helps convergence and generalization.
		// Would be a good experiment.

		auto batch = std::vector<Transition>(batch_size);

		for (auto i = 0; i < batch_size; i++)
		{
			batch[i] = sample_one();
		}
		return batch;
	}

};

}

#endif // DQN_TRANSITION_TABLE_H