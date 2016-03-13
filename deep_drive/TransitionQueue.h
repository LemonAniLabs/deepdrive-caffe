#pragma once
#ifndef TRANSITION_TABLE_H
#define TRANSITION_TABLE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <array>

namespace deep_drive
{

struct Transition
{
	cv::Mat* image;
	int action;
};

class TransitionQueue
{		
	int state_dim_;
	int replay_memory_;
	std::deque<cv::Mat*> images_;
	std::deque<int>      actions_;

	public:
	TransitionQueue(int state_dim, int replay_memory)
	{
		state_dim_ = state_dim;
		replay_memory_ = replay_memory;

		// Image queue
		images_ = std::deque<cv::Mat*>(); // No size param, let it grow lazily.
	}

	~TransitionQueue() {}

	void add(cv::Mat* m, int action) 
	{
		images_.push_back(m);
		actions_.push_back(action);
	}

	int size() const
	{
		return images_.size();
	}

	void release()
	{
		release_front(images_);
		actions_.pop_front();
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
			start = 1 + (rand() % images_.size() - 1); // start at 1 because of previous action

			// Terminal states may not matter for us, but orig DQN prompted the following during original port:
			// TODO: Make sure we are not at a terminal state
			// TODO: Discard non-terminal *next*-states (we know first state of transition is non-terminal)
			//       with prob nonTermProb.
			// TODO: Discard non-terminal, non-reward *next*-states with prob 1 - nonTermProb

			if(start + 1 < images_.size())
			{
				// Make sure s2 is available for transition
				valid = true;
			}
		}

		Transition ret;
		ret.image  = images_[start];
		ret.action = actions_[start - 1];

		return ret;
	}

	std::vector<Transition> sample(int batch_size)
	{
		// DQN batches things out into a larger buffer (1000 vs 32) presumably for efficiency.
		// Let's see if we can get away with drawing directly from the state dequeue.
		auto batch = std::vector<Transition>(batch_size);

		for (auto i = 0; i < batch_size; i++)
		{
			batch[i] = sample_one();
		}
		return batch;
	}

};

}

#endif // TRANSITION_TABLE_H