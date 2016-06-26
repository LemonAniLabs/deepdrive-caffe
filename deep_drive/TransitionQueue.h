#pragma once
#ifndef TRANSITION_TABLE_H
#define TRANSITION_TABLE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <array>
#include "deep_drive.h"

namespace deep_drive
{

struct Transition
{
	cv::Mat* image;
//	int action;
	double spin;
	double speed; // Norm of 3D speed
	double speed_change;
	double steer;
	double throttle;
};

class TransitionQueue
{		
	std::deque<cv::Mat*> images_;
	std::deque<double>   spins_;
	std::deque<double>   speeds_;
	std::deque<double>   speed_changes_;
	std::deque<double>   steers_;
	std::deque<double>   throttles_;

	public:
	TransitionQueue()
	{
		// Image queue
		images_ = std::deque<cv::Mat*>(); // No size param, let it grow lazily.
	}

	~TransitionQueue() {}

	void add(cv::Mat* m, double spin, double speed, double speed_change, double steer, double throttle) 
	{
		images_.push_back(m);
		spins_.push_back(spin);
		speeds_.push_back(speed);
		speed_changes_.push_back(speed_change);
		steers_.push_back(steer);
		throttles_.push_back(throttle);
	}

	int size() const
	{
		return images_.size();
	}

	void release()
	{
		release_front(images_);
		spins_.pop_front();
		speeds_.pop_front();
		speed_changes_.pop_front();
		steers_.pop_front();
		throttles_.pop_front();
	}

	Transition previous()
	{
		Transition ret;
		if(this->size() == 0)
		{
			ret.image = nullptr;
			return ret;
		}

		int i = this->size() - 1;
		ret.image        = images_       [i];
		ret.spin         = spins_        [i];
		ret.speed        = speeds_       [i];
		ret.speed_change = speed_changes_[i];
		ret.steer        = steers_       [i];
		ret.throttle     = throttles_    [i];
		return ret;
	}

	template <typename Dtype>
	void release_front(std::deque<Dtype>& c)
	{
		delete c[0];
		c.pop_front();
	}

	Transition sample_one()
	{
//		auto valid = true;
		auto start = get_random_int(0, images_.size() - 1);
		// Get some random starting point and make sure the size left is okay.
//		auto start = 0;
//		while (!valid)
//		{
			 // start at 1 because of previous action

			// Terminal states may not matter for us, but orig DQN prompted the following during original port:
			// TODO: Make sure we are not at a terminal state
			// TODO: Discard non-terminal *next*-states (we know first state of transition is non-terminal)
			//       with prob nonTermProb.
			// TODO: Discard non-terminal, non-reward *next*-states with prob 1 - nonTermProb

//			if(start + 1 < images_.size())
//			{
//				// Make sure s2 is available for transition
//				valid = true;
//			}
//		}

		Transition ret;
		ret.image        = images_       [start];
		ret.spin         = spins_        [start];
		ret.speed        = speeds_       [start];
		ret.speed_change = speed_changes_[start];
		ret.steer        = steers_       [start];
		ret.throttle     = throttles_    [start];

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