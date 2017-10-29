/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.0001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


	num_particles = 100;

	particles = std::vector<Particle>(num_particles);

	std::default_random_engine gen;
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> theta_dist(theta, std[2]);

	int id = 0;
	for(auto& particle : particles){
		particle.id = id;
		particle.x = x_dist(gen);
		particle.y = y_dist(gen);
		particle.theta = theta_dist(gen);
		particle.weight = 1.0;
		id++;
	}

	weights = std::vector<double>(num_particles);
	for(int i = 0; i < num_particles; i++){
		weights[i] = 1.0;
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	std::default_random_engine gen;

	double dist =  delta_t * velocity;
	double delta_theta = yaw_rate * delta_t;
	double vy = velocity / yaw_rate;


	for(auto& particle : particles){
		if (fabs(yaw_rate) < EPS){
			particle.x += dist * cos(particle.theta);
			particle.y += dist * sin(particle.theta);
		} else {
			double theta = delta_theta  + particle.theta;
			particle.x += vy * (sin(theta) - sin(particle.theta));
			particle.y += vy * (cos(particle.theta) - cos(theta));
			particle.theta += delta_theta;
		}

		// add noise
		particle.x = std::normal_distribution<double>(particle.x, std_pos[0])(gen);
		particle.y = std::normal_distribution<double>(particle.y, std_pos[1])(gen);
		particle.theta = std::normal_distribution<double>(particle.theta, std_pos[2])(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	double std_x2 = std_landmark[0] * std_landmark[0];
	double std_y2 = std_landmark[1] * std_landmark[1];
	double coeff = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);

	for(auto& particle : particles){
		double x = particle.x;
		double y = particle.y;
		double theta = particle.theta;

		//  landmarks within sensor range
		std::vector<Map::single_landmark_s> landmarks;
		for(auto& landmark : map_landmarks.landmark_list){
			if (landmark.x_f > x - sensor_range && landmark.x_f < x + sensor_range && landmark.y_f > y - sensor_range && landmark.y_f < y + sensor_range){
				landmarks.push_back(landmark);
			}
		}

		// transform observations to coordinates of particle
		std::vector<LandmarkObs> tobservations(observations.size());
		for(int i = 0; i < observations.size(); i++){
			tobservations[i].x = x + observations[i].x * cos(theta) - observations[i].y * sin(theta);
			tobservations[i].y = y + observations[i].x * sin(theta) + observations[i].y * cos(theta);
		}

		//find  the closest landmark for each transformed observation
		vector<int> associations(tobservations.size());
		vector<double> sense_x(tobservations.size());
		vector<double> sense_y(tobservations.size());

		for(int i = 0; i < tobservations.size(); i++){
			double x = tobservations[i].x;
			double y = tobservations[i].y;
			double min_distance = std::numeric_limits<double>::max();
			int best_landmark_id = 0;
			double best_x = 0.0;
			double best_y = 0.0;

			for(auto& landmark : landmarks){
				double distance = dist(x,y,landmark.x_f, landmark.y_f);
				if (distance < min_distance){
					min_distance = distance;
					best_landmark_id = landmark.id_i;
					best_x = landmark.x_f;
					best_y = landmark.y_f;
				}
			}
			associations[i] = best_landmark_id;
			sense_x[i] = best_x;
			sense_y[i] = best_y;
		}

		particle = SetAssociations(particle, associations, sense_x, sense_y);

		double weight = 1.0;
		for(int i = 0; i < associations.size(); i++){
			double x = tobservations[i].x;
			double y = tobservations[i].y;
			double e = pow(x - sense_x[i], 2.0) / (2 * std_x2) + pow(y - sense_y[i], 2.0) / (2 * std_y2);
			double res = exp(-e);
			weight *= res;
		}
		particle.weight = weight;
		weights[particle.id] = weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<> dist(weights.begin(), weights.end());
	std::default_random_engine gen;

	std::vector<Particle> sampled_particles(particles.size());
	for(int i = 0; i < particles.size(); i++){
		int index = dist(gen);
		sampled_particles[i] = particles[index];
		sampled_particles[i].id = i;
	}
	particles = sampled_particles;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
