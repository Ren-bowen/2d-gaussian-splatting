#include "simulator.h"

int main()
{
	float initial_stretch = 1.4, h = 0.004, side_len = 1, tol = 0.01;
	std::vector<float> M = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	int n_seg = 10;
	std::vector<float> k = {1000, 1000, 1000, 1000, 1000, 1000};
	std::vector<float> initial_length = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	std::vector<float> covariance = {1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 
									1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  -1.0, 0.0, 0.0, 1.0, 
									1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 1.0,
									1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, -1.0, 0.0, 1.0,
									1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 1.0, 1.0,
									1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, -1.0, 1.0};

	std::vector<int> elements = {0, 1, 0, 2, 0, 3, 0, 4, 0, 5};
	// printf("Running mass-spring simulator with parameters: rho = %f, k = %f, initial_stretch = %f, n_seg = %d, h = %f, side_len = %f, tol = %f\n", rho, k, initial_stretch, n_seg, h, side_len, tol);
	MassSpringSimulator<float, 2> simulator(M, side_len, initial_stretch, k, h, tol, n_seg, initial_length, covariance, elements);
	simulator.run();
}