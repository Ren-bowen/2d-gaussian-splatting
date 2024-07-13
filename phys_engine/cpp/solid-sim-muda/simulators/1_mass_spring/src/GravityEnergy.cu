#include "GravityEnergy.h"
#include "uti.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "device_uti.h"
#define GRAVITY -9.81
using namespace muda;

template <typename T, int dim>
struct GravityEnergy<T, dim>::Impl
{
	DeviceBuffer<T> device_x, device_grad, device_m;
	int N;
	T val;
	Impl(int N, std::vector<T> m);
};
template <typename T, int dim>
GravityEnergy<T, dim>::GravityEnergy() = default;

template <typename T, int dim>
GravityEnergy<T, dim>::~GravityEnergy() = default;

template <typename T, int dim>
GravityEnergy<T, dim>::GravityEnergy(GravityEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
GravityEnergy<T, dim> &GravityEnergy<T, dim>::operator=(GravityEnergy<T, dim> &&rhs) = default;

template <typename T, int dim>
GravityEnergy<T, dim>::GravityEnergy(const GravityEnergy<T, dim> &rhs)
	: pimpl_{std::make_unique<Impl>(*rhs.pimpl_)} {}

template <typename T, int dim>
GravityEnergy<T, dim>::GravityEnergy(int N, std::vector<T> m) : pimpl_{std::make_unique<Impl>(N, m)}
{
}

template <typename T, int dim>
GravityEnergy<T, dim>::Impl::Impl(int N_, std::vector<T> m) : N(N_)
{
	device_m.resize(N);
	device_m = m;
	device_x.resize(N * dim);
	device_grad.resize(N * dim);
}

template <typename T, int dim>
void GravityEnergy<T, dim>::update_x(const DeviceBuffer<T> &x)
{
	pimpl_->device_x.view().copy_from(x);
}

template <typename T, int dim>
void GravityEnergy<T, dim>::update_m(const DeviceBuffer<T> device_m)
{
	pimpl_->device_m = device_m;
}

template <typename T, int dim>
T GravityEnergy<T, dim>::val()
{
	auto &device_x = pimpl_->device_x;
	auto &device_m = pimpl_->device_m;
	auto N = pimpl_->N * dim;
	DeviceBuffer<T> device_val(N);
	ParallelFor(256)
		.apply(N,
			   [device_val = device_val.viewer(), device_x = device_x.cviewer(), device_m = device_m.cviewer()] __device__(int i) mutable
			   {
				   device_val(i) = i % dim == 1 ? -device_m(int(i / dim)) * GRAVITY * device_x(i) : 0;
			   })
		.wait();
	return devicesum(device_val);
}

template <typename T, int dim>
const DeviceBuffer<T> &GravityEnergy<T, dim>::grad()
{
	auto &device_x = pimpl_->device_x;
	auto &device_m = pimpl_->device_m;
	auto N = pimpl_->N * dim;
	auto &device_grad = pimpl_->device_grad;
	ParallelFor(256)
		.apply(N,
			   [device_x = device_x.cviewer(), device_m = device_m.cviewer(), N, device_grad = device_grad.viewer()] __device__(int i) mutable
			   {
				   device_grad(i) = i % dim == 1 ? -device_m(int(i / dim)) * GRAVITY : 0;
			   })
		.wait();
	// display_vec(device_grad);
	return device_grad;
} // Calculate the gradient of the energy

template class GravityEnergy<float, 2>;
template class GravityEnergy<float, 3>;
template class GravityEnergy<double, 2>;
template class GravityEnergy<double, 3>;
