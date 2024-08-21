#include "MassSpringEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include <stdio.h>
#include "device_uti.h"
#include "uti.h"
using namespace muda;

template <typename T>
T devicesum(const DeviceBuffer<T> &buffer)
{
    T sum = 0.0f;                  // Result of the reduction
    T *d_out;                      // Device memory to store the result of the reduction
    cudaMalloc(&d_out, sizeof(T)); // Allocate memory for the result

    // DeviceReduce is assumed to be part of the 'muda' library or similar
    DeviceReduce().Sum(buffer.data(), d_out, buffer.size());

    // Copy the result back to the host
    cudaMemcpy(&sum, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_out);
    return sum;
}
template float devicesum<float>(const DeviceBuffer<float> &);
template double devicesum<double>(const DeviceBuffer<double> &);

template <typename T, int Size>
void __device__ make_PSD(const Eigen::Matrix<T, Size, Size> &hess, Eigen::Matrix<T, Size, Size> &PSD)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Size, Size>> eigensolver(hess);
    Eigen::Matrix<T, Size, 1> lam = eigensolver.eigenvalues();
    Eigen::Matrix<T, Size, Size> V = eigensolver.eigenvectors();
    // set all negative eigenvalues to zero
    Eigen::Matrix<T, Size, Size> lamDiag;
    lamDiag.setZero();
    for (int i = 0; i < Size; i++)
        lamDiag(i, i) = lam(i);


    Eigen::Matrix<T, Size, Size> VT = V.transpose();

    PSD = V * lamDiag * VT;
}

template void __device__ make_PSD<float, 4>(const Eigen::Matrix<float, 4, 4> &hess, Eigen::Matrix<float, 4, 4> &PSD);
template void __device__ make_PSD<double, 4>(const Eigen::Matrix<double, 4, 4> &hess, Eigen::Matrix<double, 4, 4> &PSD);
template void __device__ make_PSD<float, 6>(const Eigen::Matrix<float, 6, 6> &hess, Eigen::Matrix<float, 6, 6> &PSD);
template void __device__ make_PSD<double, 6>(const Eigen::Matrix<double, 6, 6> &hess, Eigen::Matrix<double, 6, 6> &PSD);

template <typename T, int dim>
Eigen::Matrix<T, dim, dim> __device__ sqrt_matrix(const Eigen::Matrix<T, dim, dim> &A) {
    Eigen::Matrix<T, dim, dim> identity = Eigen::Matrix<T, dim, dim>::Identity();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigensolver(A);
    // printf("identity: %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", identity(0, 0), identity(0, 1), identity(0, 2), identity(1, 0), identity(1, 1), identity(1, 2), identity(2, 0), identity(2, 1), identity(2, 2));
    Eigen::Matrix<T, dim, 1> lam = eigensolver.eigenvalues();
    // printf("lam: %.10f %.10f %.10f\n", lam(0), lam(1), lam(2));
    Eigen::Matrix<T, dim, dim> V = eigensolver.eigenvectors();
    
    Eigen::Matrix<T, dim, dim> lamDiag;
    lamDiag.setZero();
    
    for (int i = 0; i < dim; i++) {
        if (lam(i) < 0) {
            lamDiag(i, i) = 0;
        } else {
        lamDiag(i, i) = sqrt(lam(i));
        }
        // lamDiag(i, i) = (lam(i) < 0) ? 0 : sqrt(lam(i));
    }
    
    Eigen::Matrix<T, dim, dim> VT = V.transpose();
    
    return V * lamDiag * VT;
}
template Eigen::Matrix<float, 2, 2> __device__ sqrt_matrix<float, 2>(const Eigen::Matrix<float, 2, 2> &A);
template Eigen::Matrix<double, 2, 2> __device__ sqrt_matrix<double, 2>(const Eigen::Matrix<double, 2, 2> &A);
template Eigen::Matrix<float, 3, 3> __device__ sqrt_matrix<float, 3>(const Eigen::Matrix<float, 3, 3> &A);
template Eigen::Matrix<double, 3, 3> __device__ sqrt_matrix<double, 3>(const Eigen::Matrix<double, 3, 3> &A);

template <typename T, int dim>
DeviceBuffer<T> shape_matching_kernal(const DeviceBuffer<T> &x, const DeviceBuffer<T> &x0, const DeviceBuffer<int> &elements, const DeviceBuffer<T> &covariance)
{
    int n = x.size() / dim;
    int num_elements = elements.size() / 2;
    std::cout << "num_elements: " << num_elements << std::endl;
    DeviceBuffer<T> covariance_device = covariance;
    ParallelFor(256)
        .apply(256,
                [x = x.cviewer(), x0 = x0.cviewer(), elements = elements.cviewer(), covariance_device = covariance_device.viewer(), num_elements] __device__(int i) mutable
                {
                    Eigen::Matrix<T, dim, dim> identity = Eigen::Matrix<T, dim, dim>::Identity();
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigensolver(identity * 2);
                    Eigen::Matrix<T, dim, 1> lam_ = eigensolver.eigenvalues();
                    Eigen::Matrix<T, dim, dim> V = eigensolver.eigenvectors();


                    printf("lam: %.10f %.10f %.10f\n", lam_(0), lam_(1), lam_(2));
                    printf("V: %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n", V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));
                    T connected_x[100][3];
                    T connected_x0[100][3];
                    int count = 0;
                    for (int j = 0; j < num_elements; j++)
                    {
                        if (elements(2 * j) == i)
                        {
                            for (int k = 0; k < dim; k++)
                            {
                                connected_x[count][k] = x(elements(2 * j + 1) * dim + k);
                                connected_x0[count][k] = x0(elements(2 * j + 1) * dim + k);
                            }
                            count++;
                        }
                        else if (elements(2 * j + 1) == i)
                        {
                            for (int k = 0; k < dim; k++)
                            {
                                connected_x[count][k] = x(elements(2 * j) * dim + k);
                                connected_x0[count][k] = x0(elements(2 * j) * dim + k);
                            }
                            count++;
                        }
                    }
                    if (count < dim)
                    {
                        for (int k = 0; k < dim; k++)
                        {
                            for (int d = 0; d < dim; d++)
                            {
                                // return rotation matrix as identity matrix
                                if (k == d)
                                {
                                    covariance_device(i * (dim + 1) * (dim + 1) + k * (dim + 1) + d) = 1;
                                }
                                else
                                {
                                    covariance_device(i * (dim + 1) * (dim + 1) + k * (dim + 1) + d) = 0;
                                }
                            }
                        }
                        return;
                    }
                    Eigen::Matrix<T, dim, 1> x_cm = Eigen::Matrix<T, dim, 1>::Zero();
                    Eigen::Matrix<T, dim, 1> x0_cm = Eigen::Matrix<T, dim, 1>::Zero();
                    for (int k = 0; k < count; k++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            x_cm(d) += connected_x[k][d];
                            x0_cm(d) += connected_x0[k][d];
                        }
                    }
                    x_cm /= count;
                    x0_cm /= count;
                    Eigen::Matrix<T, Eigen::Dynamic, dim> P(count, dim);
                    Eigen::Matrix<T, Eigen::Dynamic, dim> Q(count, dim);
                    for (int k = 0; k < count; k++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            P(k, d) = connected_x[k][d] - x_cm(d);
                            Q(k, d) = connected_x0[k][d] - x0_cm(d);
                        }
                    }
                    Eigen::Matrix<T, dim, dim> A_pq = P.transpose() * Q;
                    // Eigen::Matrix<T, dim, dim> A_qq = (Q.transpose() * Q).inverse();
                    Eigen::Matrix<T, dim, dim> A = A_pq.transpose() * A_pq;
                    // printf("A: %f %f %f %f %f %f %f %f %f\n", A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
                    Eigen::Matrix<T, dim, dim> S = sqrt_matrix(A);
                    // printf("A: %f %f %f %f %f %f %f %f %f\n", A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
                    // printf("S: %f %f %f %f %f %f %f %f %f\n", S(0, 0), S(0, 1), S(0, 2), S(1, 0), S(1, 1), S(1, 2), S(2, 0), S(2, 1), S(2, 2));
                    Eigen::Matrix<T, dim, dim> R = A_pq * S;
                    for (int k = 0; k < dim; k++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            covariance_device(i * (dim + 1) * (dim + 1) + k * (dim + 1) + d) = R(k, d);
                        }
                    }
                })
        .wait();
    return covariance_device;
}
template DeviceBuffer<float> shape_matching_kernal<float, 2>(const DeviceBuffer<float> &x, const DeviceBuffer<float> &x0, const DeviceBuffer<int> &elements, const DeviceBuffer<float> &covariance);
template DeviceBuffer<double> shape_matching_kernal<double, 2>(const DeviceBuffer<double> &x, const DeviceBuffer<double> &x0, const DeviceBuffer<int> &elements, const DeviceBuffer<double> &covariance);
template DeviceBuffer<float> shape_matching_kernal<float, 3>(const DeviceBuffer<float> &x, const DeviceBuffer<float> &x0, const DeviceBuffer<int> &elements, const DeviceBuffer<float> &covariance);
template DeviceBuffer<double> shape_matching_kernal<double, 3>(const DeviceBuffer<double> &x, const DeviceBuffer<double> &x0, const DeviceBuffer<int> &elements, const DeviceBuffer<double> &covariance);
