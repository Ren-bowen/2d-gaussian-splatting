#include "device_uti.h"
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

template <typename T, int Size>
Eigen::Matrix<T, dim, dim> __device__ rotation_decomposation(const Eigen::Matrix<T, dim, dim> &A) {
    Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T, dim, dim> U = svd.matrixU();
    Eigen::Matrix<T, dim, dim> V = svd.matrixV();

    Eigen::Matrix<T, dim, dim> R = U * V.transpose();
    return R;
}
template Eigen::Matrix<float, 2, 2> __device__ rotation_decomposation<float, 2>(const Eigen::Matrix<float, 2, 2> &A);
template Eigen::Matrix<double, 2, 2> __device__ rotation_decomposation<double, 2>(const Eigen::Matrix<double, 2, 2> &A);
template Eigen::Matrix<float, 3, 3> __device__ rotation_decomposation<float, 3>(const Eigen::Matrix<float, 3, 3> &A);
template Eigen::Matrix<double, 3, 3> __device__ rotation_decomposation<double, 3>(const Eigen::Matrix<double, 3, 3> &A);