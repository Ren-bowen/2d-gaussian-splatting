#include "uti.h"

using namespace muda;

template <typename T>
DeviceBuffer<T> add_vector(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b, const T &factor1, const T &factor2)
{

    int N = a.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), a_device = a.cviewer(), b_device = b.cviewer(), factor1, factor2] __device__(int i) mutable
               {
                   c_device(i) = a_device(i) * factor1 + b_device(i) * factor2;
               })
        .wait();
    return c_device;
}
template DeviceBuffer<float> add_vector<float>(const DeviceBuffer<float> &a, const DeviceBuffer<float> &b, const float &factor1, const float &factor2);
template DeviceBuffer<double> add_vector<double>(const DeviceBuffer<double> &a, const DeviceBuffer<double> &b, const double &factor1, const double &factor2);

template <typename T>
DeviceBuffer<T> update_covariance_x(const DeviceBuffer<T> &x, const DeviceBuffer<T> &covariance)
{
    int n = x.size() / 3;
    DeviceBuffer<T> new_covariance_device(n * 16);
    std::cout<<"update_covariance_x" << std::endl;
    ParallelFor(256)
        .apply(n,
               [new_covariance_device = new_covariance_device.viewer(), x_device = x.cviewer(), covariance_device = covariance.cviewer()] __device__(int i) mutable
               {
                    new_covariance_device(i * 16) = covariance_device(i * 16);
                    new_covariance_device(i * 16 + 1) = covariance_device(i * 16 + 1);
                    new_covariance_device(i * 16 + 2) = covariance_device(i * 16 + 2);
                    new_covariance_device(i * 16 + 3) = covariance_device(i * 16 + 3);
                    new_covariance_device(i * 16 + 4) = covariance_device(i * 16 + 4);
                    new_covariance_device(i * 16 + 5) = covariance_device(i * 16 + 5);
                    new_covariance_device(i * 16 + 6) = covariance_device(i * 16 + 6);
                    new_covariance_device(i * 16 + 7) = covariance_device(i * 16 + 7);
                    new_covariance_device(i * 16 + 8) = covariance_device(i * 16 + 8);
                    new_covariance_device(i * 16 + 9) = covariance_device(i * 16 + 9);
                    new_covariance_device(i * 16 + 10) = covariance_device(i * 16 + 10);
                    new_covariance_device(i * 16 + 11) = covariance_device(i * 16 + 11);
                    new_covariance_device(i * 16 + 12) = covariance_device(i * 16 + 12) + x_device(i * 3);
                    new_covariance_device(i * 16 + 13) = covariance_device(i * 16 + 13) + x_device(i * 3 + 1);
                    new_covariance_device(i * 16 + 14) = covariance_device(i * 16 + 14) + x_device(i * 3 + 2);
                    new_covariance_device(i * 16 + 15) = covariance_device(i * 16 + 15);
               })
        .wait();
    std::cout<<"update_covariance_x end" << std::endl;
    return new_covariance_device;
}
template DeviceBuffer<float> update_covariance_x<float>(const DeviceBuffer<float> &x, const DeviceBuffer<float> &covariance);
template DeviceBuffer<double> update_covariance_x<double>(const DeviceBuffer<double> &x, const DeviceBuffer<double> &covariance);

template <typename T>
DeviceBuffer<T> mult_vector(const DeviceBuffer<T> &a, const T &b)
{
    int N = a.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), a_device = a.cviewer(), b] __device__(int i) mutable
               {
                   c_device(i) = a_device(i) * b;
               })
        .wait();
    return c_device;
}
template DeviceBuffer<float> mult_vector<float>(const DeviceBuffer<float> &a, const float &b);
template DeviceBuffer<double> mult_vector<double>(const DeviceBuffer<double> &a, const double &b);

template <typename T>
DeviceBuffer<T> multi_vector(const DeviceBuffer<T> &a, const DeviceBuffer<T> &b)
{
    int N = a.size();
    DeviceBuffer<T> c_device(N);
    ParallelFor(256)
        .apply(N,
               [c_device = c_device.viewer(), a_device = a.cviewer(), b_device = b.cviewer()] __device__(int i) mutable
               {
                   c_device(i) = a_device(i) * b_device(i);
               })
        .wait();
    return c_device;
}

template DeviceBuffer<float> multi_vector<float>(const DeviceBuffer<float> &a, const DeviceBuffer<float> &b);
template DeviceBuffer<double> multi_vector<double>(const DeviceBuffer<double> &a, const DeviceBuffer<double> &b);


template <typename T>
DeviceTripletMatrix<T, 1> add_triplet(const DeviceTripletMatrix<T, 1> &a, const DeviceTripletMatrix<T, 1> &b, const T &factor1, const T &factor2)
{
    int Na = a.triplet_count();
    int Nb = b.triplet_count();
    DeviceTripletMatrix<T, 1> c;
    c.resize_triplets(Na + Nb);
    c.reshape(a.rows(), a.cols());
    ParallelFor(256)
        .apply(Na,
               [c_device_values = c.values().viewer(), c_device_row_indices = c.row_indices().viewer(), c_device_col_indices = c.col_indices().viewer(),
                a_device_values = a.values().cviewer(), a_device_row_indices = a.row_indices().cviewer(), a_device_col_indices = a.col_indices().cviewer(), factor1] __device__(int i) mutable
               {
                   c_device_row_indices(i) = a_device_row_indices(i);
                   c_device_col_indices(i) = a_device_col_indices(i);
                   c_device_values(i) = a_device_values(i) * factor1;
               })
        .wait();
    ParallelFor(256)
        .apply(Nb,
               [c_device_values = c.values().viewer(), c_device_row_indices = c.row_indices().viewer(), c_device_col_indices = c.col_indices().viewer(),
                b_device_values = b.values().cviewer(), b_device_row_indices = b.row_indices().cviewer(), b_device_col_indices = b.col_indices().cviewer(), factor2, Na] __device__(int i) mutable
               {
                   c_device_row_indices(i + Na) = b_device_row_indices(i);
                   c_device_col_indices(i + Na) = b_device_col_indices(i);
                   c_device_values(i + Na) = b_device_values(i) * factor2;
               })
        .wait();

    return c;
}
template DeviceTripletMatrix<float, 1> add_triplet<float>(const DeviceTripletMatrix<float, 1> &a, const DeviceTripletMatrix<float, 1> &b, const float &factor1, const float &factor2);
template DeviceTripletMatrix<double, 1> add_triplet<double>(const DeviceTripletMatrix<double, 1> &a, const DeviceTripletMatrix<double, 1> &b, const double &factor1, const double &factor2);
template <typename T>
T max_vector(const DeviceBuffer<T> &a)
{
    DeviceBuffer<T> buffer(a);
    T vec_max = 0.0f;              // Result of the reduction
    T *d_out;                      // Device memory to store the result of the reduction
    cudaMalloc(&d_out, sizeof(T)); // Allocate memory for the result
    int N = buffer.size();
    ParallelFor(256)
        .apply(N,
               [buffer = buffer.viewer()] __device__(int i) mutable
               {
                   buffer(i) = fabs(buffer(i));
               })
        .wait();
    DeviceReduce().Max(buffer.data(), d_out, buffer.size());

    // Copy the result back to the host
    cudaMemcpy(&vec_max, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_out);
    return vec_max;
}
template float max_vector<float>(const DeviceBuffer<float> &a);
template double max_vector<double>(const DeviceBuffer<double> &a);

template <typename T, int dim>
void search_dir(const DeviceBuffer<T> &grad, const DeviceTripletMatrix<T, 1> &hess, DeviceBuffer<T> &dir, const DeviceBuffer<int> &DBC)
{
    static LinearSystemContext ctx;
    auto neg_grad = mult_vector<T>(grad, -1);
    int N = grad.size();
    DeviceDenseVector<T> grad_device;
    grad_device.resize(N);
    DeviceCOOMatrix<T> A_coo;
    ctx.convert(hess, A_coo);
    DeviceCSRMatrix<T> A_csr;
    ctx.convert(A_coo, A_csr);
    set_DBC<T, dim>(neg_grad, A_csr, DBC);
    grad_device.buffer_view().copy_from(neg_grad);
    DeviceDenseVector<T> dir_device;
    dir_device.resize(N);
    ctx.solve(dir_device.view(), A_csr.cview(), grad_device.cview());
    ctx.sync();
    dir.view().copy_from(dir_device.buffer_view());
}
template void search_dir<float, 2>(const DeviceBuffer<float> &grad, const DeviceTripletMatrix<float, 1> &hess, DeviceBuffer<float> &dir, const DeviceBuffer<int> &DBC);
template void search_dir<double, 2>(const DeviceBuffer<double> &grad, const DeviceTripletMatrix<double, 1> &hess, DeviceBuffer<double> &dir, const DeviceBuffer<int> &DBC);
template void search_dir<float, 3>(const DeviceBuffer<float> &grad, const DeviceTripletMatrix<float, 1> &hess, DeviceBuffer<float> &dir, const DeviceBuffer<int> &DBC);
template void search_dir<double, 3>(const DeviceBuffer<double> &grad, const DeviceTripletMatrix<double, 1> &hess, DeviceBuffer<double> &dir, const DeviceBuffer<int> &DBC);


template <typename T>
void matrix_plus_vector(const DeviceTripletMatrix<T, 1> &A, const DeviceBuffer<T> &x, DeviceBuffer<T> &b)
{
    static LinearSystemContext ctx;
    int N = x.size();
    DeviceDenseVector<T> x_device;
    x_device.resize(N);
    x_device.buffer_view().copy_from(x);
    DeviceCOOMatrix<T> A_coo;
    ctx.convert(A, A_coo);
    DeviceCSRMatrix<T> A_csr;
    ctx.convert(A_coo, A_csr);
    DeviceDenseVector<T> b_device;
    b_device.resize(N);
    ctx.spmv(A_csr.cview(), x_device.cview(), b_device.view());
    ctx.sync();
    b.view().copy_from(b_device.buffer_view());
}
template void matrix_plus_vector<float>(const DeviceTripletMatrix<float, 1> &A, const DeviceBuffer<float> &x, DeviceBuffer<float> &b);
template void matrix_plus_vector<double>(const DeviceTripletMatrix<double, 1> &A, const DeviceBuffer<double> &x, DeviceBuffer<double> &b);

template <typename T>
void display_vec(const DeviceBuffer<T> &vec)
{
    int N = vec.size();
    ParallelFor(256)
        .apply(N,
               [vec = vec.cviewer()] __device__(int i) mutable
               {
                   printf("%d %f\n", i, vec(i));
               })
        .wait();
}
template void display_vec<float>(const DeviceBuffer<float> &vec);
template void display_vec<double>(const DeviceBuffer<double> &vec);

template <typename T>
void test(const DeviceBuffer<T> &a, DeviceBuffer<T> &b)
{
    int N = a.size();
    ParallelFor(256)
        .apply(N,
               [a = a.cviewer(), b = b.viewer()] __device__(int i) mutable
               {
                   b(i) = a(i);
               })
        .wait();
}
template void test<float>(const DeviceBuffer<float> &a, DeviceBuffer<float> &b);
template void test<double>(const DeviceBuffer<double> &a, DeviceBuffer<double> &b);

template <typename T, int dim>
void set_DBC(DeviceBuffer<T> &grad, DeviceCSRMatrix<T> &hess, const DeviceBuffer<int> &DBC)
{
    int N = hess.non_zeros();
    int Nr = hess.rows();
    ParallelFor(256)
        .apply(N,
               [hess_row_offsets = hess.row_offsets().cviewer(), hess_col_indices = hess.col_indices().cviewer(), hess_values = hess.values().viewer(), DBC = DBC.cviewer(), Nr] __device__(int i) mutable
               {
                   // search for the row index
                   int right = Nr;
                   int left = -1;

                   while (left < right)
                   {
                       int mid = (left + right) / 2;
                       if (hess_row_offsets(mid) <= i)
                       {
                           left = mid + 1;
                       }
                       else
                       {
                           right = mid;
                       }
                   }
                   int row = left - 1;
                   int col = hess_col_indices(i);
                   if (DBC(int(row / dim)) || DBC(int(col / dim)))
                   {
                       hess_values(i) = row == col ? 1 : 0;
                   }
               })
        .wait();
    int NDBC = DBC.size();
    ParallelFor(256)
        .apply(NDBC,
               [grad = grad.viewer(), DBC = DBC.cviewer()] __device__(int i) mutable
               {
                   if (DBC(i) == 1)
                   {
                       for (int d = 0; d < dim; d++)
                       {
                           grad(i * dim + d) = 0;
                       }
                   }
               })
        .wait();
}

template void set_DBC<float, 2>(DeviceBuffer<float> &grad, DeviceCSRMatrix<float> &hess, const DeviceBuffer<int> &DBC);
template void set_DBC<float, 3>(DeviceBuffer<float> &grad, DeviceCSRMatrix<float> &hess, const DeviceBuffer<int> &DBC);
template void set_DBC<double, 2>(DeviceBuffer<double> &grad, DeviceCSRMatrix<double> &hess, const DeviceBuffer<int> &DBC);
template void set_DBC<double, 3>(DeviceBuffer<double> &grad, DeviceCSRMatrix<double> &hess, const DeviceBuffer<int> &DBC);