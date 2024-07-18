#include "simulator.h"
// #include <SFML/Graphics.hpp>
#include "InertialEnergy.h"
#include "MassSpringEnergy.h"
#include "GravityEnergy.h"
#include <muda/muda.h>
#include <muda/container.h>
#include "uti.h"
#include "device_uti.h"
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace muda;



template <typename T, int dim>
struct MassSpringSimulator<T, dim>::Impl
{
    int n_seg;
    T h, side_len, initial_stretch, tol;
    std::vector<T> m;
    std::vector<T> initial_length;
    std::vector<T> covariance_;
    std::vector<int> elements;
    DeviceBuffer<int> device_DBC;
    int resolution = 900, scale = 200, offset = resolution / 2, radius = 5;
    std::vector<T> x, x_tilde, v, k, l2;
    std::vector<int> e;
    // sf::RenderWindow window;
    InertialEnergy<T, dim> inertialenergy;
    MassSpringEnergy<T, dim> massspringenergy;
    GravityEnergy<T, dim> gravityenergy;
    Impl(std::vector<T> M, T side_len, T initial_stretch, std::vector<T> K, T h_, T tol_, int n_seg, std::vector<T> initial_length, std::vector<T> covariance, std::vector<int> elements);
    void update_x(const DeviceBuffer<T> &new_x);
    void update_x_tilde(const DeviceBuffer<T> &new_x_tilde);
    void update_v(const DeviceBuffer<T> &new_v);
    void update_covariance(const DeviceBuffer<T> &new_covariance);
    void shape_matching(const std::vector<T> &x0);
    T IP_val();
    void step_forward();
    //void draw();
    DeviceBuffer<T> IP_grad();
    DeviceTripletMatrix<T, 1> IP_hess();
    DeviceBuffer<T> search_direction();
    //T screen_projection_x(T point);
    //T screen_projection_y(T point);
};
template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::~MassSpringSimulator() = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator(MassSpringSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringSimulator<T, dim> &MassSpringSimulator<T, dim>::operator=(MassSpringSimulator<T, dim> &&rhs) = default;

template <typename T, int dim>
MassSpringSimulator<T, dim>::MassSpringSimulator(std::vector<T> M, T side_len, T initial_stretch, std::vector<T> K, T h_, T tol_, int n_seg, std::vector<T> initial_length, std::vector<T> covariance, std::vector<int> elements) : pimpl_{std::make_unique<Impl>(M, side_len, initial_stretch, K, h_, tol_, n_seg, initial_length, covariance, elements)}
{
}
template <typename T, int dim>
MassSpringSimulator<T, dim>::Impl::Impl(std::vector<T> M, T side_len, T initial_stretch, std::vector<T> K, T h_, T tol_, int n_seg, std::vector<T> initial_length, std::vector<T> covariance, std::vector<int> elements) : tol(tol_), h(h_)
{
    int N = covariance.size() / ((dim + 1) * (dim + 1));
    covariance_.resize(N * (dim + 1) * (dim + 1));
    for (int i = 0; i < N * (dim + 1) * (dim + 1); i++)
    {
        covariance_[i] = covariance[i];
    }
    x.resize(N * dim);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dim; j++) {
            x[i * dim + j] = covariance[i * 16 + 12 + j];
        }
    }
    std::vector<int> DBC(x.size() / dim, 0);
    std::vector<int> max_id;
    T max_y = x[1];
    for (int i = 0; i < x.size() / dim; i++)
    {
        if (x[i * dim + 1] > max_y)
        {
            max_y = x[i * dim + 1];
        }
    }
    T min_y = x[1];
    for (int i = 0; i < x.size() / dim; i++)
    {
        if (x[i * dim + 1] < min_y)
        {
            min_y = x[i * dim + 1];
        }
    }
    for (int i = 0; i < x.size() / dim; i++)
    {
        if (x[i * dim + 1] > 0.75 * max_y + 0.25 * min_y)
        {
            max_id.push_back(i);
        }
    }
    std::cout << "max_y " << max_y << std::endl;
    std::cout << "min_y " << min_y << std::endl;
    std::cout << "max_id.size() " << max_id.size() << std::endl;
    for (int i = 0; i < max_id.size(); i++)
    {
        DBC[max_id[i]] = 1;
    }
    // DBC[0] = 1;
    // DBC[9] = 1;
    v.resize(x.size(), 0);
    k.resize(K.size());
    for (int i = 0; i < K.size(); i++)
    {
        k[i] = K[i];
    }
    e.resize(elements.size());
    for (int i = 0; i < elements.size(); i++)
    {
        e[i] = elements[i];
    }
    l2.resize(initial_length.size());
    for (int i = 0; i < initial_length.size(); i++)
    {
        l2[i] = initial_length[i] * initial_length[i];
    }
    m.resize(M.size());
    for (int i = 0; i < M.size(); i++)
    {
        m[i] = M[i];
    }
    // initial stretch
    for (int i = 0; i < N; i++) {
        x[i * dim + 0] *= initial_stretch;
        // std::cout<<"x: "<<x[i * dim + 0]<<std::endl;
    }
    inertialenergy = InertialEnergy<T, dim>(N, m);
    massspringenergy = MassSpringEnergy<T, dim>(x, e, l2, k);
    gravityenergy = GravityEnergy<T, dim>(N, m);
    DeviceBuffer<T> x_device(x);
    update_x(x_device);
    device_DBC = DeviceBuffer<int>(DBC);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::run()
{
    assert(dim == 3);
    // bool running = true;
    pimpl_->step_forward();
    /*
    auto &window = pimpl_->window;
    while (running)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                running = false;
        }

        pimpl_->draw(); // Draw the current state

        // Update the simulation state
        pimpl_->step_forward();

        // Wait according to time step
        // sf::sleep(sf::milliseconds(static_cast<int>(h * 1000)));
    }

    window.close();
    */
}

template <typename T, int dim>
std::vector<T> MassSpringSimulator<T, dim>::get_x()
{
    return pimpl_->x;
}

template <typename T, int dim>
std::vector<T> MassSpringSimulator<T, dim>::get_v()
{
    return pimpl_->v;
}

template <typename T, int dim>
void MassSpringSimulator<T, dim>::set_v(const std::vector<T> v_new)
{
    pimpl_->update_v(v_new);
}

template <typename T, int dim>
std::vector<T> MassSpringSimulator<T, dim>::get_covariance()
{
    // Here covariance[:3][:3] is the rotation matrix computed by shape matching
    return pimpl_->covariance_;
}

template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::step_forward()
{
    DeviceBuffer<T> x_tilde(x.size()); // Predictive position
    // std::cout<<"h "<<h<<std::endl;
    update_x_tilde(add_vector<T>(x, v, 1, h));
    std::vector<T> x_ = x;
    std::vector<T> x_n = x; // Copy current positions to x_n
    std::cout <<"x.size() "<<x.size()<<std::endl;
    int iter = 0;
    T E_last = IP_val();
    std::cout << "Initial E_last " << E_last << "\n";
    DeviceBuffer<T> p = search_direction();
    T residual = max_vector(p) / h;
    std::cout << "Initial residual " << residual << "\n";
    std::vector<T> test;
    test.resize(x.size());
    for (int i = 0; i < x.size(); i++) {
        if (i / 2 != 0)
            test[i] = 1e-4;
        else
            test[i] = 0;
    }
    DeviceBuffer<T> test_device = test;
    update_x(add_vector<T>(x, test_device, 1.0, 1.0));
    T val1 = IP_val();
    DeviceBuffer<T> grad1 = IP_grad();
    update_x(add_vector<T>(x, test_device, 1.0, -2.0));
    T val2 = IP_val();
    DeviceBuffer<T> grad2 = IP_grad();
    update_x(add_vector<T>(x, test_device, 1.0, 1.0));
    T numerical_diff = 0.5 * (val1 - val2);
    DeviceBuffer<T> grad_numerical_diff = add_vector<T>(grad1, grad2, 0.5, -0.5);
    T analytical_diff = devicesum(multi_vector<T>(IP_grad(), test_device));
    DeviceBuffer<T> grad_analytical_diff;
    grad_analytical_diff.resize(x.size());
    matrix_plus_vector(IP_hess(), test_device, grad_analytical_diff);
    DeviceBuffer<T> grad_error_device;
    grad_error_device.resize(x.size());
    // hess = d(grad)dx, hess(x) @ test = grad(x + 0.5 * test) - grad(x - 0.5 * test)
    grad_error_device = add_vector<T>(grad_numerical_diff, grad_analytical_diff, 1.0, -1.0);
    T grad_error = sqrt(devicesum(multi_vector(grad_error_device, grad_error_device)));
    T grad_numerical_diff_norm = sqrt(devicesum(multi_vector(grad_numerical_diff, grad_numerical_diff)));
    T grad_analytical_diff_norm = sqrt(devicesum(multi_vector(grad_analytical_diff, grad_analytical_diff)));
    // T numerical_diff = IP_val() - E_last;
    // T analytical_diff = devicesum(multi_vector<T>(IP_grad(), test_device));
    // std::cout << "Numerical diff " << numerical_diff << " Analytical diff " << analytical_diff << "\n";
    std::cout << "Numerical diff " << numerical_diff << " Analytical diff " << analytical_diff << "\n";
    std::cout << "grad_error " << grad_error << "\n";
    std::cout << "grad_numerical_diff_norm " << grad_numerical_diff_norm << " grad_analytical_diff_norm " << grad_analytical_diff_norm << "\n";
    // std::cout <<"x.size() "<<x.size()<<std::endl;
    while (residual > tol)
    {
        // Line search
        T alpha = 1;
        DeviceBuffer<T> x0 = x;
        update_x(add_vector<T>(x0, p, 1.0, alpha));
        while (IP_val() > E_last)
        {
            alpha /= 2;
            update_x(add_vector<T>(x0, p, 1.0, alpha));
        }
        std::cout << "step size = " << alpha << "\n";
        E_last = IP_val();
        // std::cout << "E_last " << E_last << "\n";
        p = search_direction();
        residual = max_vector(p) / h;
        iter += 1;
        std::cout << "Iteration " << iter << " residual " << residual << "E_last" << E_last << "\n";
        if (iter > 100 or alpha < 1e-8)
        {
            std::cout << "Newton iteration failed\n" << std::endl;
            break;
        }
    }
    update_v(add_vector<T>(x, x_n, 1 / h, -1 / h));
    std::cout <<"x.size() "<<x.size()<<std::endl;
    std::cout << "Final E_last " << E_last << "\n";
    std::cout << "covariance_.size(): " << covariance_.size() << std::endl;
    /*
    for (int i = 0; i < x.size() / dim; i++) {
        for (int j = 0; j < dim; j++) {
            covariance_[i * 16 + 12 + j] = x[i * dim + j];
        }
    }
    */
    shape_matching(x_n);
}
/*
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::screen_projection_x(T point)
{
    return offset + scale * point;
}
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::screen_projection_y(T point)
{
    return resolution - (offset + scale * point);
}
*/
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_x(const DeviceBuffer<T> &new_x)
{
    inertialenergy.update_x(new_x);
    massspringenergy.update_x(new_x);
    gravityenergy.update_x(new_x);
    new_x.copy_to(x);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_x_tilde(const DeviceBuffer<T> &new_x_tilde)
{
    inertialenergy.update_x_tilde(new_x_tilde);
    new_x_tilde.copy_to(x_tilde);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_v(const DeviceBuffer<T> &new_v)
{
    new_v.copy_to(v);
}
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::update_covariance(const DeviceBuffer<T> &new_covariance)
{
    new_covariance.copy_to(covariance_);
}
/*
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::draw()
{
    window.clear(sf::Color::White); // Clear the previous frame

    // Draw springs as lines
    for (int i = 0; i < e.size() / 2; ++i)
    {
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2] * dim]), screen_projection_y(x[e[i * 2] * dim + 1])), sf::Color::Blue),
            sf::Vertex(sf::Vector2f(screen_projection_x(x[e[i * 2 + 1] * dim]), screen_projection_y(x[e[i * 2 + 1] * dim + 1])), sf::Color::Blue)};
        window.draw(line, 2, sf::Lines);
    }

    // Draw masses as circles
    for (int i = 0; i < x.size() / dim; ++i)
    {
        sf::CircleShape circle(radius); // Set a fixed radius for each mass
        circle.setFillColor(sf::Color::Red);
        circle.setPosition(screen_projection_x(x[i * dim]) - radius, screen_projection_y(x[i * dim + 1]) - radius); // Center the circle on the mass
        window.draw(circle);
    }

    window.display(); // Display the rendered frame
}
*/
template <typename T, int dim>
T MassSpringSimulator<T, dim>::Impl::IP_val()
{
    // std::cout<<"Inertial energy "<<inertialenergy.val()<<std::endl;
    // std::cout<<"Mass spring energy "<<massspringenergy.val()<<std::endl;
    // return massspringenergy.val() * h * h;
    return inertialenergy.val() + massspringenergy.val() * h * h + gravityenergy.val() * h * h;

}

template <typename T, int dim>
DeviceBuffer<T> MassSpringSimulator<T, dim>::Impl::IP_grad()
{
    return add_vector<T>(add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h), gravityenergy.grad(), 1.0, h * h);

    // return add_vector<T>(inertialenergy.grad(), massspringenergy.grad(), 1.0, h * h);
}

template <typename T, int dim>
DeviceTripletMatrix<T, 1> MassSpringSimulator<T, dim>::Impl::IP_hess()
{
    DeviceTripletMatrix<T, 1> inertial_hess = inertialenergy.hess();
    DeviceTripletMatrix<T, 1> massspring_hess = massspringenergy.hess();
    DeviceTripletMatrix<T, 1> hess = add_triplet<T>(inertial_hess, massspring_hess, 1.0, h * h);
    // DeviceTripletMatrix<T, 1> hess = inertial_hess;
    return hess;
}

template <typename T, int dim>
DeviceBuffer<T> MassSpringSimulator<T, dim>::Impl::search_direction()
{
    DeviceBuffer<T> dir;
    dir.resize(x.size());
    search_dir<T, dim>(IP_grad(), IP_hess(), dir, device_DBC);
    return dir;
}
/*
template <typename T, int dim>
__global__ void shape_matching_kernel(
    const T* x_device, const T* x0_device, const int* elements_device,
    T* covariance_device, int num_particles, int num_elements) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // const int tid = threadIdx.x;if(tid==0) printf("!!%d\n",tid);
    printf("i: %d\n", i);
    if (i >= num_particles) return;

    extern __shared__ char shared_mem_raw[];
    T* shared_mem = reinterpret_cast<T*>(shared_mem_raw);

    T* connected_x = shared_mem;
    T* connected_x0 = shared_mem + num_elements * dim;

    int count = 0;

    for (int j = 0; j < num_elements; j++) {
        if (elements_device[2 * j] == i) {
            for (int k = 0; k < dim; k++) {
                connected_x[count * dim + k] = x_device[elements_device[2 * j + 1] * dim + k];
                connected_x0[count * dim + k] = x0_device[elements_device[2 * j + 1] * dim + k];
            }
            count++;
        } else if (elements_device[2 * j + 1] == i) {
            for (int k = 0; k < dim; k++) {
                connected_x[count * dim + k] = x_device[elements_device[2 * j] * dim + k];
                connected_x0[count * dim + k] = x0_device[elements_device[2 * j] * dim + k];
            }
            count++;
        }
    }
    // print count
    printf("count: %d\n", count);
    if (count < dim) {
        for (int k = 0; k < dim; k++) {
            for (int d = 0; d < dim; d++) {
                // return rotation matrix as identity matrix
                if (k == d) {
                    covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = 1;
                } else {
                    covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = 0;
                }
            }
        }
        return;
    }

    Eigen::Matrix<T, dim, 1> x_cm = Eigen::Matrix<T, dim, 1>::Zero();
    Eigen::Matrix<T, dim, 1> x0_cm = Eigen::Matrix<T, dim, 1>::Zero();

    for (int k = 0; k < count; k++) {
        for (int d = 0; d < dim; d++) {
            x_cm(d) += connected_x[k * dim + d];
            x0_cm(d) += connected_x0[k * dim + d];
        }
    }
    x_cm /= count;
    x0_cm /= count;

    Eigen::Matrix<T, Eigen::Dynamic, dim> P(count, dim);
    Eigen::Matrix<T, Eigen::Dynamic, dim> Q(count, dim);
    for (int k = 0; k < count; k++) {
        for (int d = 0; d < dim; d++) {
            P(k, d) = connected_x[k * dim + d] - x_cm(d);
            Q(k, d) = connected_x0[k * dim + d] - x0_cm(d);
        }
    }

    Eigen::Matrix<T, dim, dim> A_pq = P.transpose() * Q;
    Eigen::Matrix<T, dim, dim> A_qq = (Q.transpose() * Q).inverse();
    Eigen::Matrix<T, dim, dim> A = A_pq.transpose() * A_pq;
    Eigen::Matrix<T, dim, dim> S = sqrt_matrix(A);
    Eigen::Matrix<T, dim, dim> R = A_pq * S.inverse();

    for (int k = 0; k < dim; k++) {
        for (int d = 0; d < dim; d++) {
            
            // covariance[:3, :3] =  R @ covariance[:3, :3] 
            covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = 0;
            for (int l = 0; l < dim; l++) {
                covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] += R(d, l) * covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + l];
            }
            
            // Covariance[:3, :3] is not the rotation matrix, but that after rescaling.
            // So we directly return the rotation matrix
            covariance_device[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = R(k, d);
        }
    }
}
*/

template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::shape_matching(const std::vector<T> &x0) {
    int num_particles = x.size() / dim;
    int num_elements = e.size() / 2;
    for (int i = 0; i < num_particles; i++) {
        T connected_x[100][3];
        T connected_x0[100][3];
        int count = 0;
        for (int j = 0; j < num_elements; j++)
        {
            if (e[2 * j] == i)
            {
                for (int k = 0; k < dim; k++)
                {
                    connected_x[count][k] = x[e[2 * j + 1] * dim + k];
                    connected_x0[count][k] = x0[e[2 * j + 1] * dim + k];
                }
                count++;
            }
            else if (e[2 * j + 1] == i)
            {
                for (int k = 0; k < dim; k++)
                {
                    connected_x[count][k] = x[e[2 * j] * dim + k];
                    connected_x0[count][k] = x0[e[2 * j] * dim + k];
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
                        covariance_[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = 1;
                    }
                    else
                    {
                        covariance_[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = 0;
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
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigensolver(A);
        Eigen::Matrix<T, dim, 1> lam = eigensolver.eigenvalues();
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
        
        Eigen::Matrix<T, dim, dim> S = V * lamDiag * VT;
        // printf("A: %f %f %f %f %f %f %f %f %f\n", A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
        Eigen::Matrix<T, dim, dim> R = A_pq * S.inverse();
        for (int k = 0; k < dim; k++)
        {
            for (int d = 0; d < dim; d++)
            {
                covariance_[i * (dim + 1) * (dim + 1) + k * (dim + 1) + d] = R(k, d);
            }
        }
    }

}
/*
template <typename T, int dim>
void MassSpringSimulator<T, dim>::Impl::shape_matching(const DeviceBuffer<T> &x0) {
    std::cout << "shape matching" << std::endl;
    // Convert input vectors to device buffers
    DeviceBuffer<T> x_device = x;
    DeviceBuffer<T> x0_device = x0;
    DeviceBuffer<int> elements_device = e;
    DeviceBuffer<T> covariance_device = covariance_;

    int num_particles = x.size() / dim;
    int num_elements = e.size() / 2;
    std::cout << "num_particles: " << num_particles << " num_elements: " << num_elements << "\n";

    int threads_per_block = 32;
    int num_blocks = (num_particles + threads_per_block - 1) / threads_per_block;
    size_t shared_memory_size = num_elements * dim * sizeof(T) * 2;
    std::cout << "shape matching" << std::endl;
    // DeviceBuffer<T> covariance_update = shape_matching_kernal<T, dim>(x_device, x0_device, elements_device, covariance_device);
    shape_matching_kernel<T, dim><<<num_blocks, threads_per_block, shared_memory_size>>>(
    x_device.data(), x0_device.data(), elements_device.data(),
    covariance_device.data(), num_particles, num_elements);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "shape matching" << std::endl;
    // Copy data back to host if necessary
    update_covariance(covariance_device);
}
*/

template class MassSpringSimulator<float, 2>;
template class MassSpringSimulator<double, 2>;
template class MassSpringSimulator<float, 3>;
template class MassSpringSimulator<double, 3>;