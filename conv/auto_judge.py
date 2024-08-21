from simulation import *

np.random.seed(42)
def check_grad(f, g, x):
    dim = len(x)
    E = f(x)
    grad = g(x)
    histo = []
    test_dir = np.random.randn(dim)
    while np.linalg.norm(test_dir) < 1e-6: test_dir = np.random.randn(dim)
    test_dir /= np.linalg.norm(test_dir)
    for eps in [10 ** (-j) for j in range(4, 13)]:
        x_plus = x.copy()
        x_plus += eps * test_dir
        try:
            E_plus = f(x_plus)
            if np.isclose((E_plus - E) / eps, grad @ test_dir, 1e-3, 1e-2).all():
                return True
            else:
                histo.append((E_plus - E) / eps)
        except RuntimeError:
            continue
        x_neg = x.copy()
        x_neg -= eps * test_dir
        try:
            E_neg = f(x_neg)
            if np.isclose((E - E_neg) / eps, grad @ test_dir, 1e-3, 1e-2).all():
                return True
            else:
                histo.append((E - E_neg) / eps)
        except RuntimeError:
            continue
    if len(histo) == 0:
        raise RuntimeError("Your simulation raises RuntimeError for every call of f(x).")
    if (np.min(histo, 0) < grad @ test_dir).all() and (grad @ test_dir < np.max(histo, 0)).all():
        # Numerical gradient approximation is ill-formed.
        return True
    return False

sim, h, frame_num, _ = create_sim((4, 4), 1.)
bending_energy_score = 2
bending_gradient_score = 3
bending_hessian_score = 3
stretching_and_shearing_gradient_score = 2
stretching_and_shearing_hessian_score = 2
bending_gt = np.load("bending_gt.npy")
bending_energy_list = []

for _ in range(frame_num):
    sim.Forward(h)
    def bending_energy(x): return np_real([sim.ComputeBendingEnergy(x.reshape((-1, 3)).T)])
    def bending_gradient(x): return -sim.ComputeBendingForce(x.reshape((-1, 3)).T).T.ravel()
    def bending_hessian(x): return sim.ComputeBendingHessian(x.reshape((-1, 3)).T).toarray()
    def stretching_and_shearing_energy(x): return np_real([sim.ComputeStretchingAndShearingEnergy(x.reshape((-1, 3)).T)])
    def stretching_and_shearing_gradient(x): return -sim.ComputeStretchingAndShearingForce(x.reshape((-1, 3)).T).T.ravel()
    def stretching_and_shearing_hessian(x): return sim.ComputeStretchingAndShearingHessian(x.reshape((-1, 3)).T).toarray()
    x0 = sim.position().T.ravel()
    bending_energy_list.append(bending_energy(x0))
    if not check_grad(stretching_and_shearing_energy, stretching_and_shearing_gradient, x0):
        stretching_and_shearing_gradient_score = 0
        stretching_and_shearing_hessian_score = 0
    print("stretching_and_shearing_gradient_score", stretching_and_shearing_gradient_score)
    if not check_grad(stretching_and_shearing_gradient, stretching_and_shearing_hessian, x0):
        stretching_and_shearing_hessian_score = 0
    if not check_grad(bending_energy, bending_gradient, x0):
        bending_gradient_score = 0
        bending_hessian_score = 0
    if not check_grad(bending_gradient, bending_hessian, x0):
        bending_hessian_score = 0
if not np.isclose(np.array(bending_energy_list).ravel(), bending_gt, 1e-4, 1e-8).all():
    bending_energy_score = bending_gradient_score = bending_hessian_score = 0
check_name = ['bending_energy_score', 'bending_gradient_score', 'bending_hessian_score', 'stretching_and_shearing_gradient_score', 'stretching_and_shearing_hessian_score']
check_list = [bending_energy_score, bending_gradient_score, bending_hessian_score, stretching_and_shearing_gradient_score, stretching_and_shearing_hessian_score]
for name, score in zip(check_name, check_list):
    print(name, ":", score)
