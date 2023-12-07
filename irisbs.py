from pydrake.all import RandomGenerator, HPolyhedron, VPolytope
import numpy as np
from visualization_utils import plot_HPoly

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def aligned_box(pa, pb, longitudinal_margin= 1e-3, lateral_margin = 0.1):
    assert len(pa) == len(pb)
    dim = len(pa)
    rab = pb-pa
    n = (rab/(np.linalg.norm(pb-pa) + 1e-6)).reshape(1,-1)
    basis = gs(np.concatenate((n, np.random.randn(dim-1, dim)), axis=0))
    A = np.zeros((2*dim, dim))
    b = np.zeros((2*dim))
    A[0, :] = -basis[0, :]
    b[0] = -basis[0, :]@pa + longitudinal_margin
    A[1, :] = basis[0, :]
    b[1] = basis[0, :]@pb + longitudinal_margin
    for i in range(1, dim):
        A[2*i, :] = -basis[i, :]
        b[2*i] = -basis[i, :]@pa + lateral_margin
        A[2*i+1, :] = basis[i, :]
        b[2*i+1] = basis[i, :]@pa + lateral_margin
    return HPolyhedron(A,b)



def bisect_collision_point(coll_free, coll, checker, N_steps=5):
    curr_coll = coll
    curr_coll_free = coll_free
    for _ in range(N_steps):
        curr = (curr_coll+curr_coll_free)/2
        if checker.CheckConfigCollisionFree(curr):
            curr_coll_free = curr
        else:
            #print('col updated')
            curr_coll = curr
    return curr_coll

def gradient_descent(ellipsoid, point_in_collision, checker, gradient_steps=1, bisection_steps=5):
    curr = point_in_collision.copy()
    ATA = ellipsoid.A().T@ellipsoid.A()
    for _ in range(gradient_steps):
        grad = ATA@(curr-ellipsoid.center())
        grad = grad/np.linalg.norm(grad)
        dist = -(grad.T@ATA@(curr-ellipsoid.center()))/(grad.T@ATA@grad)
        next_point = curr + dist*grad
        if not checker.CheckConfigCollisionFree(next_point):
            curr = next_point
        else:
            #do bisection
            curr_next = bisect_collision_point(next_point, 
                                               curr, 
                                               checker, 
                                               N_steps=bisection_steps)
            if np.array_equal(curr, curr_next):
                break
            else:
                curr[:] = curr_next[:]
    return curr

def tangent_hyperplane(ellipse, point, backoff):
    a = ellipse.A().T@ellipse.A()@(point - ellipse.center())
    a = a/(np.linalg.norm(a) +1e-6)
    b = a@point - backoff
    return a.reshape(1,-1), b

def update_samples(samples, ellipse, checker, gradient_steps, bisection_steps, use_bisection_only):
    updates = []
    for s in samples:
        if use_bisection_only:
            boundary_point = bisect_collision_point(ellipse.center(), s, checker, N_steps = bisection_steps)
        else:
            boundary_point = gradient_descent(ellipse, s, checker, gradient_steps=gradient_steps, bisection_steps=bisection_steps)#
        updates.append(boundary_point)
    return updates

def IRISBS(domain, 
           starting_ellipse, 
           checker, 
           N = 100, 
           NumFails = 100, 
           backoff = 1e-5, 
           gradient_steps = 5, 
           bisection_steps = 5, 
           use_bisection_only = False, 
           plot = None):
    
    H = HPolyhedron(domain.A(), domain.b())
    seedp = starting_ellipse.center()
    gen = RandomGenerator(1)
    num_consec_fails = 0
    prev = seedp
    iters = 0
    while num_consec_fails < NumFails: 
        if iters%50 == 0:
            print(f"iterations: {iters}, #faces {H.A().shape[0]}, seed point contained {H.PointInSet(seedp)}")
        samples = []
        new_faces_a = []
        new_faces_b = []
        for _ in range(N):
            curr = H.UniformSample(gen, prev)
            samples.append(curr)
            prev = curr.copy()
        #print('sampling done')
        colfree = checker.CheckConfigsCollisionFree(np.array(samples))
        collision_idx = np.where(1-np.array(colfree))[0]
        samples_col = [samples[i] for i in collision_idx]

        samples_col_updated = update_samples(samples_col,
                                             starting_ellipse,
                                             checker, 
                                             gradient_steps, 
                                             bisection_steps,
                                             use_bisection_only) 

        samples_col_sorted = sorted(samples_col_updated, 
                            key=lambda x: 
                            (x - seedp).T@starting_ellipse.A().T@starting_ellipse.A()@(x - seedp))
        samples_col_sorted = np.array(samples_col_sorted)

        if plot is not None:
            s = np.array(samples)
            sc = np.array(samples_col)
           # plot.scatter(s[:,0], s[:,1], c = 'k', s= 1)
            if len(samples_col):
                plot.scatter(sc[:,0], sc[:,1], s = 20, c = 'r')
            
        #print(np.where(1-np.array(in_col)))
        #print(f"nr in col {len(remaining_idx)}")
        if not len(collision_idx):
            num_consec_fails +=1
            #print("hahah", num_consec_fails)
        else:
            num_consec_fails = 0
      
        while len(samples_col_sorted):
            boundary_point = samples_col_sorted[0]
            
            if plot is not None:
                start = np.array(samples_col)[np.where(boundary_point[0] == np.array(samples_col_updated)[:,0])[0]]
                plot.plot([start[0,0], boundary_point[0]], [start[0,1], boundary_point[1]], c = 'k')
                plot.scatter([start[0,0], boundary_point[0]], [start[0,1], boundary_point[1]], c = 'g', s = 20)
                plot.text(start[0,0], start[0,1],f"{len(new_faces_a)} iter {iters}", fontsize = 8)

            a, b = tangent_hyperplane(starting_ellipse, boundary_point, backoff)
            samples_col_sorted = np.delete(samples_col_sorted, 0 , axis = 0)
            is_redunant = np.where([a@s-b>=-0.1 for s in samples_col_sorted])[0]
            samples_col_sorted = np.delete(samples_col_sorted, is_redunant, axis=0)
            new_faces_a.append(a)
            new_faces_b.append(b)
            
        H = HPolyhedron(np.concatenate(tuple([H.A()] + new_faces_a ), axis = 0,), 
                        np.concatenate((H.b(), np.array(new_faces_b) ), axis = 0,))
        plot_HPoly(plot, H)
        #print(H.PointInSet(seedp))
        #print(H.A().shape)
        #print(H.MaximumVolumeInscribedEllipsoid().Volume())
        iters+=1
        prev = H.UniformSample(gen)
    return H