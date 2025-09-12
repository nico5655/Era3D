import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_fill_holes

def harmonic_prior_from_contour(
    contour_mask: np.ndarray,
    contour_depth: np.ndarray,
    region_mask: np.ndarray | None = None,
    connectivity: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a rough depth prior inside a closed contour by harmonic interpolation.
    Dirichlet BC: depth on the contour is fixed to contour_depth; interior is solved.

    Args:
        contour_mask: (H,W) bool, True on boundary pixels with known depth.
        contour_depth: (H,W) float, depth values; only used where contour_mask is True.
        region_mask: (H,W) bool domain to fill (inside the contour). If None, fill holes.
        connectivity: 4 or 8 neighbour Laplacian (4 is standard and robust).

    Returns:
        depth_prior: (H,W) float with NaN outside region; inside has the filled depths.
        prior_mask:  (H,W) bool, True where depth_prior is defined (the domain).
    """
    H, W = contour_mask.shape
    if region_mask is None:
        # Infer interior from the contour; assumes the contour is (mostly) closed
        region_mask = binary_fill_holes(contour_mask)

    # Domain & sets
    domain   = region_mask.astype(bool)
    boundary = contour_mask & domain
    interior = domain & ~boundary

    n_int = int(interior.sum())
    depth_prior = np.full((H, W), np.nan, dtype=float)
    depth_prior[boundary] = contour_depth[boundary]

    if n_int == 0:
        # Nothing to solve
        return depth_prior, domain

    # Index interior pixels compactly
    idx = -np.ones((H, W), dtype=int)
    idx[interior] = np.arange(n_int)

    # Neighbourhood
    if connectivity == 4:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    elif connectivity == 8:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    rows, cols, data = [], [], []
    rhs = np.zeros(n_int, dtype=float)

    # Assemble Laplace(z)=0 with Dirichlet boundary
    I, J = np.nonzero(interior)
    for i, j in zip(I, J):
        p = idx[i, j]
        diag = 0.0
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < H and 0 <= nj < W):  # outside image
                continue
            if not domain[ni, nj]:
                continue
            if interior[ni, nj]:
                rows.append(p); cols.append(idx[ni, nj]); data.append(-1.0)
                diag += 1.0
            else:  # neighbour on boundary -> contributes known value to RHS
                diag += 1.0
                rhs[p] += depth_prior[ni, nj]
        rows.append(p); cols.append(p); data.append(diag)

    L = csr_matrix((data, (rows, cols)), shape=(n_int, n_int))
    z_int = spsolve(L, rhs)

    depth_prior[interior] = z_int
    prior_mask = domain
    return depth_prior, prior_mask
depth_prior,prior_mask=harmonic_prior_from_contour((img_l[...,1]==1),img_l[...,0],img4==1)

import trimesh

def apply_symmetry_by_edition(surface, sym_plan_vec,transfo_center):
  ortho=sym_plan_vec
  p0=ntransfo_center[0]*np.array([1,1,-1])
  n=ortho[0]
  c=np.dot(p0,n)
  m2=surface.copy()
  nmesh=m2.submesh([(m2.triangles_center*n[None,:]).sum(axis=-1)<c])[0]

  R = np.eye(4)
  R[:3, :3] -= 2.0 * np.outer(n, n)
  R[:3, 3] = 2.0 * (c) * n

  mirrored = nmesh.copy()
  mirrored.apply_transform(R)
  pred=(m2.vertices*n[None,:]).sum(axis=-1)>c
  dists=((m2.vertices[pred][:,None,:]-mirrored.vertices[None,:,:])**2).sum(axis=-1)
  targs=np.argmin(dists,axis=-1)
  m2.vertices[pred]=mirrored.vertices[targs]
  return m2

m2=apply_symmetry_by_edition(surface1,sym_plan_vec,ntransfo_center)
m2.apply_translation(-m2.centroid*np.array([1,1,1]))
m2.show()