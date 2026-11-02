import math
from dataclasses import dataclass, field

import igl
import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers import LegacyCoupler
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_batch
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils import array_class
from genesis.utils.misc import qd_to_torch, tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud

from .base_solver import GravityMixin, Solver, TimeBasedMixin


# Number of rigid links whose boundary a fluid particle caches (see init_particle_fields).
N_BOUNDARY_SLOTS = 3

# Number of fluid neighbors a particle lists for the solves of a substep (see _func_for_all_fluid_neighbors). A particle
# at rest density has about 33 to 42 neighbors, depending on the volume factor; one with more searches the hash grid.
N_LISTED_NEIGHBORS = 64
# parts of the fluid load on a link (see get_links_fluid_wrench_parts)
LOAD_PARTS = ("pressure", "viscous", "clearance")
# activation diagnostics of the corrections, per environment and substep (see SPHOptions.correction_diagnostics): name,
# and whether the stats sum it over the substeps (True) or take its largest value (False)
DIAGNOSTICS = (
    ("n_visc_particles", True),
    ("n_visc_blend_partial", True),
    ("n_visc_blend_zero", True),
    ("n_visc_pairs", True),
    ("n_visc_pairs_clipped", True),
    ("n_shear_scaled", True),
    ("n_shear_fallback", True),
    ("n_shear_at_low_bound", True),
    ("n_shear_at_high_bound", True),
    ("shear_scale_sum", True),
    ("shear_scale_sq_sum", True),
    ("max_projection_linear", False),
    ("max_projection_angular", False),
    ("clearance_shift_sum", True),
    ("clearance_impulse_sum", True),
    ("max_clearance_rotation", False),
    ("clearance_potential_energy_change", True),
)
N_DIAGNOSTICS = len(DIAGNOSTICS)
N_LOAD_PARTS = len(LOAD_PARTS)
N_LISTED_BOUNDARY_NEIGHBORS = 48
# resolution of the gap between a wall and the fluid nearest to it, in units per support radius (see
# boundary_particles_fluid_gap)
FLUID_GAP_UNITS = 1 << 24
# threads that share the fluid neighbors of a particle in the iterations of the pressure solves
N_NEIGHBOR_SHARES = 4
# particles summed serially by one thread before its atomic add to a per-environment accumulator: the atomic adds of
# every particle to the same few addresses serialize on the GPU (a 6x6 system per particle took 1.6 ms for 19k particles)
REDUCTION_CHUNK = 64

# Number of even spherical harmonics up to degree 4 (1 + 5 + 9), the directional basis of the correction of the viscosity
# operator (see _kernel_compute_viscosity_geometry).
N_VISCOSITY_BASIS = 15


def sample_reentrant_edges(verts, faces, offset, spacing, min_angle=math.radians(10.0)):
    """Return the boundary samples (positions, normals, areas) that close the offset surface at the reentrant edges.

    The boundary particles of a face are moved by the boundary offset into the geom along the normal of the face. At an
    edge where the surface turns into the geom (the inner corner of a cup), the two moved faces stop short of each
    other and leave a slot as wide as the offset, through which fluid can enter the geom. Each moved face is extended
    to the line where the two moved planes meet, offset * tan(angle / 2) beyond the edge, with the angle between the
    face normals, by rows of samples along the edge at the boundary spacing. Edges turning by less than min_angle (the
    facets of a smooth curved surface) leave a negligible slot and are skipped.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    pos, normals, areas = [], [], []
    face_normals = mesh.face_normals
    for (f_1, f_2), edge, is_convex in zip(mesh.face_adjacency, mesh.face_adjacency_edges, mesh.face_adjacency_convex):
        if is_convex:
            continue
        cos_angle = float(np.clip(np.dot(face_normals[f_1], face_normals[f_2]), -1.0, 1.0))
        angle = math.acos(cos_angle)
        if angle < min_angle or angle > math.pi - min_angle:
            continue
        extension = offset * math.tan(0.5 * angle)
        p_0, p_1 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
        length = float(np.linalg.norm(p_1 - p_0))
        if length <= 0.0:
            continue
        tangent = (p_1 - p_0) / length
        n_along = max(math.ceil(length / spacing), 1)
        n_across = max(math.ceil(extension / spacing), 1)
        along = (np.arange(n_along) + 0.5) / n_along * length
        across = np.arange(1, n_across + 1) / n_across * extension
        for i_f in (f_1, f_2):
            normal = face_normals[i_f]
            # direction in the plane of the face, across the edge, away from the face
            direction = np.cross(normal, tangent)
            vertex_off_edge = [v for v in mesh.faces[i_f] if v not in edge][0]
            if np.dot(mesh.vertices[vertex_off_edge] - p_0, direction) > 0.0:
                direction = -direction
            grid = p_0 + along[:, None, None] * tangent + across[None, :, None] * direction - offset * normal
            pos.append(grid.reshape(-1, 3))
            normals.append(np.tile(normal, (n_along * n_across, 1)))
            areas.append(np.full(n_along * n_across, length / n_along * extension / n_across))
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pos), np.concatenate(normals), np.concatenate(areas)


def compute_volume_map(signed_distance, lower, cell_size, n_nodes, support_radius):
    """Return the boundary volume of a solid at the nodes of a grid, the integral of the smoothing kernel over it.

    'signed_distance' maps points (N, 3) to their signed distance to the surface of the solid, negative inside. The grid
    has 'n_nodes' nodes along each axis, spaced by 'cell_size' from 'lower'. The signed distance is evaluated on a grid
    four times coarser and interpolated, then evaluated again at the nodes near the surface, where the interpolation may
    get its sign wrong. The solid is the occupancy of the nodes, smoothed over one cell across the surface, convolved
    with the cubic spline kernel by FFT. The kernel is normalized on the grid, so that the volume is 1 deep inside the
    solid, as the fluid density of a particle surrounded by fluid (Koschier and Bender 2017, density maps).
    """
    n_nodes = np.asarray(n_nodes, dtype=np.int64)
    lower = np.asarray(lower, dtype=np.float64)
    coarse = 4
    n_coarse = (n_nodes - 2) // coarse + 2
    axes = [lower[i] + coarse * cell_size * np.arange(n_coarse[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    phi_coarse = torch.as_tensor(signed_distance(points).reshape(tuple(n_coarse)), dtype=torch.float32)
    # coarse node k sits at fine node coarse * k
    phi = torch.nn.functional.interpolate(
        phi_coarse[None, None],
        size=tuple(int(n) for n in coarse * (n_coarse - 1) + 1),
        mode="trilinear",
        align_corners=True,
    )[0, 0, : n_nodes[0], : n_nodes[1], : n_nodes[2]].contiguous()
    # A signed distance changes by at most the distance between two points, so the interpolation error stays below the
    # diagonal of a coarse cell.
    near = phi.abs() < math.sqrt(3.0) * coarse * cell_size
    near_idx = near.nonzero().numpy()
    if len(near_idx) > 0:
        phi[near] = torch.as_tensor(signed_distance(lower + cell_size * near_idx), dtype=torch.float32)
    occupancy = (0.5 - phi / cell_size).clamp(0.0, 1.0)

    # kernel on the grid, its center at index 0 of a periodic array large enough that the solid does not wrap around
    r_cells = math.ceil(support_radius / cell_size)
    size = tuple(int(n) + 2 * r_cells for n in n_nodes)
    offsets = np.arange(-r_cells, r_cells + 1) * cell_size
    dist = np.linalg.norm(np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1), axis=-1)
    q = dist / support_radius
    kernel = np.where(q <= 0.5, 6.0 * q**3 - 6.0 * q**2 + 1.0, np.where(q <= 1.0, 2.0 * (1.0 - q) ** 3, 0.0))
    kernel /= kernel.sum()
    kernel_periodic = torch.zeros(size, dtype=torch.float64)
    idx = [torch.as_tensor(np.arange(-r_cells, r_cells + 1) % n) for n in size]
    kernel_periodic[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]] = torch.as_tensor(kernel)
    occupancy_padded = torch.zeros(size, dtype=torch.float64)
    occupancy_padded[r_cells:-r_cells, r_cells:-r_cells, r_cells:-r_cells] = occupancy
    volume = torch.fft.irfftn(torch.fft.rfftn(occupancy_padded) * torch.fft.rfftn(kernel_periodic), s=size)
    volume = volume[r_cells:-r_cells, r_cells:-r_cells, r_cells:-r_cells]
    return volume.clamp(0.0, 1.0).numpy().astype(gs.np_float)


def half_space_volume(dist, support_radius):
    """Return the boundary volume of a half-space at a signed distance from its surface (negative inside).

    Integral of the cubic spline kernel over the half-space, 16 times the integral of the plane integral of the kernel
    (see SPHSolver.cubic_kernel_plane_integral) from |dist| to the support radius, mirrored as 1 minus it inside.
    """
    q = np.minimum(np.abs(dist) / support_radius, 1.0)

    def p_outer(s):
        return 0.1 * s - s**3 / 3.0 + 0.5 * s**4 - 0.3 * s**5 + 0.4 / 6.0 * s**6

    def p_inner(s):
        return 0.0875 * s - s**3 / 6.0 + 0.3 * s**5 - 0.2 * s**6

    tail = np.where(
        q > 0.5,
        p_outer(1.0) - p_outer(q),
        p_outer(1.0) - p_outer(0.5) + p_inner(0.5) - p_inner(q),
    )
    volume = 16.0 * tail
    return np.where(dist >= 0.0, volume, 1.0 - volume)


def flush_lattice_volume_offset(particle_volume, support_radius):
    """Return the distance by which to shrink the solid of a volume map so that fluid sampled flush starts at rest.

    A cubic lattice of spacing V^(1/3), whose first layer lies half a spacing from a planar surface, has at its first
    layer the density of its half lattice plus the boundary volume of the half-space shrunk by the returned distance,
    which this distance makes 1.
    """
    spacing = particle_volume ** (1.0 / 3.0)
    n = math.ceil(support_radius / spacing)
    k = np.arange(-n, n + 1)
    nodes = np.stack(np.meshgrid(k, k, np.arange(n + 1), indexing="ij"), axis=-1).reshape((-1, 3)) * spacing
    q = np.linalg.norm(nodes, axis=-1) / support_radius
    kernel = np.where(q <= 0.5, 6.0 * q**3 - 6.0 * q**2 + 1.0, np.where(q <= 1.0, 2.0 * (1.0 - q) ** 3, 0.0))
    half_lattice = particle_volume * 8.0 / (np.pi * support_radius**3) * kernel.sum()
    dist = np.linspace(support_radius, 0.0, 20001)
    dist_rest = np.interp(1.0 - half_lattice, half_space_volume(dist, support_radius), dist)
    return max(float(dist_rest) - 0.5 * spacing, 0.0)


def volume_map_to_distance(volume, support_radius):
    """Return the signed distance at which a half-space has the given boundary volume, within the support radius.

    A volume map stores this distance rather than the volume (see SPHSolver._func_volume_map): the distance to a
    planar surface is linear, which the trilinear interpolation of the grid reproduces exactly, where the interpolated
    volume would ripple along a surface not aligned with the grid, a roughness of the wall.
    """
    dist = np.linspace(support_radius, -support_radius, 20001)
    return np.interp(volume, half_space_volume(dist, support_radius), dist).astype(gs.np_float)


@dataclass
class SPHSolveStats:
    """Iteration counts and residuals of the solves of the SPH solver, summed over the substeps of the last step.

    A residual is the final error of a solve over its tolerance, so a solve has converged when it is at most 1. A solve
    is unconverged when it stops at its iteration cap with a residual above 1, and a substep holding one is no valid
    accuracy or performance sample.
    """

    n_substeps: int = 0
    n_divergence_iters: int = 0
    n_density_iters: int = 0
    n_viscosity_iters: int = 0
    n_coupling_passes: int = 0
    n_unconverged_divergence: int = 0
    n_unconverged_density: int = 0
    n_unconverged_viscosity: int = 0
    n_unconverged_coupling: int = 0
    max_divergence_residual: float = 0.0
    max_density_residual: float = 0.0
    max_viscosity_residual: float = 0.0
    max_coupling_residual: float = 0.0
    # with SPHOptions(density_correction="position"), iterations of the drift correction of the positions, substeps
    # where the cap stopped it, and its largest final error over the tolerance (see SPHSolver._drift_correction)
    n_drift_iters: int = 0
    n_unconverged_drift: int = 0
    max_drift_residual: float = 0.0
    # repetitions of the fluid stage after the constraint solve of the rigid solver changed a coupled link (see
    # SPHSolver.repeat_for_rigid_constraints), substeps where the cap stopped them, and the largest final disagreement
    n_constraint_passes: int = 0
    n_unconverged_constraint: int = 0
    max_constraint_residual: float = 0.0
    # fluid particles near more rigid links than they cache, whose surplus links they ignore (see N_BOUNDARY_SLOTS)
    n_boundary_slot_overflows: int = 0
    # fluid particles whose clearance from a rigid surface the wall clearance guard restored (see
    # _kernel_enforce_wall_clearance), summed over the substeps
    n_clearance_corrections: int = 0
    # per substep, per coupling pass: (fluid coupling residual, rigid velocity change residual, density iterations,
    # viscosity iterations), each residual relative to its tolerance
    coupling_history: list = field(default_factory=list)
    # with SPHOptions.energy_audit, the change of kinetic energy of the fluid in J caused by each stage of the substeps,
    # summed over the substeps and environments: divergence solve, explicit forces (gravity, surface tension), coupled
    # viscosity and density solves, wall clearance guard, and the drift correction with the domain bounds after
    # advection
    ke_change_divergence: float = 0.0
    ke_change_forces: float = 0.0
    ke_change_coupled: float = 0.0
    ke_change_clearance: float = 0.0
    ke_change_bounds: float = 0.0
    # with SPHOptions.correction_diagnostics, how often and how strongly the corrections act, summed over the substeps
    # and environments (largest value for max_*): particles of the viscosity correction, those in its free-surface blend
    # (normalized density 0.85-0.95) and below it (isotropic scale), their fluid pairs and the pairs whose weight is
    # clipped at zero; particles near a wall whose shear consistency scale is computed, those that keep the scale 1 (a
    # corner or a free surface, see _func_wall_shear_scale), those at its bounds 0.25 and 4, and the sum and squared sum
    # of the computed scales; the largest rigid velocity correction of the projection of the viscosity residual (m/s,
    # rad/s); the summed position shifts (m) and corrective impulses (N s) of the wall clearance guard, its largest
    # rotational velocity correction (rad/s) and the change of potential energy of its shifts (J)
    n_visc_particles: int = 0
    n_visc_blend_partial: int = 0
    n_visc_blend_zero: int = 0
    n_visc_pairs: int = 0
    n_visc_pairs_clipped: int = 0
    n_shear_scaled: int = 0
    n_shear_fallback: int = 0
    n_shear_at_low_bound: int = 0
    n_shear_at_high_bound: int = 0
    shear_scale_sum: float = 0.0
    shear_scale_sq_sum: float = 0.0
    max_projection_linear: float = 0.0
    max_projection_angular: float = 0.0
    clearance_shift_sum: float = 0.0
    clearance_impulse_sum: float = 0.0
    max_clearance_rotation: float = 0.0
    clearance_potential_energy_change: float = 0.0


@qd.data_oriented
class SPHSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = SPH.Base
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        # Divergence-free SPH (DFSPH) parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_chebyshev_radius = options.pressure_chebyshev_radius
        self._df_min_div_iters = options.min_divergence_solver_iterations
        self._df_min_den_iters = options.min_density_solver_iterations
        self._df_skip_div = options.skip_divergence_solver
        self._df_warm_start = options.warm_start
        self._micropolar = options.micropolar
        self._visc_max_error = options.max_viscosity_error
        self._visc_max_iters = options.max_viscosity_solver_iterations
        self._max_coupling_iters = options.max_coupling_iterations
        self._max_constraint_iters = options.max_constraint_iterations
        self._max_constraint_error = options.max_constraint_error
        self._n_constraint_passes = 0
        self._energy_audit = options.energy_audit
        self._isotropic_pressure = options.isotropic_pressure
        self._isotropic_density = options.isotropic_density
        self._isotropic_relaxation = options.isotropic_pressure_relaxation
        self._wall_shear_consistency = options.wall_shear_consistency
        # benchmark switches of the corrections (SPH_SOLVER_GOAL.md N6), the defaults being the solver
        self._viscosity_correction = options.viscosity_correction
        self._wall_viscosity_model = options.wall_viscosity_model
        self._n_wall_ghosts = 1 if options.wall_viscosity_model == "boundary_pairs" else options.wall_ghost_points
        self._viscosity_residual_projection = options.viscosity_residual_projection
        self._clearance_velocity = options.wall_clearance_velocity
        self._clearance_position = options.wall_clearance_position
        self._clearance_rotation = options.wall_clearance_rotation
        self._correction_diagnostics = options.correction_diagnostics
        self._position_drift = options.density_correction == "position"
        self._max_drift_iters = options.max_drift_solver_iterations
        # Whether any material has a viscosity, which enables the implicit viscosity solve. Resolved at build time.
        self._has_viscosity = False
        self._df_eps = 1e-5
        # A particle with fewer neighbors sits at a free surface, where the divergence estimate is unreliable.
        self._df_min_div_neighbors = 20

        self._particle_volume = options.particle_volume_factor * self._particle_size**3

        # Boundary particles sampled on the surface of the rigid geoms, on a regular triangle sampling denser than the
        # fluid so that the fluid cannot leak through (Akinci et al. 2012). The tangential part of their pressure force
        # acts as a roughness of the wall, which drops as the spacing shrinks from 0.75 to 0.5 particle sizes and
        # little below (SPH_SOLVER_STATUS.md, finding 2).
        self._boundary_spacing = 0.5 * self._particle_size
        # Upper bound on the number of boundary particles, which grows with the area of the simulation domain.
        self._max_n_boundary_particles = 5_000_000
        # Upper bound on the number of grid nodes of the volume maps.
        self._max_n_volume_map_nodes = 200_000_000
        # With volume maps (see compute_volume_map), the maps set how close the fluid comes to the surface, and the
        # boundary particles, which carry the wall viscosity and the wall clearance, sit on the surface itself.
        self._has_volume_maps = options.boundary_model == "volume_map"
        self._volume_map_cell_size = options.volume_map_cell_size
        self._volume_map_offset = options.volume_map_offset
        self._boundary_offset = 0.0 if self._has_volume_maps else options.boundary_offset
        self._n_boundary_particles = 0
        self._has_boundary = False
        # Links with degrees of freedom that carry boundary particles, which exchange momentum with the fluid in both
        # directions (SPH_SOLVER_DESIGN.md section 2). Resolved at build time.
        self._coupled_links_idx = np.zeros(0, dtype=gs.np_int)
        self._has_rigid_coupling = False
        self._solve_stats = SPHSolveStats()

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

    def init_particle_fields(self):
        # dynamic particle state
        # TODO: Some fields may not need to be in both reordered and un-reoredered.
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # non-pressure acceleration
            material=gs.qd_int,  # material index
            reordered_idx=gs.qd_int,  # TODO: see above.
            active=gs.qd_bool,
            # Micro-rotation (Bender et al. 2017, "Turbulent Micropolar SPH Fluids with Foam"), allocated on demand.
            **(dict(angular_vel=gs.qd_vec3, angular_acc=gs.qd_vec3) if self._micropolar else {}),
            # --- DFSPH fields ---
            # 1 / (|sum_j V_j grad W_ij|^2 + sum_j |V_j grad W_ij|^2), i.e. alpha * rho_0^2 / rho
            alpha_factor_adj=gs.qd_float,
            # sum_j V_j (a_i - a_j) . grad W_ij, the density change rate induced by the current pressure
            predicted_source=gs.qd_float,
            # p / rho^2 of the density solve, kept across substeps for warm starting
            pressure_density2_norm_p=gs.qd_float,
            # p / rho^2 of the divergence solve, kept across substeps for warm starting
            pressure_density2_norm_d=gs.qd_float,
            # pressure acceleration
            pressure_acc=gs.qd_vec3,
            # predicted density rho* / rho_0 (density solve), or density change rate (divergence solve)
            density_adv_norm=gs.qd_float,
            # rho / rho_0 = sum_j V_j W_ij
            density_norm=gs.qd_float,
            # number of neighbors within the support radius
            n_neighbors=gs.qd_int,
            # directional correction of the pressure pair weights of the particle in the degree 2 spherical harmonics
            # (see _kernel_compute_pressure_correction)
            **(dict(pressure_correction=qd.types.vector(5, gs.qd_float)) if self._isotropic_pressure else {}),
        )

        # static material info
        struct_material_info = qd.types.struct(
            rest_volume=gs.qd_float,
            mass=gs.qd_float,
            rest_density=gs.qd_float,  # mass / rest_volume
            # mu = rest_density * nu, in Pa s (the material option is the kinematic viscosity nu, in m^2/s)
            dynamic_viscosity=gs.qd_float,
            surface_tension=gs.qd_float,
            vorticity=gs.qd_float,
            angular_viscosity=gs.qd_float,
            inverse_microinertia=gs.qd_float,  # 1 / microinertia
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        # One material per entity.
        self.materials = struct_material_info.field(shape=(len(self.entities),), needs_grad=False, layout=qd.Layout.AOS)

        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        # Fluid neighbors of every particle within the support radius, in hashed (reordered) order, listed by the density
        # kernel of the substep for the later stages, whose positions do not change (see
        # _func_for_all_fluid_neighbors). The neighbor rank comes first, so that the lists of consecutive particles lie
        # side by side in memory.
        self.particles_n_fluid_neighbors = qd.field(gs.qd_int, shape=(self._n_particles, self._B))
        # pressure of the previous iteration of the running divergence or density solve (see _pressure_solve)
        self.particles_pressure_prev = qd.field(gs.qd_float, shape=(self._n_particles, self._B))
        self.particles_fluid_neighbors = qd.field(gs.qd_int, shape=(N_LISTED_NEIGHBORS, self._n_particles, self._B))
        # partial sums of the shares of the neighbors of a particle in the iterations of the pressure solves (see
        # _kernel_pressure_solve_iteration)
        self.particles_acc_shares = qd.Vector.field(
            3, dtype=gs.qd_float, shape=(N_NEIGHBOR_SHARES, self._n_particles, self._B)
        )
        self.particles_source_shares = qd.field(gs.qd_float, shape=(N_NEIGHBOR_SHARES, self._n_particles, self._B))

        # Per-environment reductions of the DFSPH solver iterations.
        self.envs_n_active = qd.field(gs.qd_int, shape=(self._B,))
        self.envs_solver_error = qd.field(gs.qd_float, shape=(self._B,))
        # activation diagnostics of the corrections in the current substep (see DIAGNOSTICS)
        self.envs_diagnostics = qd.field(gs.qd_float, shape=(self._B, N_DIAGNOSTICS))
        # Environments stop their solves independently (candidate 9 of SPH_SOLVER_GOAL.md): an environment takes part in
        # the coupling passes of a substep until its own coupling has converged, and in the iterations of a pressure
        # solve until its own error meets its own target (see _run_fluid_stage and _kernel_pressure_solve_error). The
        # kernels of the passes skip the other environments, which keep the state of their last pass or iteration.
        self.envs_coupling_active = qd.field(gs.qd_int, shape=(self._B,))
        self.envs_solve_active = qd.field(gs.qd_int, shape=(self._B,))
        self.envs_solve_target = qd.field(gs.qd_float, shape=(self._B,))
        self.envs_coupling_rel = qd.field(gs.qd_float, shape=(self._B,))
        # error of every particle in the last iteration of the running divergence or density solve, summed over the
        # environment only when the solve reads it (see _kernel_pressure_solve_error)
        self.particles_solver_error = qd.field(gs.qd_float, shape=(self._n_particles, self._B))
        # terms of every particle in the sums over the environment of the viscosity solve, the coupling residual and
        # the dissipation readout, summed in chunks (see _func_chunk_sum)
        self.particles_reduction = qd.Vector.field(4, dtype=gs.qd_float, shape=(self._n_particles, self._B))
        if self._position_drift:
            # Drift correction of the positions (see _drift_correction): pressure multiplier, its acceleration, the
            # predicted density and the displacement of each particle; error, mass, center of mass, inertia and the
            # angular momentum change of each environment
            struct_drift = qd.types.struct(
                kappa=gs.qd_float, kappa_prev=gs.qd_float, acc=gs.qd_vec3, density_adv=gs.qd_float, disp=gs.qd_vec3
            )
            self.particles_drift = struct_drift.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
            self.envs_drift_error = qd.field(gs.qd_float, shape=(self._B,))
            self.envs_drift_count = qd.field(gs.qd_float, shape=(self._B,))
            self.envs_drift_mass = qd.field(gs.qd_float, shape=(self._B,))
            self.envs_drift_center = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._B,))
            self.envs_drift_inertia = qd.Matrix.field(3, 3, dtype=gs.qd_float, shape=(self._B,))
            self.envs_drift_angular = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._B,))

        if self._has_viscosity:
            # State of the implicit viscosity solve (M + h D) v = M v_rhs, by preconditioned conjugate gradient, in
            # hashed (reordered) particle order. See SPH_SOLVER_DESIGN.md section 3.4.
            struct_viscosity_state = qd.types.struct(
                # velocity after the explicit accelerations of the substep (v*)
                vel_star=gs.qd_vec3,
                # velocity change of the last density solve, and its change since the previous coupling pass
                pressure_vel=gs.qd_vec3,
                pressure_vel_delta=gs.qd_vec3,
                residual=gs.qd_vec3,
                residual_precond=gs.qd_vec3,
                direction=gs.qd_vec3,
                # image of the search direction by the system matrix
                direction_image=gs.qd_vec3,
                # inverse of the 3x3 diagonal block of the system matrix (block-Jacobi preconditioner)
                precond_inv=gs.qd_mat3,
                # sum of the wall viscosity tensors of the particle (see _func_wall_viscosity_tensor)
                wall_tensor=gs.qd_mat3,
                # coefficients of the directional correction of the viscosity operator of the particle in the even
                # spherical harmonics (see _kernel_compute_viscosity_geometry)
                viscosity_correction=qd.types.vector(N_VISCOSITY_BASIS, gs.qd_float),
                # Chebyshev acceleration of the coupling passes (see _kernel_accelerate_coupling): the pressures the
                # current and the previous pass started from
                pressure_in=gs.qd_float,
                pressure_in_prev=gs.qd_float,
            )
            self.particles_viscosity = struct_viscosity_state.field(
                shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )
            struct_viscosity_env = qd.types.struct(
                # preconditioned squared residual norm r^T z, and its value after the current update
                residual_norm2=gs.qd_float,
                residual_norm2_next=gs.qd_float,
                # d^T A d of the current search direction
                direction_norm2=gs.qd_float,
                # squared preconditioned norm of the residual at the start of the solve, the scale of the tolerance
                residual_norm2_init=gs.qd_float,
                # squared preconditioned norm of the right-hand side, the scale of the floor of the tolerance
                rhs_norm2=gs.qd_float,
                # residual norm of the viscosity-pressure agreement
                coupling_norm2=gs.qd_float,
                # Upper bound of the eigenvalues of M^-1 (M + h D), from the Gershgorin disks of the viscosity system
                viscous_stiffness_bound=gs.qd_float,
                # Chebyshev step, squared spectrum ratio and weight of the current coupling pass
                chebyshev_step=gs.qd_float,
                chebyshev_ratio2=gs.qd_float,
                chebyshev_weight=gs.qd_float,
                # squared preconditioned norms of the viscous impulse of the first coupling pass of the substep and of
                # the gravity impulse, the scales of the tolerance of the viscosity-pressure agreement
                viscous_norm2=gs.qd_float,
                gravity_norm2=gs.qd_float,
                # squared mass norm of the change of the pressure velocity by the last density solve, and the mass
                delta_norm2=gs.qd_float,
                mass=gs.qd_float,
            )
            self.envs_viscosity = struct_viscosity_env.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.AOS)
            # Rigid velocity correction that removes the net force and torque of the residual of the viscosity solve
            # (see _kernel_project_viscosity_residual): mass and center of mass of the fluid, the 6x6 system of the
            # correction, its right-hand side (net residual force and torque about the center) and its solution.
            self.envs_viscosity_mass = qd.field(gs.qd_float, shape=(self._B,))
            self.envs_viscosity_center = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._B,))
            self.envs_viscosity_rigid_system = qd.Matrix.field(6, 6, dtype=gs.qd_float, shape=(self._B,))
            self.envs_viscosity_rigid_rhs = qd.Vector.field(6, dtype=gs.qd_float, shape=(self._B,))
            self.envs_viscosity_rigid_correction = qd.Vector.field(6, dtype=gs.qd_float, shape=(self._B,))
            # Kinetic energy the viscosity dissipates over the substeps of the current step, in J (see
            # _kernel_accumulate_viscous_dissipation).
            self.envs_viscous_dissipation = qd.field(gs.qd_float, shape=(self._B,))

        if self._has_boundary:
            # Fluid load on each rigid link, accumulated over the substeps of the current step: impulses in world frame,
            # the torque about the center of mass of the link. See SPH_SOLVER_DESIGN.md section 8.
            n_links = self.sim.rigid_solver.n_links
            self.links_fluid_impulse = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_fluid_angular_impulse = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            # the parts of that load due to the pressure, the viscosity and the wall clearance guard (index 0, 1, 2),
            # whose sum it is (see get_links_fluid_wrench_parts)
            self.links_fluid_part_impulse = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(n_links, self._B, N_LOAD_PARTS)
            )
            self.links_fluid_part_angular_impulse = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(n_links, self._B, N_LOAD_PARTS)
            )
            # center of mass of each link in world frame at the current substep, the reference point of the torque
            self.links_com = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            # spatial velocity of each link at the current substep (see func_vel_at_point), which gives the rigid
            # velocity anywhere in space, as the wall viscosity needs it at the fluid particles
            self.links_root_com = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_vel = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_ang = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            if self._has_volume_maps:
                # Volume maps (see compute_volume_map), flattened one after another in C order, with the link frame
                # position of their first node, their number of nodes along each axis and their first value, and the
                # map of each link (-1 if none) with the pose of the link at the current substep.
                n_maps = len(self._volume_maps)
                n_values = sum(values.size for *_, values in self._volume_maps)
                self.volume_maps_values = qd.field(gs.qd_float, shape=(n_values,))
                self.volume_maps_lower = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_maps,))
                self.volume_maps_n_nodes = qd.Vector.field(3, dtype=gs.qd_int, shape=(n_maps,))
                self.volume_maps_start = qd.field(gs.qd_int, shape=(n_maps,))
                self.links_volume_map = qd.field(gs.qd_int, shape=(n_links,))
                self.links_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
                self.links_quat = qd.Vector.field(4, dtype=gs.qd_float, shape=(n_links, self._B))
            # Boundary of every fluid particle, per rigid link near it, in hashed (reordered) particle order: the
            # gradient of its density constraint (see _kernel_compute_density_and_factor) and the wall viscosity tensor
            # (see _kernel_compute_viscosity_geometry). A slot holds the link index -1 when unused. The pressure and
            # viscosity solves read these aggregates instead of the boundary particles.
            struct_boundary_slot = qd.types.struct(
                link_idx=gs.qd_int,
                grad=gs.qd_vec3,
                wall_tensor=gs.qd_mat3,
                # signed distance of the fluid particle to the geom surface of the link along the outward normal of the
                # boundary particle it faces most closely, and that normal (see _task_compute_boundary_density_and_factor)
                clearance=gs.qd_float,
                clearance_normal=gs.qd_vec3,
                # impulse of the wall clearance guard on the link over the substep (see _kernel_enforce_wall_clearance)
                clearance_impulse=gs.qd_vec3,
            )
            self.particles_boundary_slots = struct_boundary_slot.field(
                shape=(self._n_particles, self._B, N_BOUNDARY_SLOTS), needs_grad=False, layout=qd.Layout.SOA
            )
            self.envs_boundary_slot_overflows = qd.field(gs.qd_int, shape=(self._B,))
            # boundary neighbors of every fluid particle, listed by the density kernel of the substep (see
            # _func_for_all_boundary_neighbors)
            self.particles_boundary_neighbors = qd.field(
                gs.qd_int, shape=(N_LISTED_BOUNDARY_NEIGHBORS, self._n_particles, self._B)
            )
            self.particles_n_boundary_neighbors = qd.field(gs.qd_int, shape=(self._n_particles, self._B))
            # wall viscosity tensors of the boundary slots summed over a share of the boundary neighbors of a particle
            # (see _kernel_compute_viscosity_geometry)
            self.particles_wall_tensor_shares = qd.Matrix.field(
                3, 3, dtype=gs.qd_float, shape=(N_NEIGHBOR_SHARES, self._n_particles, self._B, N_BOUNDARY_SLOTS)
            )
            self.envs_clearance_corrections = qd.field(gs.qd_int, shape=(self._B,))
            # mass, center of mass, inertia about it and angular momentum change of the position recovery of the wall
            # clearance guard, then the angular velocity of its rigid correction (see _kernel_enforce_wall_clearance)
            self.envs_clearance_mass = qd.field(gs.qd_float, shape=(self._B,))
            self.envs_clearance_center = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._B,))
            self.envs_clearance_inertia = qd.Matrix.field(3, 3, dtype=gs.qd_float, shape=(self._B,))
            self.envs_clearance_angular = qd.Vector.field(3, dtype=gs.qd_float, shape=(self._B,))

        if self._has_rigid_coupling:
            # Rigid side of the coupling, in the space of the spatial velocities of the coupled links (angular then
            # linear part, about the center of mass of their kinematic tree, see func_vel_at_point), per environment.
            # The mobility Lambda = S M^-1 S^T maps a wrench on any coupled link to the change of the spatial
            # acceleration of every coupled link through the articulation, S stacking the motion subspaces of the
            # degrees of freedom of each link and its ancestors. See SPH_SOLVER_DESIGN.md section 5.
            rigid_solver = self.sim.rigid_solver
            n_coupled_6 = 6 * len(self._coupled_links_idx)
            self.links_coupled_idx = qd.field(gs.qd_int, shape=(rigid_solver.n_links,))
            self.coupled_links_idx = qd.field(gs.qd_int, shape=(len(self._coupled_links_idx),))
            self.coupled_mobility = qd.field(gs.qd_float, shape=(n_coupled_6, n_coupled_6, self._B))
            # spatial velocity seen by the fluid, its value before the impulses of the substep plus the smooth
            # acceleration, and its value at the start of the current solve
            self.coupled_vel = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_vel_star = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_vel_base = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # spatial acceleration of the response to the pressure of the current iteration, and its wrench
            self.coupled_acc = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_wrench = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # impulses of the fluid on the coupled links in the substep, from the divergence, viscosity and density
            # solves (wrenches times the substep)
            self.coupled_impulse_div = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_impulse_visc = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_impulse_den = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # wall viscosity of the coupled links in their spatial velocity space, K (block diagonal, 6 x 6 per link)
            # and the wrench of the fluid velocity, f, and the dense system (I + h Lambda K) xi = rhs
            self.coupled_visc_stiffness = qd.field(gs.qd_float, shape=(n_coupled_6, 6, self._B))
            self.coupled_visc_force = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_system = qd.field(gs.qd_float, shape=(n_coupled_6, n_coupled_6 + 1, self._B))
            # implicit correction of the velocity the fluid sees in the next coupling pass (see
            # _kernel_solve_coupled_viscosity), and the size of each coupled link, the distance from its frame origin to
            # its furthest boundary particle, which weighs its angular velocity in the convergence measure
            self.coupled_vel_delta = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_links_extent = qd.field(gs.qd_float, shape=(len(self._coupled_links_idx),))
            # disagreement between the coupled velocities the fluid saw in the last viscosity pass and their implicit
            # solution, relative to their magnitude
            self.envs_coupled_vel_change = qd.field(gs.qd_float, shape=(self._B,))
            # velocity change of the coupled links by the constraint solve of the rigid solver, held as an external
            # impulse by a repeated fluid stage, kept across substeps as the guess of the next one (see
            # repeat_for_rigid_constraints), and the fluid impulse the stage put in the smooth force and acceleration of
            # the dofs, which a repetition takes back
            self.coupled_vel_constraint = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.dofs_fluid_force = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))
            self.dofs_fluid_acc = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))
            self.envs_constraint_residual = qd.field(gs.qd_float, shape=(self._B,))
            # readouts at the start of the fluid stage, which a repetition restores
            self.links_fluid_impulse_saved = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(rigid_solver.n_links, self._B)
            )
            self.links_fluid_angular_impulse_saved = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(rigid_solver.n_links, self._B)
            )
            self.links_fluid_part_impulse_saved = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(rigid_solver.n_links, self._B, N_LOAD_PARTS)
            )
            self.links_fluid_part_angular_impulse_saved = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(rigid_solver.n_links, self._B, N_LOAD_PARTS)
            )
            self.envs_viscous_dissipation_saved = qd.field(gs.qd_float, shape=(self._B,))
            # right-hand side and solution of the solves with the rigid mass matrix
            self.dofs_coupling_rhs = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))
            self.dofs_coupling_sol = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))

        if self._has_boundary:
            # Boundary particles, sampled on the rigid geoms and moved with their link. They carry the static
            # sampling data as well, so that their hashed (reordered) copy is self-contained.
            struct_boundary_particle_state = qd.types.struct(
                pos_local=gs.qd_vec3,  # position in link frame
                normal_local=gs.qd_vec3,  # outward unit normal of the geom surface at the sample, in link frame
                normal=gs.qd_vec3,
                area=gs.qd_float,  # share of the geom surface the sample stands for
                link_idx=gs.qd_int,
                is_fixed=gs.qd_bool,  # whether the link is fixed wrt the world
                pos=gs.qd_vec3,
                vel=gs.qd_vec3,
                # Akinci et al. 2012 volume 1 / sum_l W_bl, which weighs a boundary particle by its sampling density.
                volume=gs.qd_float,
                reordered_idx=gs.qd_int,
                active=gs.qd_bool,
            )
            self.boundary_particles = struct_boundary_particle_state.field(
                shape=(self._n_boundary_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )
            self.boundary_particles_reordered = struct_boundary_particle_state.field(
                shape=(self._n_boundary_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )
            # Distance from the surface of the nearest fluid particle in front of each boundary particle, in hashed
            # (reordered) boundary order, over the substep (see _task_find_boundary_faces and _func_wall_ghost).
            # The gap in units of FLUID_GAP_UNITS per support radius, an integer so that the fluid particles take its
            # minimum by an integer atomic: a floating-point atomic minimum runs as a compare-and-swap loop on the GPU,
            # which took 1.1 ms per substep in the density kernel of FluidBench S1 R2 (SPH_SOLVER_STATUS.md finding 67).
            self.boundary_particles_fluid_gap = qd.field(gs.qd_int, shape=(self._n_boundary_particles, self._B))

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles
        self._n_reduction_chunks = max((self._n_particles + REDUCTION_CHUNK - 1) // REDUCTION_CHUNK, 1)

        self._coupler = self.sim._coupler

        if self.is_active:
            self.sh.build(self._B)

            # Sample the boundary particles of the rigid geoms coupled with the fluid, in link frame.
            boundaries_pos_local, boundaries_link_idx, boundaries_is_fixed, boundaries_active = [], [], [], []
            boundaries_normal_local, boundaries_area = [], []
            # signed distance functions of the geoms of each link, in link frame, for the volume maps
            links_signed_distance = {}
            rigid_solver = self.sim.rigid_solver
            if isinstance(self._coupler, LegacyCoupler) and self._coupler.options.rigid_sph and rigid_solver.is_active:
                # Fluid stays within the domain, so boundary particles further away than the support radius are
                # never anyone's neighbor.
                area_lower = self._lower_bound - self._support_radius
                area_upper = self._upper_bound + self._support_radius
                n_boundary_particles = 0
                for entity in rigid_solver.entities:
                    for link in entity.links:
                        link_pos = np.atleast_2d(tensor_to_array(link.get_pos(relative=False)))[0]
                        link_quat = np.atleast_2d(tensor_to_array(link.get_quat(relative=False)))[0]
                        for geom in link.geoms:
                            if not geom.needs_coup:
                                continue
                            if geom.type == gs.GEOM_TYPE.PLANE:
                                # A plane collides as an infinite half-space while its mesh is a large finite quad,
                                # so it is sampled over its cut through the boundary area instead.
                                plane_pos = gu.transform_by_trans_quat(geom.init_pos, link_pos, link_quat)
                                plane_normal = gu.transform_by_quat(
                                    gu.transform_by_quat(geom.data[:3], geom.init_quat), link_quat
                                )
                                verts, faces = gu.plane_aabb_intersection(
                                    plane_pos, plane_normal, area_lower, area_upper
                                )
                                verts = gu.inv_transform_by_trans_quat(verts, link_pos, link_quat)
                                # The winding of the cut polygon is arbitrary, so its face normal may point inward.
                                surface_normal = gu.transform_by_quat(geom.data[:3], geom.init_quat)
                                links_signed_distance.setdefault(link.idx, []).append(
                                    lambda x, p=geom.init_pos.copy(), n=surface_normal: (x - p) @ n
                                )
                            else:
                                verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                                faces = geom.init_faces
                                verts_f64 = np.asarray(verts, dtype=np.float64)
                                links_signed_distance.setdefault(link.idx, []).append(
                                    lambda x, v=verts_f64, f=np.asarray(faces): igl.signed_distance(
                                        np.asarray(x, dtype=np.float64), v, f
                                    )[0]
                                )
                            if len(faces) == 0:
                                continue

                            # Number of points of a regular triangular lattice with the boundary spacing.
                            area = trimesh.Trimesh(vertices=verts, faces=faces, process=False).area
                            n_points = math.ceil(2.0 * area / (math.sqrt(3.0) * self._boundary_spacing**2))
                            n_boundary_particles += n_points
                            if n_boundary_particles > self._max_n_boundary_particles:
                                gs.raise_exception(
                                    f"Sampling the SPH boundary of geom {geom.idx} of entity {entity.idx} requires "
                                    f"{n_points} particles, which brings the total above "
                                    f"{self._max_n_boundary_particles}. Please tighten the SPH domain "
                                    "(`lower_bound` / `upper_bound`) around the fluid, or shrink the geom."
                                )
                            # Furthest point sampling only spaces the samples evenly given enough candidates. Holes
                            # in the sampling lower the boundary density locally, and fluid leaks through them.
                            pos_local, normals_local = sample_mesh_point_cloud(
                                verts, faces, n_points, n_candidates=16 * n_points, seed=0, return_normals=True
                            )
                            areas_local = np.full(len(pos_local), area / n_points, dtype=gs.np_float)
                            if geom.type == gs.GEOM_TYPE.PLANE:
                                normals_local = np.tile(surface_normal, (len(pos_local), 1))
                            pos_local = pos_local - self._boundary_offset * normals_local
                            if geom.type != gs.GEOM_TYPE.PLANE and self._boundary_offset > 0.0:
                                edge_pos, edge_normals, edge_areas = sample_reentrant_edges(
                                    verts, faces, self._boundary_offset, self._boundary_spacing
                                )
                                pos_local = np.concatenate([pos_local, edge_pos])
                                normals_local = np.concatenate([normals_local, edge_normals])
                                areas_local = np.concatenate([areas_local, edge_areas]).astype(gs.np_float)

                            # A fixed link never moves, so the samples outside the boundary area can be dropped.
                            if link.is_fixed:
                                pos_world = gu.transform_by_trans_quat(pos_local, link_pos, link_quat)
                                is_inside = np.all((pos_world > area_lower) & (pos_world < area_upper), axis=1)
                                pos_local = pos_local[is_inside]
                                normals_local = normals_local[is_inside]
                                areas_local = areas_local[is_inside]

                            if geom.active_envs_mask is None:
                                active = np.ones((self._B, len(pos_local)), dtype=gs.np_int)
                            else:
                                envs_active = tensor_to_array(geom.active_envs_mask, dtype=gs.np_int)
                                active = np.tile(envs_active[:, None], (1, len(pos_local)))
                            boundaries_pos_local.append(pos_local)
                            boundaries_normal_local.append(normals_local)
                            # share of the surface of the geom each sample stands for
                            boundaries_area.append(areas_local)
                            boundaries_link_idx.append(np.full(len(pos_local), link.idx, dtype=gs.np_int))
                            boundaries_is_fixed.append(np.full(len(pos_local), link.is_fixed, dtype=gs.np_int))
                            boundaries_active.append(active)
            self._n_boundary_particles = sum(len(pos_local) for pos_local in boundaries_pos_local)
            self._has_boundary = self._n_boundary_particles > 0
            if self._has_boundary:
                boundaries_link_idx_all = np.concatenate(boundaries_link_idx)
                boundaries_is_fixed_all = np.concatenate(boundaries_is_fixed)
                self._coupled_links_idx = np.unique(boundaries_link_idx_all[boundaries_is_fixed_all == 0])
            self._has_rigid_coupling = len(self._coupled_links_idx) > 0
            self._has_viscosity = any(entity.material.viscosity > 0.0 for entity in self.entities)

            # Volume map of each link carrying boundary particles, over their bounding box widened by the support radius
            self._volume_maps = []
            if self._has_boundary and self._has_volume_maps:
                pos_local_all = np.concatenate(boundaries_pos_local)
                link_idx_all = np.concatenate(boundaries_link_idx)
                cell_size = self._volume_map_cell_size
                margin = self._support_radius + 2.0 * cell_size
                if self._volume_map_offset is None:
                    self._volume_map_offset = flush_lattice_volume_offset(self._particle_volume, self._support_radius)
                offset = self._volume_map_offset
                n_nodes_total = 0
                for i_l, link_signed_distance in links_signed_distance.items():
                    pos_link = pos_local_all[link_idx_all == i_l]
                    if len(pos_link) == 0:
                        continue
                    lower = pos_link.min(axis=0) - margin
                    n_nodes = np.ceil((pos_link.max(axis=0) + margin - lower) / cell_size).astype(np.int64) + 1
                    n_nodes_total += int(np.prod(n_nodes))
                    if n_nodes_total > self._max_n_volume_map_nodes:
                        gs.raise_exception(
                            f"The SPH volume maps require more than {self._max_n_volume_map_nodes} grid nodes. Please "
                            "tighten the SPH domain around the fluid, or raise `volume_map_cell_size`."
                        )
                    with gs.logger.timer(f"Computing the SPH volume map of link {i_l} on a grid of {n_nodes} nodes."):
                        volume = compute_volume_map(
                            lambda x, fs=link_signed_distance: np.min([f(x) for f in fs], axis=0) + offset,
                            lower,
                            cell_size,
                            n_nodes,
                            self._support_radius,
                        )
                        values = volume_map_to_distance(volume, self._support_radius)
                    self._volume_maps.append((i_l, lower, n_nodes, values))
            self._has_volume_maps = len(self._volume_maps) > 0
            if self._has_boundary:
                gs.logger.info(f"SPH boundary sampled with ~<{self._n_boundary_particles}>~ particles.")
                self.sh_boundary = gu.SpatialHasher(cell_size=self.sh.cell_size, grid_res=self.sh.grid_res)
                self.sh_boundary.build(self._B)

            self.init_particle_fields()
            self.init_ckpt()
            if self._has_rigid_coupling:
                links_coupled_idx = np.full(rigid_solver.n_links, -1, dtype=gs.np_int)
                links_coupled_idx[self._coupled_links_idx] = np.arange(len(self._coupled_links_idx), dtype=gs.np_int)
                pos_local_all = np.concatenate(boundaries_pos_local)
                coupled_links_extent = np.array(
                    [
                        np.linalg.norm(pos_local_all[boundaries_link_idx_all == i_l], axis=1).max()
                        for i_l in self._coupled_links_idx
                    ],
                    dtype=gs.np_float,
                )
                self._kernel_set_coupled_links(links_coupled_idx, self._coupled_links_idx, coupled_links_extent)

            if self._has_volume_maps:
                links_volume_map = np.full(rigid_solver.n_links, -1, dtype=gs.np_int)
                starts = np.cumsum([0] + [values.size for *_, values in self._volume_maps])[:-1]
                for i_vm, (i_l, *_) in enumerate(self._volume_maps):
                    links_volume_map[i_l] = i_vm
                self.links_volume_map.from_numpy(links_volume_map)
                self.volume_maps_values.from_numpy(np.concatenate([values.ravel() for *_, values in self._volume_maps]))
                self.volume_maps_lower.from_numpy(np.stack([m[1] for m in self._volume_maps]).astype(gs.np_float))
                self.volume_maps_n_nodes.from_numpy(np.stack([m[2] for m in self._volume_maps]).astype(gs.np_int))
                self.volume_maps_start.from_numpy(starts.astype(gs.np_int))

            for entity in self.entities:
                entity._add_to_solver()

            if self._has_boundary:
                self._kernel_add_boundary_particles(
                    np.concatenate(boundaries_pos_local),
                    np.concatenate(boundaries_normal_local),
                    np.concatenate(boundaries_area),
                    np.concatenate(boundaries_link_idx),
                    np.concatenate(boundaries_is_fixed),
                    np.concatenate(boundaries_active, axis=1),
                )
                self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
                self._kernel_compute_boundary_volume()

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity(as_field=True)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )

        self.entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles.active, self.particles.reordered_idx
        )

        # copy to reordered
        self.particles_reordered.active.fill(False)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                reordered_idx = self.particles[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                # only need to copy back dynamic state, i.e. self.particles
                self.particles[i_p, i_b] = self.particles_reordered[self.particles[i_p, i_b].reordered_idx, i_b]

    @qd.func
    def _task_compute_non_pressure_forces(self, i, j, ret: qd.template(), i_b):
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        mass_i = self.materials[i_m_i].mass
        mass_j = self.materials[i_m_j].mass

        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        dist = x_ij.norm()

        # Surface tension (cohesion). The kernel is evaluated no closer than one particle size so that the attraction
        # vanishes as particles touch instead of peaking at the kernel center.
        effective_dist = qd.max(dist, self._particle_size)
        ret.acc -= self.materials[i_m_i].surface_tension / mass_i * mass_j * x_ij * self.cubic_kernel(effective_dist)

        # The viscosity is implicit, see _kernel_init_viscosity_solve.
        if qd.static(self._micropolar):
            # The micro-rotation field only couples particles of the same material.
            if i_m_i == i_m_j:
                rho_j = self.particles_reordered[j, i_b].density_norm * self.materials[i_m_j].rest_density
                grad_W = self.cubic_kernel_derivative(x_ij)
                vorticity = self.materials[i_m_i].vorticity
                inverse_microinertia = self.materials[i_m_i].inverse_microinertia
                rho_i = self.particles_reordered[i, i_b].density_norm * self.materials[i_m_i].rest_density
                omega_ij = self.particles_reordered[i, i_b].angular_vel - self.particles_reordered[j, i_b].angular_vel
                vel_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel

                # Smoothing of the micro-rotation field between neighbors (XSPH-like angular viscosity).
                ret.angular_acc -= (
                    inverse_microinertia
                    * self.materials[i_m_i].angular_viscosity
                    / self._substep_dt
                    * (mass_j / rho_j)
                    * omega_ij
                    * self.cubic_kernel(dist)
                )
                # Exchange between the curl of the velocity field and the micro-rotation field.
                ret.acc += vorticity / rho_i * mass_j * omega_ij.cross(grad_W)
                ret.angular_acc += vorticity / rho_i * inverse_microinertia * mass_j * vel_ij.cross(grad_W)

    @qd.kernel
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: qd.f32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(acc=self._gravity[i_b], angular_acc=qd.Vector.zero(gs.qd_float, 3))
                self._func_for_all_fluid_neighbors(i_p, ret, self._task_compute_non_pressure_forces, i_b)

                for i_ff in qd.static(range(len(self._ffs))):
                    ret.acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = ret.acc

                if qd.static(self._micropolar):
                    # Decay of the micro-rotation towards zero, which balances the vorticity injected by the flow.
                    i_m = self.particles_reordered[i_p, i_b].material
                    ret.angular_acc -= (
                        2.0
                        * self.materials[i_m].inverse_microinertia
                        * self.materials[i_m].vorticity
                        * self.particles_reordered[i_p, i_b].angular_vel
                    )
                    self.particles_reordered[i_p, i_b].angular_acc = ret.angular_acc

        # Velocities and micro-rotations are integrated in a separate pass, since the accelerations above read the
        # velocities and micro-rotations of the neighbors.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc
                if qd.static(self._has_viscosity):
                    self.particles_viscosity[i_p, i_b].vel_star = self.particles_reordered[i_p, i_b].vel
                    self.particles_viscosity[i_p, i_b].pressure_vel = qd.Vector.zero(gs.qd_float, 3)
                if qd.static(self._micropolar):
                    self.particles_reordered[i_p, i_b].angular_vel += (
                        self._substep_dt * self.particles_reordered[i_p, i_b].angular_acc
                    )

    @qd.kernel
    def _kernel_kinetic_energy(self) -> float:
        """Return the kinetic energy of the fluid in J, summed over the environments (for SPHOptions.energy_audit)."""
        energy = gs.qd_float(0.0)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                energy += 0.5 * self.materials[i_m].mass * self.particles_reordered[i_p, i_b].vel.norm_sqr()
        return energy

    @qd.func
    def _task_drift_density_change(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        ret += self.materials[i_m].rest_volume * v_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.func
    def _task_drift_acc(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        kappa_sum = self.particles_drift[i, i_b].kappa + self.particles_drift[j, i_b].kappa
        ret -= kappa_sum * self.materials[i_m].rest_volume * self.cubic_kernel_derivative(x_ij)

    @qd.func
    def _task_drift_source(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        a_ij = self.particles_drift[i, i_b].acc - self.particles_drift[j, i_b].acc
        ret += self.materials[i_m].rest_volume * a_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_drift_init(self):
        """Predict the density of every particle at the end of the substep from its final velocity."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                change = gs.qd_float(0.0)
                self._func_for_all_fluid_neighbors(i_p, change, self._task_drift_density_change, i_b)
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                            change += vel_rel.dot(self.particles_boundary_slots[i_p, i_b, k].grad)
                self.particles_drift[i_p, i_b].density_adv = (
                    self.particles_reordered[i_p, i_b].density_norm + self._substep_dt * change
                )
                self.particles_drift[i_p, i_b].kappa = 0.0
                self.particles_drift[i_p, i_b].kappa_prev = 0.0
                self.particles_drift[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_drift_acc(self):
        """Compute the acceleration of the drift multipliers, the rigid surfaces held in place."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                acc = qd.Vector.zero(gs.qd_float, 3)
                self._func_for_all_fluid_neighbors(i_p, acc, self._task_drift_acc, i_b)
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            acc -= (
                                self.particles_drift[i_p, i_b].kappa * self.particles_boundary_slots[i_p, i_b, k].grad
                            )
                self.particles_drift[i_p, i_b].acc = acc

    @qd.kernel
    def _kernel_drift_iteration(self, omega: float) -> float:
        """Run one Jacobi iteration of the drift correction and return the worst average error over envs.

        omega is the weight of the Chebyshev semi-iterative acceleration (see _pressure_solve), 1 for a plain iteration.
        """
        dt2 = qd.static(self._substep_dt**2)
        for i_b in range(self._B):
            self.envs_drift_error[i_b] = 0.0
            self.envs_drift_count[i_b] = 0.0
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                source = gs.qd_float(0.0)
                self._func_for_all_fluid_neighbors(i_p, source, self._task_drift_source, i_b)
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            source += self.particles_drift[i_p, i_b].acc.dot(
                                self.particles_boundary_slots[i_p, i_b, k].grad
                            )
                residual = 1.0 - self.particles_drift[i_p, i_b].density_adv - dt2 * source
                factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                kappa_cur = self.particles_drift[i_p, i_b].kappa
                kappa = qd.max(kappa_cur - 0.5 * residual * factor / dt2, 0.0)
                if omega != 1.0:
                    kappa_prev = self.particles_drift[i_p, i_b].kappa_prev
                    kappa = qd.max(omega * (kappa - kappa_prev) + kappa_prev, 0.0)
                self.particles_drift[i_p, i_b].kappa_prev = kappa_cur
                self.particles_drift[i_p, i_b].kappa = kappa
                self.envs_drift_error[i_b] += -qd.min(residual, 0.0)
                self.envs_drift_count[i_b] += 1.0
        max_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            qd.atomic_max(max_error, self.envs_drift_error[i_b] / qd.max(self.envs_drift_count[i_b], 1.0))
        return max_error

    @qd.kernel
    def _kernel_drift_apply(self):
        """Move the particles by the drift correction, keeping the angular momentum with a rigid rotation.

        A displacement carries the velocity of a particle to a new position, which changes the angular momentum by
        sum_i m_i dx_i x v_i; a rigid rotation w x (x_i - c) about the center of mass c of the fluid, with
        I_c w = -sum_i m_i dx_i x v_i, cancels it. The linear momentum is unchanged.
        """
        for i_b in range(self._B):
            self.envs_drift_mass[i_b] = 0.0
            self.envs_drift_center[i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.envs_drift_inertia[i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.envs_drift_angular[i_b] = qd.Vector.zero(gs.qd_float, 3)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                disp = self._substep_dt**2 * self.particles_drift[i_p, i_b].acc
                self.particles_drift[i_p, i_b].disp = disp
                mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                self.envs_drift_mass[i_b] += mass
                self.envs_drift_center[i_b] += mass * (self.particles_reordered[i_p, i_b].pos + disp)
        for i_b in range(self._B):
            if self.envs_drift_mass[i_b] > 0.0:
                self.envs_drift_center[i_b] /= self.envs_drift_mass[i_b]
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                disp = self.particles_drift[i_p, i_b].disp
                arm = self.particles_reordered[i_p, i_b].pos + disp - self.envs_drift_center[i_b]
                self.envs_drift_angular[i_b] += mass * disp.cross(self.particles_reordered[i_p, i_b].vel)
                self.envs_drift_inertia[i_b] += mass * (
                    arm.norm_sqr() * qd.Matrix.identity(gs.qd_float, 3) - arm.outer_product(arm)
                )
        for i_b in range(self._B):
            omega = qd.Vector.zero(gs.qd_float, 3)
            if self.envs_drift_mass[i_b] > 0.0:
                inertia = self.envs_drift_inertia[i_b]
                inertia += 1e-6 * inertia.trace() * qd.Matrix.identity(gs.qd_float, 3)
                omega = -(inertia.inverse() @ self.envs_drift_angular[i_b])
            self.envs_drift_angular[i_b] = omega
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                disp = self.particles_drift[i_p, i_b].disp
                arm = self.particles_reordered[i_p, i_b].pos + disp - self.envs_drift_center[i_b]
                self.particles_reordered[i_p, i_b].vel += self.envs_drift_angular[i_b].cross(arm)
                self.particles_reordered[i_p, i_b].pos += disp

    def _drift_correction(self):
        """Correct the compression of the predicted positions by moving the particles, without changing velocities.

        With SPHOptions(density_correction="position"), the density solve of the coupled passes only removes the
        compression the velocity would cause over the substep, and carries the pressure (the hydrostatic pressure and
        the loads on the rigid links). The density error that remains (the error the solves leave and the second order
        of the advection) is then corrected by a density solve of the same form, h^2 times whose acceleration moves the
        particles, the rigid surfaces held in place. A correction of the positions changes neither the
        momentum nor the kinetic energy, so the density error no longer turns into particle velocity, the particle noise
        of water at rest. Its iterations stop at the tolerance of the density solve, within
        `max_drift_solver_iterations`.
        """
        self._kernel_drift_init()
        rho2 = self._df_chebyshev_radius**2
        n_iters, error, omega = 0, 0.0, 1.0
        while n_iters < self._max_drift_iters:
            if rho2 > 0.0 and n_iters > 0:
                omega = 2.0 / (2.0 - rho2) if n_iters == 1 else 4.0 / (4.0 - rho2 * omega)
            self._kernel_drift_acc()
            error = self._kernel_drift_iteration(omega)
            n_iters += 1
            if error <= self._df_max_error_den and n_iters >= self._df_min_den_iters:
                break
        self._kernel_drift_acc()
        self._kernel_drift_apply()
        residual = error / self._df_max_error_den
        self._solve_stats.n_drift_iters += n_iters
        self._solve_stats.n_unconverged_drift += residual > 1.0
        self._solve_stats.max_drift_residual = max(self._solve_stats.max_drift_residual, residual)

    @qd.kernel
    def _kernel_advect_position(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                new_vel = self.particles_reordered[i_p, i_b].vel

                new_pos = pos + self._substep_dt * new_vel

                # Clamp to the simulation domain.
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                self.particles_reordered[i_p, i_b].vel = corrected_vel
                self.particles_reordered[i_p, i_b].pos = corrected_pos

    # ------------------------------------------------------------------------------------
    # ------------------------------------ boundary --------------------------------------
    # ------------------------------------------------------------------------------------
    # Particle-based rigid boundaries (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for Incompressible SPH"),
    # entering the DFSPH solve as in Bender et al. 2023, "Consistent SPH Rigid-Fluid Coupling" (BWJ23). Boundary
    # particles only add constant contributions to the density constraints of the fluid particles, so they never
    # need a pressure value of their own.

    @qd.kernel
    def _kernel_add_boundary_particles(
        self,
        pos_local: qd.types.ndarray(),
        normal_local: qd.types.ndarray(),
        area: qd.types.ndarray(),
        link_idx: qd.types.ndarray(),
        is_fixed: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            for i in qd.static(range(3)):
                self.boundary_particles[i_bp, i_b].pos_local[i] = pos_local[i_bp, i]
                self.boundary_particles[i_bp, i_b].normal_local[i] = normal_local[i_bp, i]
            self.boundary_particles[i_bp, i_b].area = area[i_bp]
            self.boundary_particles[i_bp, i_b].link_idx = link_idx[i_bp]
            self.boundary_particles[i_bp, i_b].is_fixed = qd.cast(is_fixed[i_bp], gs.qd_bool)
            self.boundary_particles[i_bp, i_b].active = qd.cast(active[i_b, i_bp], gs.qd_bool)

    @qd.kernel
    def _kernel_update_boundary_particles(self, links_state: array_class.LinksState):
        for i_l, i_b in qd.ndrange(self.links_com.shape[0], self._B):
            self.links_com[i_l, i_b] = links_state.i_pos[i_l, i_b] + links_state.root_COM[i_l, i_b]
            self.links_root_com[i_l, i_b] = links_state.root_COM[i_l, i_b]
            self.links_vel[i_l, i_b] = links_state.cd_vel[i_l, i_b]
            self.links_ang[i_l, i_b] = links_state.cd_ang[i_l, i_b]
            if qd.static(self._has_volume_maps):
                self.links_pos[i_l, i_b] = links_state.pos[i_l, i_b]
                self.links_quat[i_l, i_b] = links_state.quat[i_l, i_b]

        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_l = self.boundary_particles[i_bp, i_b].link_idx
                pos = gu.qd_transform_by_trans_quat(
                    self.boundary_particles[i_bp, i_b].pos_local, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].pos = pos
                self.boundary_particles[i_bp, i_b].normal = gu.qd_transform_by_quat(
                    self.boundary_particles[i_bp, i_b].normal_local, links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].vel = self.sim.rigid_solver._func_vel_at_point(
                    i_l, i_b, pos, links_state
                )

        self.sh_boundary.compute_reordered_idx(
            self._n_boundary_particles,
            self.boundary_particles.pos,
            self.boundary_particles.active,
            self.boundary_particles.reordered_idx,
        )

        self.boundary_particles_reordered.active.fill(False)
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_r = self.boundary_particles[i_bp, i_b].reordered_idx
                self.boundary_particles_reordered[i_r, i_b] = self.boundary_particles[i_bp, i_b]

    @qd.func
    def _task_compute_boundary_volume(self, i, j, ret: qd.template(), i_b):
        # The sampling of a moving link is rigid, so only its own particles count and its volumes stay valid as it
        # moves. The fixed links never move with respect to each other, so they form a single boundary.
        is_same_boundary = self.boundary_particles_reordered[j, i_b].link_idx == ret.link_idx
        if ret.is_fixed and self.boundary_particles_reordered[j, i_b].is_fixed:
            is_same_boundary = True
        if is_same_boundary:
            x_ij = self.boundary_particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
            ret.sum_w += self.cubic_kernel(x_ij.norm())

    @qd.kernel
    def _kernel_compute_boundary_volume(self):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_r = self.boundary_particles[i_bp, i_b].reordered_idx
                ret = qd.Struct(
                    sum_w=self.cubic_kernel(0.0),
                    link_idx=self.boundary_particles[i_bp, i_b].link_idx,
                    is_fixed=self.boundary_particles[i_bp, i_b].is_fixed,
                )
                self.sh_boundary.for_all_neighbors(
                    i_r,
                    self.boundary_particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_boundary_volume,
                    i_b,
                )
                self.boundary_particles[i_bp, i_b].volume = 1.0 / ret.sum_w

    @qd.func
    def _func_find_boundary_slot(self, i, i_l, i_b):
        """Return the boundary slot of fluid particle i that holds link i_l, -1 if none."""
        i_s = -1
        for k in range(N_BOUNDARY_SLOTS):
            if i_s == -1 and self.particles_boundary_slots[i, i_b, k].link_idx == i_l:
                i_s = k
        return i_s

    @qd.func
    def _task_find_boundary_faces(self, i, j, ret: qd.template(), i_b):
        if ret.n_boundary < N_LISTED_BOUNDARY_NEIGHBORS:
            self.particles_boundary_neighbors[ret.n_boundary, i, i_b] = j
        ret.n_boundary += 1
        # Assign the link of the boundary particle a slot, the first unused one taking a new link.
        i_l = self.boundary_particles_reordered[j, i_b].link_idx
        i_s = -1
        for k in range(N_BOUNDARY_SLOTS):
            i_l_slot = self.particles_boundary_slots[i, i_b, k].link_idx
            if i_s == -1 and (i_l_slot == i_l or i_l_slot == -1):
                i_s = k
        if i_s == -1:
            ret.n_overflows += 1
        else:
            self.particles_boundary_slots[i, i_b, i_s].link_idx = i_l
            # Clearance from the surface of the link: the signed distance to the tangent plane of the nearest boundary
            # particle the fluid particle faces, those within a sampling spacing of it along their surface. A boundary
            # particle on another face of a convex edge does not face it, and its tangent plane, which the fluid
            # particle may lie behind while outside the geom, does not count. Of the faced ones, the nearest along the
            # normal sits on the near face of a thin wall, not on its far face.
            normal = self.boundary_particles_reordered[j, i_b].normal
            x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
            dist_normal = x_ib.dot(normal)
            dist_lateral2 = x_ib.norm_sqr() - dist_normal**2
            # the fluid nearest to the surface patch of the boundary particle, within half a particle spacing of its
            # normal (see _func_wall_ghost)
            if dist_normal > self._boundary_offset and dist_lateral2 < (0.5 * self._particle_size) ** 2:
                gap_units = (dist_normal - self._boundary_offset) * (FLUID_GAP_UNITS / self._support_radius)
                qd.atomic_min(
                    self.boundary_particles_fluid_gap[j, i_b], qd.cast(qd.min(gap_units, FLUID_GAP_UNITS), gs.qd_int)
                )
            if dist_lateral2 < (0.75 * self._boundary_spacing) ** 2 and qd.abs(dist_normal) < qd.abs(
                self.particles_boundary_slots[i, i_b, i_s].clearance + self._boundary_offset
            ):
                self.particles_boundary_slots[i, i_b, i_s].clearance = dist_normal - self._boundary_offset
                self.particles_boundary_slots[i, i_b, i_s].clearance_normal = normal

    @qd.func
    def _task_compute_boundary_density_and_factor(self, i, j, ret: qd.template(), i_b):
        ret.n_neighbors += 1
        i_l = self.boundary_particles_reordered[j, i_b].link_idx
        has_map = False
        if qd.static(self._has_volume_maps):
            has_map = self.links_volume_map[i_l] != -1
        if not has_map:
            V_b = self.boundary_particles_reordered[j, i_b].volume
            x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
            ret.density_norm += V_b * self.cubic_kernel(x_ib.norm())
            i_s = self._func_find_boundary_slot(i, i_l, i_b)
            if i_s != -1:
                # The full kernel gradient of the boundary particles is the gradient of the density they add, so the
                # pressure force is the gradient of the compression energy and does no work on the fluid at rest. A
                # wall force projected on the surface normal, which a smooth wall exerts, is not that gradient: the
                # density of a fluid particle sliding past the discrete samples varies, the pressure solve answers it
                # along the normal only, and water at rest heats up (SPH_SOLVER_STATUS.md, finding 2). The tangential
                # part of the full gradient acts as a roughness instead, which the sampling spacing of the boundary
                # sets. A volume map (see _func_volume_map) has neither defect.
                self.particles_boundary_slots[i, i_b, i_s].grad += V_b * self.cubic_kernel_derivative(x_ib)

    @qd.func
    def _func_half_space_volume(self, dist):
        """Return the boundary volume of a half-space at a signed distance from its surface (see half_space_volume)."""
        q = qd.min(qd.abs(dist) / self._support_radius, 1.0)
        tail = gs.qd_float(0.0)
        if q > 0.5:
            tail = 1.0 / 30.0 - (0.1 * q - q**3 / 3.0 + 0.5 * q**4 - 0.3 * q**5 + 0.4 / 6.0 * q**6)
        else:
            tail = 0.03125 - (0.0875 * q - q**3 / 6.0 + 0.3 * q**5 - 0.2 * q**6)
        volume = 16.0 * tail
        if dist < 0.0:
            volume = 1.0 - volume
        return volume

    @qd.func
    def _func_volume_map(self, i_vm, pos_local):
        """Return the boundary volume of volume map i_vm at a point in link frame, and its gradient there.

        The map holds at its nodes the distance d at which a half-space has the boundary volume of the solid (see
        volume_map_to_distance), the control points of a uniform cubic B-spline. The spline reproduces the linear
        distance to a planar surface exactly, whatever its orientation to the grid, and its error elsewhere (curved
        surfaces, edges) varies smoothly, where a trilinear interpolation ripples from cell to cell: a roughness of the
        wall, which slowed fluid in rigid rotation in a cup by 13 % per second at a cell of a quarter particle size
        (SPH_SOLVER_STATUS.md). The volume is that of the half-space at the interpolated distance, and its gradient is
        exact, so that the pressure force of the wall is the gradient of the density it adds. Zero outside the grid,
        which extends a support radius and two cells beyond the surface.
        """
        volume = gs.qd_float(0.0)
        grad = qd.Vector.zero(gs.qd_float, 3)
        n_nodes = self.volume_maps_n_nodes[i_vm]
        u = (pos_local - self.volume_maps_lower[i_vm]) / self._volume_map_cell_size
        is_inside = True
        for d in qd.static(range(3)):
            if u[d] < 1.0 or u[d] > n_nodes[d] - 2:
                is_inside = False
        if is_inside:
            # uniform cubic B-spline weights of the four nodes around the point along each axis, and their derivatives
            idx = qd.Vector.zero(gs.qd_int, 3)
            w = qd.Matrix.zero(gs.qd_float, 3, 4)
            dw = qd.Matrix.zero(gs.qd_float, 3, 4)
            for d in qd.static(range(3)):
                idx[d] = qd.min(qd.cast(qd.floor(u[d]), gs.qd_int), n_nodes[d] - 3)
                t = u[d] - idx[d]
                w[d, 0] = (1.0 - t) ** 3 / 6.0
                w[d, 1] = (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0
                w[d, 2] = (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0
                w[d, 3] = t**3 / 6.0
                dw[d, 0] = -0.5 * (1.0 - t) ** 2
                dw[d, 1] = 1.5 * t**2 - 2.0 * t
                dw[d, 2] = -1.5 * t**2 + t + 0.5
                dw[d, 3] = 0.5 * t**2
            start = self.volume_maps_start[i_vm]
            dist = gs.qd_float(0.0)
            grad_dist = qd.Vector.zero(gs.qd_float, 3)
            for a0 in qd.static(range(4)):
                for a1 in qd.static(range(4)):
                    row = start + ((idx[0] + a0 - 1) * n_nodes[1] + idx[1] + a1 - 1) * n_nodes[2] + idx[2] - 1
                    for a2 in qd.static(range(4)):
                        value = self.volume_maps_values[row + a2]
                        dist += w[0, a0] * w[1, a1] * w[2, a2] * value
                        grad_dist += value * qd.Vector(
                            [
                                dw[0, a0] * w[1, a1] * w[2, a2],
                                w[0, a0] * dw[1, a1] * w[2, a2],
                                w[0, a0] * w[1, a1] * dw[2, a2],
                            ]
                        )
            volume = self._func_half_space_volume(dist)
            # the volume decreases away from the solid at the rate of the plane integral of the kernel
            grad = -self.cubic_kernel_plane_integral(dist) / self._volume_map_cell_size * grad_dist
        return volume, grad

    @qd.func
    def _func_slot_wall_vel(self, i_p, i_b, k):
        """Return the rigid velocity of the link of boundary slot k of fluid particle i_p, at the fluid particle."""
        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
        return self._func_link_vel_at_pos(i_l, i_b, self.particles_reordered[i_p, i_b].pos)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    # See https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp for reference.

    @qd.func
    def _task_compute_density_and_factor(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        V_j = self.materials[i_m].rest_volume
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos

        ret.density_norm += V_j * self.cubic_kernel(x_ij.norm())
        grad_p_j = -V_j * self.cubic_kernel_derivative(x_ij)
        ret.sum_grad_p_k += grad_p_j.norm_sqr()
        ret.grad_p_i -= grad_p_j
        if ret.n_neighbors < N_LISTED_NEIGHBORS:
            self.particles_fluid_neighbors[ret.n_neighbors, i, i_b] = j
        ret.n_neighbors += 1

    @qd.func
    def _func_for_all_fluid_neighbors(self, i_p, ret: qd.template(), task: qd.template(), i_b):
        """Run task(i_p, j, ret, i_b) on every fluid neighbor j of particle i_p within the support radius.

        Reads the list of the density kernel of the substep (_kernel_compute_density_and_factor), which holds the same
        neighbors as a search of the hash grid as long as the positions have not changed since, and searches the grid
        for a particle with more neighbors than a list holds.
        """
        n_neighbors = self.particles_n_fluid_neighbors[i_p, i_b]
        if n_neighbors <= N_LISTED_NEIGHBORS:
            for k in range(n_neighbors):
                task(i_p, self.particles_fluid_neighbors[k, i_p, i_b], ret, i_b)
        else:
            self.sh.for_all_neighbors(i_p, self.particles_reordered.pos, self._support_radius, ret, task, i_b)

    @qd.func
    def _func_for_all_boundary_neighbors(self, i_p, ret: qd.template(), task: qd.template(), i_b):
        """Run task(i_p, j, ret, i_b) on every boundary particle j within the support radius of fluid particle i_p.

        Reads the list of the density kernel of the substep (_kernel_compute_density_and_factor), as
        _func_for_all_fluid_neighbors does, and searches the boundary hash grid for a particle with more boundary
        neighbors than a list holds. A search of the grid scans every boundary particle hashed to the 27 cells around
        the particle, those outside the grid of the domain included, since the slots of the grid repeat in space.
        """
        n_neighbors = self.particles_n_boundary_neighbors[i_p, i_b]
        if n_neighbors <= N_LISTED_BOUNDARY_NEIGHBORS:
            for k in range(n_neighbors):
                task(i_p, self.particles_boundary_neighbors[k, i_p, i_b], ret, i_b)
        else:
            self.sh_boundary.for_all_neighbors_of_pos(
                i_p,
                self.particles_reordered[i_p, i_b].pos,
                -1,
                self.boundary_particles_reordered.pos,
                self._support_radius,
                ret,
                task,
                i_b,
            )

    @qd.func
    def _func_for_boundary_neighbor_share(self, i_p, i_s, ret: qd.template(), task: qd.template(), i_b):
        """Run task(i_p, j, ret, i_b) on share i_s of the boundary neighbors of fluid particle i_p, as
        _func_for_fluid_neighbor_share does for the fluid neighbors."""
        n_neighbors = self.particles_n_boundary_neighbors[i_p, i_b]
        if n_neighbors <= N_LISTED_BOUNDARY_NEIGHBORS:
            for m in range((n_neighbors - i_s + N_NEIGHBOR_SHARES - 1) // N_NEIGHBOR_SHARES):
                task(i_p, self.particles_boundary_neighbors[i_s + m * N_NEIGHBOR_SHARES, i_p, i_b], ret, i_b)
        elif i_s == 0:
            self.sh_boundary.for_all_neighbors_of_pos(
                i_p,
                self.particles_reordered[i_p, i_b].pos,
                -1,
                self.boundary_particles_reordered.pos,
                self._support_radius,
                ret,
                task,
                i_b,
            )

    @qd.func
    def _func_for_fluid_neighbor_share(self, i_p, i_s, ret: qd.template(), task: qd.template(), i_b):
        """Run task(i_p, j, ret, i_b) on share i_s of the fluid neighbors of particle i_p: the listed neighbors
        i_s, i_s + N_NEIGHBOR_SHARES, ..., or for a particle with more neighbors than a list holds, all of them (from
        the hash grid) in share 0 and none in the others."""
        n_neighbors = self.particles_n_fluid_neighbors[i_p, i_b]
        if n_neighbors <= N_LISTED_NEIGHBORS:
            for m in range((n_neighbors - i_s + N_NEIGHBOR_SHARES - 1) // N_NEIGHBOR_SHARES):
                task(i_p, self.particles_fluid_neighbors[i_s + m * N_NEIGHBOR_SHARES, i_p, i_b], ret, i_b)
        elif i_s == 0:
            self.sh.for_all_neighbors(i_p, self.particles_reordered.pos, self._support_radius, ret, task, i_b)

    @qd.kernel
    def _kernel_compute_density_and_factor(self, f: qd.i32):
        for i_b in range(self._B):
            self.envs_n_active[i_b] = 0

        if qd.static(self._has_boundary):
            for i_b in range(self._B):
                self.envs_boundary_slot_overflows[i_b] = 0
            for i_j, i_b in qd.ndrange(self._n_boundary_particles, self._B):
                self.boundary_particles_fluid_gap[i_j, i_b] = FLUID_GAP_UNITS

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                ret = qd.Struct(
                    density_norm=self.materials[i_m].rest_volume * self.cubic_kernel(0.0),
                    sum_grad_p_k=gs.qd_float(0.0),
                    grad_p_i=qd.Vector.zero(gs.qd_float, 3),
                    n_neighbors=0,
                    n_overflows=0,
                    n_boundary=0,
                )
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_and_factor,
                    i_b,
                )
                self.particles_n_fluid_neighbors[i_p, i_b] = ret.n_neighbors

                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        self.particles_boundary_slots[i_p, i_b, k].link_idx = -1
                        self.particles_boundary_slots[i_p, i_b, k].grad = qd.Vector.zero(gs.qd_float, 3)
                        self.particles_boundary_slots[i_p, i_b, k].clearance = self._support_radius
                        self.particles_boundary_slots[i_p, i_b, k].clearance_impulse = qd.Vector.zero(gs.qd_float, 3)
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        ret,
                        self._task_find_boundary_faces,
                        i_b,
                    )
                    self.particles_n_boundary_neighbors[i_p, i_b] = ret.n_boundary
                    self._func_for_all_boundary_neighbors(i_p, ret, self._task_compute_boundary_density_and_factor, i_b)
                    if ret.n_overflows > 0:
                        self.envs_boundary_slot_overflows[i_b] += 1
                    if qd.static(self._has_volume_maps):
                        for k in range(N_BOUNDARY_SLOTS):
                            i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                            if i_l != -1:
                                i_vm = self.links_volume_map[i_l]
                                if i_vm != -1:
                                    quat = self.links_quat[i_l, i_b]
                                    pos_local = gu.qd_inv_transform_by_trans_quat(
                                        self.particles_reordered[i_p, i_b].pos, self.links_pos[i_l, i_b], quat
                                    )
                                    volume, grad_local = self._func_volume_map(i_vm, pos_local)
                                    ret.density_norm += volume
                                    self.particles_boundary_slots[i_p, i_b, k].grad = gu.qd_transform_by_quat(
                                        grad_local, quat
                                    )
                    # BWJ23 eq. (32): boundary particles carry no constraint of their own, so they only enter the
                    # gradient of the constraint of particle i, and not the sum of squared neighbor gradients.
                    for k in range(N_BOUNDARY_SLOTS):
                        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                        if i_l != -1:
                            grad = self.particles_boundary_slots[i_p, i_b, k].grad
                            ret.grad_p_i += grad
                            if qd.static(self._has_rigid_coupling):
                                # The pressure of the particle also moves the coupled link it pushes, which feeds back
                                # on its own constraint through the mobility of the link (BWJ23 for dynamic bodies).
                                i_c = self.links_coupled_idx[i_l]
                                if i_c != -1:
                                    torque = (
                                        self.particles_reordered[i_p, i_b].pos - self.links_root_com[i_l, i_b]
                                    ).cross(grad)
                                    wrench = qd.Vector([torque[0], torque[1], torque[2], grad[0], grad[1], grad[2]])
                                    response = gs.qd_float(0.0)
                                    for a in qd.static(range(6)):
                                        for b in qd.static(range(6)):
                                            response += (
                                                wrench[a]
                                                * self.coupled_mobility[6 * i_c + a, 6 * i_c + b, i_b]
                                                * wrench[b]
                                            )
                                    ret.sum_grad_p_k += self.materials[i_m].mass * response

                sum_grad_p_k = ret.sum_grad_p_k + ret.grad_p_i.norm_sqr()
                factor = gs.qd_float(0.0)
                if sum_grad_p_k > self._df_eps:
                    factor = 1.0 / sum_grad_p_k

                self.particles_reordered[i_p, i_b].density_norm = ret.density_norm
                self.particles_reordered[i_p, i_b].alpha_factor_adj = factor
                self.particles_reordered[i_p, i_b].n_neighbors = ret.n_neighbors
                self.envs_n_active[i_b] += 1

    @qd.func
    def _func_add_pressure_moment(self, x, weight, ret: qd.template()):
        """Add a neighbor at x with the second moment weight to the moment equations of the pressure correction."""
        basis = self._func_viscosity_basis(x / x.norm())
        for a in qd.static(range(5)):
            ret.rhs[a] -= weight * basis[a + 1]
            for b in qd.static(range(5)):
                ret.gram[a, b] += weight * basis[a + 1] * basis[b + 1]
        ret.trace += weight

    @qd.func
    def _task_compute_pressure_moment(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        r2 = x_ij.norm_sqr()
        if r2 > self._df_eps * self._support_radius**2:
            i_m = self.particles_reordered[j, i_b].material
            weight = self.materials[i_m].rest_volume * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
            self._func_add_pressure_moment(x_ij, weight, ret)

    @qd.func
    def _task_compute_boundary_pressure_moment(self, i, j, ret: qd.template(), i_b):
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        r2 = x_ib.norm_sqr()
        if r2 > self._df_eps * self._support_radius**2:
            weight = self.boundary_particles_reordered[j, i_b].volume * (-self.cubic_kernel_derivative(x_ib).dot(x_ib))
            self._func_add_pressure_moment(x_ib, weight, ret)

    @qd.kernel
    def _kernel_compute_pressure_correction(self):
        """Make the pressure pair forces of every particle carry an isotropic stress.

        The symmetric pair form of the pressure force, -m_i V_j (kappa_i + kappa_j) grad W_ij, gives each particle the
        virial sum_j V_j (kappa_i + kappa_j) (-W'(r_ij) r_ij) e_ij e_ij^T, whose traceless part is the shear stress the
        pressure carries. A shear flow under pressure distorts the arrangement of the neighbors, and the pressure then
        carries a shear stress proportional to its value, a friction that is not in the equations: in a Taylor-Couette
        flow at 300 Pa the pressure pairs carried twice the viscous stress. The pair weights are therefore multiplied by
        the mean c_ij = 1 + (s_i(e_ij) + s_j(e_ij)) / 2 of directional corrections s_i in the five degree 2 spherical
        harmonics, solved per particle so that the second moment sum_j w_ij (1 + s_i(e_ij)) e_ij e_ij^T, w_ij =
        V_j (-W'(r_ij) r_ij), boundary particles included, is isotropic (Tikhonov term of a thousandth of the mean
        diagonal). The weight is symmetric and the force stays along e_ij, so the pairs keep conserving linear and
        angular momentum, and the corrected second moment also makes the pressure gradient consistent to first order.
        The weight is bounded to [0.7, 1.3]: the bound only acts at edges and corners of rigid geoms, where large
        corrections amplify the particle noise. The velocity constraint of the pressure solves (the density change rate
        and the predicted source) takes the same weights, so that the pressure velocity change stays the orthogonal
        projection in the norm of the masses, which never adds kinetic energy; with the weights in the force alone, the
        projection is oblique and heats a fluid at rest.
        At the free surface (normalized density from 0.95 down to 0.85) the correction fades out, since there the
        anisotropy of the half neighborhood is that of the surface.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(
                    gram=qd.Matrix.zero(gs.qd_float, 5, 5),
                    rhs=qd.Vector.zero(gs.qd_float, 5),
                    trace=gs.qd_float(0.0),
                )
                self._func_for_all_fluid_neighbors(i_p, ret, self._task_compute_pressure_moment, i_b)
                if qd.static(self._has_boundary):
                    self._func_for_all_boundary_neighbors(i_p, ret, self._task_compute_boundary_pressure_moment, i_b)
                correction = qd.Vector.zero(gs.qd_float, 5)
                blend = qd.min(qd.max((self.particles_reordered[i_p, i_b].density_norm - 0.85) / 0.1, 0.0), 1.0)
                if blend > 0.0 and ret.trace > self._df_eps:
                    gram = ret.gram + 1e-3 * ret.gram.trace() / 5.0 * qd.Matrix.identity(gs.qd_float, 5)
                    correction = blend * self._func_solve_spd5(gram, ret.rhs)
                # relaxation in time of the correction toward its value for the current arrangement
                previous = self.particles_reordered[i_p, i_b].pressure_correction
                self.particles_reordered[i_p, i_b].pressure_correction = previous + self._isotropic_relaxation * (
                    correction - previous
                )

    @qd.func
    def _task_compute_corrected_density(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        weight = self._func_pressure_pair_weight(i, j, x_ij, i_b)
        ret += (weight - 1.0) * self.materials[i_m].rest_volume * self.cubic_kernel(x_ij.norm())

    @qd.kernel
    def _kernel_compute_corrected_density(self):
        """Weigh the fluid pairs of the density estimate by the isotropic pressure correction.

        The pressure solves constrain the density whose gradient their velocity constraint and pressure force use, the
        pair weights c_ij included (see _kernel_compute_pressure_correction), so that the density solve corrects the
        density it measures: with the plain density, the pressure moves the particles along directions that do not
        reduce its error as the solve predicts, and the solve keeps answering an error it does not see, which heats a
        fluid at rest. The weights are frozen over the substep, as the kernel gradients are.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                change = gs.qd_float(0.0)
                self._func_for_all_fluid_neighbors(i_p, change, self._task_compute_corrected_density, i_b)
                self.particles_reordered[i_p, i_b].density_norm += change

    @qd.func
    def _func_solve_spd5(self, mat, rhs):
        """Solve the symmetric positive definite system mat x = rhs of size 5 by Cholesky factorization."""
        lower = qd.Matrix.zero(gs.qd_float, 5, 5)
        for c in qd.static(range(5)):
            diag = mat[c, c]
            for k in qd.static(range(c)):
                diag -= lower[c, k] * lower[c, k]
            diag = qd.sqrt(qd.max(diag, gs.EPS))
            lower[c, c] = diag
            for r in qd.static(range(c + 1, 5)):
                value = mat[r, c]
                for k in qd.static(range(c)):
                    value -= lower[r, k] * lower[c, k]
                lower[r, c] = value / diag
        sol = qd.Vector.zero(gs.qd_float, 5)
        for r in qd.static(range(5)):
            value = rhs[r]
            for k in qd.static(range(r)):
                value -= lower[r, k] * sol[k]
            sol[r] = value / lower[r, r]
        for r_ in qd.static(range(5)):
            r = qd.static(4 - r_)
            value = sol[r]
            for k in qd.static(range(r + 1, 5)):
                value -= lower[k, r] * sol[k]
            sol[r] = value / lower[r, r]
        return sol

    @qd.func
    def _func_pressure_pair_weight(self, i, j, x_ij, i_b):
        """Return the weight c_ij of the fluid pair (i, j) in the pressure force and in the velocity constraint of the
        pressure solves (see _kernel_compute_pressure_correction), 1 without the isotropic pressure correction."""
        weight = gs.qd_float(1.0)
        if qd.static(self._isotropic_pressure):
            r = x_ij.norm()
            if r > 0.0:
                basis = self._func_viscosity_basis(x_ij / r)
                correction = (
                    self.particles_reordered[i, i_b].pressure_correction
                    + self.particles_reordered[j, i_b].pressure_correction
                )
                for k in qd.static(range(5)):
                    weight += 0.5 * correction[k] * basis[k + 1]
                weight = qd.min(qd.max(weight, 0.7), 1.3)
        return weight

    @qd.func
    def _func_pressure_density2_norm(self, i_p, i_b, is_divergence):
        res = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
        if is_divergence:
            res = self.particles_reordered[i_p, i_b].pressure_density2_norm_d
        return res

    @qd.func
    def _task_compute_pressure_acc(self, i, j, ret: qd.template(), i_b):
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        density_ratio = self.materials[i_m_j].rest_density / self.materials[i_m_i].rest_density
        p_sum = ret.p_rho2_i + density_ratio * self._func_pressure_density2_norm(j, i_b, ret.is_divergence)
        if qd.abs(p_sum) > self._df_eps:
            x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
            weight = self._func_pressure_pair_weight(i, j, x_ij, i_b)
            ret.acc -= weight * p_sum * self.materials[i_m_j].rest_volume * self.cubic_kernel_derivative(x_ij)

    @qd.func
    def _func_compute_pressure_acc(self, i_p, i_b, is_divergence: qd.template()):
        ret = qd.Struct(
            acc=qd.Vector.zero(gs.qd_float, 3),
            p_rho2_i=self._func_pressure_density2_norm(i_p, i_b, is_divergence),
            is_divergence=is_divergence,
        )
        self._func_for_all_fluid_neighbors(i_p, ret, self._task_compute_pressure_acc, i_b)

        if qd.static(self._has_boundary):
            # BWJ23 eq. (33): a single term of the fluid particle pressure, with no pressure value at the boundary.
            for k in range(N_BOUNDARY_SLOTS):
                if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                    ret.acc -= ret.p_rho2_i * self.particles_boundary_slots[i_p, i_b, k].grad

        self.particles_reordered[i_p, i_b].pressure_acc = ret.acc

    @qd.func
    def _task_compute_density_change(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        weight = self._func_pressure_pair_weight(i, j, x_ij, i_b)
        ret += weight * self.materials[i_m].rest_volume * v_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_init_pressure_solve(self, is_divergence: qd.template(), keep_pressure: bool):
        """Compute the source term of the divergence or density solve and initialize its pressure.

        keep_pressure starts the density solve from the pressure it holds, which a later coupling pass of a substep does
        (SPH_SOLVER_DESIGN.md section 5.2).
        """
        inv_dt = 1.0 / self._substep_dt
        for i_b in range(self._B):
            self.envs_solve_active[i_b] = self.envs_coupling_active[i_b]
        if qd.static(self._has_rigid_coupling):
            for i_r, i_b in qd.ndrange(qd.static(self._n_wrench_rows), self._B):
                if self.envs_coupling_active[i_b]:
                    self.coupled_wrench[i_r, i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    density_change = gs.qd_float(0.0)
                    self._func_for_all_fluid_neighbors(i_p, density_change, self._task_compute_density_change, i_b)

                    if qd.static(self._has_boundary):
                        for k in range(N_BOUNDARY_SLOTS):
                            if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                                vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                                density_change += vel_rel.dot(self.particles_boundary_slots[i_p, i_b, k].grad)

                    factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                    if qd.static(is_divergence):
                        self.particles_reordered[i_p, i_b].density_adv_norm = density_change

                        # Only compression is corrected, and never at the free surface. This clamped source only
                        # seeds the solve: the iterations use the raw density change rate.
                        source = qd.max(density_change, 0.0)
                        if self.particles_reordered[i_p, i_b].n_neighbors < self._df_min_div_neighbors:
                            source = 0.0

                        p_rho2 = source * factor * inv_dt
                        if qd.static(self._df_warm_start):
                            p_rho2 = 0.0
                            if source > 0.0:
                                p_rho2 = 0.5 * qd.min(
                                    self.particles_reordered[i_p, i_b].pressure_density2_norm_d, 0.5 * inv_dt
                                )
                        self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                    else:
                        density_now = self.particles_reordered[i_p, i_b].density_norm
                        if qd.static(self._position_drift):
                            # The positions take the correction of the compression error (see _drift_correction), so the
                            # density solve of the coupled passes constrains the velocity alone: the predicted
                            # compression from the rest density, or from the density of a particle below it (free
                            # surface).
                            density_now = qd.min(density_now, 1.0)
                        density_adv = density_now + self._substep_dt * density_change
                        self.particles_reordered[i_p, i_b].density_adv_norm = density_adv

                        p_rho2 = -qd.min(1.0 - density_adv, 0.0) * factor * inv_dt * inv_dt
                        if qd.static(self._df_warm_start):
                            p_rho2 = 0.0
                            if density_adv > 1.0:
                                # The clamp bounds p * dt^2, which is what limits the warm-started velocity change.
                                p_rho2 = 0.5 * qd.min(
                                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p,
                                    0.00025 * inv_dt * inv_dt,
                                )
                        if keep_pressure:
                            p_rho2 = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                        self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2

    @qd.func
    def _task_compute_predicted_source(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        a_ij = self.particles_reordered[i, i_b].pressure_acc - self.particles_reordered[j, i_b].pressure_acc
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        weight = self._func_pressure_pair_weight(i, j, x_ij, i_b)
        ret += weight * self.materials[i_m].rest_volume * a_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_pressure_solve_iteration(self, is_divergence: qd.template(), omega: float):
        """
        Run one Jacobi iteration of the divergence or density solve; _kernel_pressure_solve_error then returns the
        worst average error of that iteration over envs.

        omega is the weight of the Chebyshev semi-iterative acceleration of the iteration (see _pressure_solve), 1 for a
        plain Jacobi iteration.

        The error is the average density deviation (normalized by the rest density) for the density solve, and the
        average normalized density change rate for the divergence solve.
        """
        # dt for the divergence solve, dt^2 for the density solve. Compile-time constants, as statements outside the loops
        # of a kernel run as serial tasks of their own.
        dt_pow = qd.static(self._substep_dt if is_divergence else self._substep_dt**2)

        # With 19172 particles a loop over the particles runs as a single wave of threads on a large GPU, and lasts as
        # long as the slowest thread walks its neighbors. The two neighbor loops of the iteration therefore split the
        # neighbors of each particle among N_NEIGHBOR_SHARES threads, whose partial sums a loop over the particles adds
        # in a fixed order, so that the result does not depend on the order of the threads. The error of the
        # environment is summed from the errors of the particles only when the solve reads it (see
        # _kernel_pressure_solve_error). The wrenches of the coupled links are reset at the start of the solve (see
        # _kernel_init_pressure_solve) and at the end of every iteration, after the response of the links has read them.
        for i_s, i_p, i_b in qd.ndrange(N_NEIGHBOR_SHARES, self._n_particles, self._B):
            if self.envs_solve_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    ret = qd.Struct(
                        acc=qd.Vector.zero(gs.qd_float, 3),
                        p_rho2_i=self._func_pressure_density2_norm(i_p, i_b, is_divergence),
                        is_divergence=is_divergence,
                    )
                    self._func_for_fluid_neighbor_share(i_p, i_s, ret, self._task_compute_pressure_acc, i_b)
                    if i_s == 0:
                        if qd.static(self._has_boundary):
                            # BWJ23 eq. (33): a single term of the fluid particle pressure, with no pressure value at
                            # the boundary.
                            for k in range(N_BOUNDARY_SLOTS):
                                if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                                    ret.acc -= ret.p_rho2_i * self.particles_boundary_slots[i_p, i_b, k].grad
                        if qd.static(self._has_rigid_coupling):
                            # Reaction of the boundary pressure of the iteration on the coupled links
                            # (SPH_SOLVER_DESIGN.md section 5.1)
                            i_m = self.particles_reordered[i_p, i_b].material
                            mass_kappa = self.materials[i_m].mass * ret.p_rho2_i
                            self._func_add_coupled_wrench(i_p, i_b, mass_kappa)
                    self.particles_acc_shares[i_s, i_p, i_b] = ret.acc

        # The response of the coupled links to that reaction runs in the first threads of the loop that adds the
        # shares, which also follows the loop of the reaction, instead of in a loop of its own (about 7 us per
        # iteration).
        for i_p, i_b in qd.ndrange(qd.static(max(self._n_particles, self._n_wrench_rows)), self._B):
            if self.envs_solve_active[i_b]:
                if qd.static(self._has_rigid_coupling):
                    if i_p < qd.static(self._n_wrench_rows):
                        acc_row = gs.qd_float(0.0)
                        for i_col in range(qd.static(self._n_wrench_rows)):
                            acc_row += self.coupled_mobility[i_p, i_col, i_b] * self.coupled_wrench[i_col, i_b]
                        self.coupled_acc[i_p, i_b] = acc_row
                if i_p < self._n_particles:
                    if self.particles_reordered[i_p, i_b].active:
                        acc = qd.Vector.zero(gs.qd_float, 3)
                        for i_s in qd.static(range(N_NEIGHBOR_SHARES)):
                            acc += self.particles_acc_shares[i_s, i_p, i_b]
                        self.particles_reordered[i_p, i_b].pressure_acc = acc

        for i_s, i_p, i_b in qd.ndrange(N_NEIGHBOR_SHARES, self._n_particles, self._B):
            if self.envs_solve_active[i_b]:
                if qd.static(self._has_rigid_coupling):
                    if i_s == 0 and i_p == 0:
                        for i_r in range(qd.static(self._n_wrench_rows)):
                            self.coupled_wrench[i_r, i_b] = 0.0
                if self.particles_reordered[i_p, i_b].active:
                    predicted_source = gs.qd_float(0.0)
                    self._func_for_fluid_neighbor_share(
                        i_p, i_s, predicted_source, self._task_compute_predicted_source, i_b
                    )
                    if i_s == 0:
                        if qd.static(self._has_boundary):
                            # BWJ23 eq. (34): the pressure acceleration of a boundary particle is the one of its link,
                            # zero for a link that is not coupled.
                            for k in range(N_BOUNDARY_SLOTS):
                                if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                                    acc_rel = self.particles_reordered[i_p, i_b].pressure_acc
                                    if qd.static(self._has_rigid_coupling):
                                        acc_rel -= self._func_coupled_point_acc(i_p, i_b, k)
                                    predicted_source += acc_rel.dot(self.particles_boundary_slots[i_p, i_b, k].grad)
                    self.particles_source_shares[i_s, i_p, i_b] = predicted_source

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_solve_active[i_b]:
                error = gs.qd_float(0.0)
                if self.particles_reordered[i_p, i_b].active:
                    predicted_source = gs.qd_float(0.0)
                    for i_s in qd.static(range(N_NEIGHBOR_SHARES)):
                        predicted_source += self.particles_source_shares[i_s, i_p, i_b]
                    self.particles_reordered[i_p, i_b].predicted_source = predicted_source

                    source = -self.particles_reordered[i_p, i_b].density_adv_norm
                    if qd.static(not is_divergence):
                        source = 1.0 + source
                    residual = source - dt_pow * predicted_source
                    error = -qd.min(residual, 0.0)

                    p_rho2_cur = self._func_pressure_density2_norm(i_p, i_b, is_divergence)
                    factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                    p_rho2 = qd.max(p_rho2_cur - 0.5 * residual * factor / dt_pow, 0.0)
                    if omega != 1.0:
                        p_rho2_prev = self.particles_pressure_prev[i_p, i_b]
                        p_rho2 = qd.max(omega * (p_rho2 - p_rho2_prev) + p_rho2_prev, 0.0)
                    self.particles_pressure_prev[i_p, i_b] = p_rho2_cur
                    if qd.static(is_divergence):
                        self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                        if self.particles_reordered[i_p, i_b].n_neighbors < self._df_min_div_neighbors:
                            error = 0.0
                    else:
                        self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2
                self.particles_solver_error[i_p, i_b] = error

    @qd.func
    def _func_chunk_sum(self, i_c, i_b):
        """Return the sum of particles_reduction over chunk i_c of REDUCTION_CHUNK particles of env i_b, and the largest
        value of its last component.

        The loops of the viscosity solve, the coupling residual and the dissipation readout store the terms of every
        particle (zero for an inactive one), and a loop over the chunks adds each chunk with one atomic addition per
        environment sum: an atomic addition per particle on a single address serializes (finding 62 of
        SPH_SOLVER_STATUS.md for the error of the pressure iterations).
        """
        total = qd.Vector.zero(gs.qd_float, 4)
        largest = gs.qd_float(0.0)
        for k in range(REDUCTION_CHUNK):
            i_p = i_c * REDUCTION_CHUNK + k
            if i_p < self._n_particles:
                terms = self.particles_reduction[i_p, i_b]
                total += terms
                largest = qd.max(largest, terms[3])
        return total, largest

    @qd.kernel
    def _kernel_pressure_solve_error(
        self, max_error: float, is_first: bool, reduce_error: bool, may_stop: bool
    ) -> float:
        """Stop the environments whose error meets their target, and return the worst error over the environments that
        were still iterating relative to their target, the solve having converged once it is at most 1.

        The target of an environment is max_error, or with reduce_error a tenth of the error of its first iteration
        within [max_error / 1000, max_error] (see _pressure_solve); is_first marks the first iteration, and may_stop
        whether the minimum number of iterations is reached. The errors of the particles are summed in chunks of
        REDUCTION_CHUNK particles, one atomic addition per chunk to the error of the environment: an atomic addition per
        particle, in every iteration, contended on that single address for 39 us per iteration of 19172 particles
        (SPH_SOLVER_STATUS.md finding 62). A stopped environment keeps the error of its last iteration.
        """
        for i_b in range(self._B):
            if self.envs_solve_active[i_b]:
                self.envs_solver_error[i_b] = 0.0
        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_solve_active[i_b]:
                chunk_error = gs.qd_float(0.0)
                for k in range(REDUCTION_CHUNK):
                    i_p = i_c * REDUCTION_CHUNK + k
                    if i_p < self._n_particles:
                        chunk_error += self.particles_solver_error[i_p, i_b]
                if chunk_error > 0.0:
                    self.envs_solver_error[i_b] += chunk_error

        max_rel_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_solve_active[i_b]:
                avg_error = self.envs_solver_error[i_b] / qd.max(self.envs_n_active[i_b], 1)
                if is_first:
                    target = max_error
                    if reduce_error:
                        target = qd.max(qd.min(max_error, 0.1 * avg_error), 1e-3 * max_error)
                    self.envs_solve_target[i_b] = target
                rel_error = avg_error / self.envs_solve_target[i_b]
                if may_stop and rel_error <= 1.0:
                    self.envs_solve_active[i_b] = 0
                qd.atomic_max(max_rel_error, rel_error)
        return max_rel_error

    @qd.kernel
    def _kernel_pressure_solve_max_error(self) -> float:
        """Return the worst average error over the environments of the solve that just ended."""
        max_avg_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                qd.atomic_max(max_avg_error, self.envs_solver_error[i_b] / qd.max(self.envs_n_active[i_b], 1))
        return max_avg_error

    @qd.kernel
    def _kernel_activate_envs(self):
        """Let every environment take part in the solves of the fluid stage that starts."""
        for i_b in range(self._B):
            self.envs_coupling_active[i_b] = 1
            self.envs_solve_active[i_b] = 1

    @qd.kernel
    def _kernel_update_coupling_active(self):
        """Stop the coupling passes of the environments whose own coupling residual, of the fluid and of the coupled
        links (see _kernel_coupling_residual and _kernel_solve_coupled_viscosity), is at most 1."""
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                rel_residual = self.envs_coupling_rel[i_b]
                if qd.static(self._has_rigid_coupling):
                    rel_residual = qd.max(rel_residual, self.envs_coupled_vel_change[i_b] / self._visc_max_error)
                if rel_residual <= 1.0:
                    self.envs_coupling_active[i_b] = 0

    @qd.kernel
    def _kernel_apply_pressure_acc(self, is_divergence: qd.template()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    self._func_compute_pressure_acc(i_p, i_b, is_divergence)
                    pressure_vel = self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc
                    self.particles_reordered[i_p, i_b].vel += pressure_vel
                    if qd.static(self._has_viscosity and not is_divergence):
                        self.particles_viscosity[i_p, i_b].pressure_vel_delta = (
                            pressure_vel - self.particles_viscosity[i_p, i_b].pressure_vel
                        )
                        self.particles_viscosity[i_p, i_b].pressure_vel = pressure_vel

        if qd.static(self._has_rigid_coupling):
            # The impulse of the converged pressure on the coupled links, applied once to their velocities.
            n_coupled_6 = 6 * self.coupled_links_idx.shape[0]
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                if self.envs_coupling_active[i_b]:
                    self.coupled_wrench[i_r, i_b] = 0.0
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.envs_coupling_active[i_b]:
                    if self.particles_reordered[i_p, i_b].active:
                        i_m = self.particles_reordered[i_p, i_b].material
                        mass_kappa = self.materials[i_m].mass * self._func_pressure_density2_norm(
                            i_p, i_b, is_divergence
                        )
                        self._func_add_coupled_wrench(i_p, i_b, mass_kappa)
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                if self.envs_coupling_active[i_b]:
                    impulse = self._substep_dt * self.coupled_wrench[i_r, i_b]
                    if qd.static(is_divergence):
                        self.coupled_impulse_div[i_r, i_b] = impulse
                    else:
                        self.coupled_impulse_den[i_r, i_b] = impulse
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                if self.envs_coupling_active[i_b]:
                    vel = self.coupled_vel_base[i_r, i_b]
                    for i_col in range(n_coupled_6):
                        vel += (
                            self._substep_dt * self.coupled_mobility[i_r, i_col, i_b] * self.coupled_wrench[i_col, i_b]
                        )
                    self.coupled_vel[i_r, i_b] = vel
            for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
                if self.envs_coupling_active[i_b]:
                    self._func_set_coupled_links_vel(i_c, i_b)

    def _pressure_solve(self, is_divergence, keep_pressure=False, accelerate=False):
        """
        Run the divergence solve or the density solve and apply the resulting pressure accelerations to velocities.

        Iterations continue until the average error of every environment drops below the tolerance, within the
        min / max iteration bounds. keep_pressure starts the density solve from the pressure it holds instead of the
        warm start, which a later coupling pass of a substep does. Such a pass also has to reduce its starting error
        tenfold, since the coupling passes only converge while their density solves keep reducing their own error
        (inexact iteration with a forcing term of 0.1), down to a floor of a thousandth of the tolerance. Returns the
        number of iterations and the final error over the tolerance (at most 1 once converged). accelerate runs a
        density solve with Chebyshev acceleration (see below).
        """
        if is_divergence:
            n_iters_min, n_iters_max = self._df_min_div_iters, self._df_max_div_iters
            # The divergence error is a rate, so its tolerance is the density tolerance per substep.
            max_error = self._df_max_error_div / self._substep_dt
        else:
            n_iters_min, n_iters_max = self._df_min_den_iters, self._df_max_den_iters
            max_error = self._df_max_error_den

        self._kernel_init_pressure_solve(is_divergence, keep_pressure)

        # Chebyshev semi-iterative acceleration of the Jacobi iteration (Wang 2015, "A chebyshev semi-iterative approach
        # for accelerating projective and position-based dynamics"): iteration k + 1 moves the pressure to
        # p_k-1 + omega_k+1 (jacobi(p_k) - p_k-1), with omega_1 = 1, omega_2 = 2 / (2 - r^2) and
        # omega_k+1 = 4 / (4 - r^2 omega_k), r the bound on the spectral radius of the Jacobi iteration. The pressure
        # stays clamped to be nonnegative and the fixed point does not change. It accelerates the density solve that
        # starts a substep. Jacobi damps the particle-scale part of the error first, while the accelerated iterate
        # keeps a share of it until the error is well below the tolerance on its average: stopped at the tolerance, it
        # sustains velocity noise that the plain solve removes (water at rest in a closed cup, SPH_SOLVER_STATUS.md
        # finding 15). An accelerated solve therefore works to a tenth of the tolerance. The divergence solve mostly
        # takes a single iteration and is not accelerated.
        is_accelerated = accelerate and not is_divergence and self._df_chebyshev_radius > 0.0
        rho2 = self._df_chebyshev_radius**2 if is_accelerated else 0.0
        if is_accelerated:
            max_error *= 0.1
        omega = 1.0
        n_iters, is_converged = 0, False
        while (not is_converged or n_iters < n_iters_min) and n_iters < n_iters_max:
            if rho2 > 0.0 and n_iters > 0:
                omega = 2.0 / (2.0 - rho2) if n_iters == 1 else 4.0 / (4.0 - rho2 * omega)
            self._kernel_pressure_solve_iteration(is_divergence, omega)
            n_iters += 1
            # The error of an iteration costs a reduction over envs and a host synchronization, about a quarter of an
            # iteration of 19k particles: it is read after the first iteration and then every fourth.
            if n_iters == 1 or n_iters % 4 == 0 or n_iters >= n_iters_max:
                rel_error = self._kernel_pressure_solve_error(
                    max_error, n_iters == 1, keep_pressure, n_iters >= n_iters_min
                )
                is_converged = rel_error <= 1.0
        avg_error = self._kernel_pressure_solve_max_error()

        gs.logger.debug(
            f"DFSPH {'divergence' if is_divergence else 'density'} solve: {n_iters} iterations, avg error {avg_error:.4g}"
        )

        self._kernel_apply_pressure_acc(is_divergence)
        return n_iters, avg_error / (10.0 * max_error if is_accelerated else max_error)

    # ------------------------------------------------------------------------------------
    # -------------------------------- implicit viscosity --------------------------------
    # ------------------------------------------------------------------------------------
    # Backward Euler step of the Rayleigh dissipation function of SPH_SOLVER_DESIGN.md section 3.3: the viscous pair
    # force between i and j is -k_ij ((v_i - v_j) . x_ij) x_ij / r_ij^2, along x_ij, so the operator conserves linear
    # and angular momentum and its null modes are the rigid motions. The solve (M + h D) v = M v_rhs is symmetric
    # positive definite, so it runs as conjugate gradient with the 3x3 diagonal blocks as preconditioner. The walls
    # enter with the velocity of their boundary particles, which gives the no-slip condition.

    @qd.func
    def _func_viscosity_pair_coefficient(self, i, j, x_ij, i_b):
        """Return k_ij / r_ij^2 of the fluid pair (i, j), in N s / m^3, with the harmonic mean of their viscosities."""
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        mu_i = self.materials[i_m_i].dynamic_viscosity
        mu_j = self.materials[i_m_j].dynamic_viscosity
        res = gs.qd_float(0.0)
        r2 = x_ij.norm_sqr()
        if mu_i * mu_j > 0.0 and r2 > self._df_eps * self._support_radius**2:
            mu_ij = 2.0 * mu_i * mu_j / (mu_i + mu_j)
            basis = self._func_viscosity_basis(x_ij / qd.sqrt(r2))
            scale_ij = qd.max(
                0.5
                * basis.dot(
                    self.particles_viscosity[i, i_b].viscosity_correction
                    + self.particles_viscosity[j, i_b].viscosity_correction
                ),
                0.0,
            )
            res = (
                2.0
                * (3.0 + 2.0)
                * scale_ij
                * mu_ij
                * self._particle_volume**2
                * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
                / ((r2 + 0.01 * self._support_radius**2) * r2)
            )
        return res

    @qd.func
    def _func_viscosity_basis(self, u):
        """Return the even real spherical harmonics of degree 0, 2 and 4 of the unit vector u (unnormalized)."""
        x, y, z = u[0], u[1], u[2]
        x2, y2, z2 = x * x, y * y, z * z
        return qd.Vector(
            [
                1.0,
                x * y,
                y * z,
                x * z,
                0.5 * (x2 - y2),
                0.28867513459481287 * (3.0 * z2 - 1.0),
                x * y * (x2 - y2),
                y * z * (3.0 * x2 - y2),
                x * y * (7.0 * z2 - 1.0),
                y * z * (7.0 * z2 - 3.0),
                0.125 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
                x * z * (7.0 * z2 - 3.0),
                0.5 * (x2 - y2) * (7.0 * z2 - 1.0),
                x * z * (x2 - 3.0 * y2),
                0.25 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)),
            ],
            dt=gs.qd_float,
        )

    @qd.func
    def _func_wall_ghost(self, i, j, i_q, i_b):
        """Return ghost fluid point i_q of the wall column of boundary particle j seen from fluid particle i.

        A boundary particle stands for the column of solid behind its share of the geom surface, which a no-slip
        wall fills with ghost fluid whose velocity extrapolates linearly the velocity of the fluid particle
        relative to the wall, to the opposite value at the mirrored distance: the ghost at depth s behind the
        surface, with the fluid particle at distance d in front of it, moves relative to the wall by -(s / d)
        (v_i - v_wall). The column is integrated by the midpoint rule over the kernel support with
        SPHOptions.wall_ghost_points points. Returns x_ig, the weight of the point in the moment of the
        viscosity operator, V_g (-grad W_ig . x_ig) / (r_ig^2 + eta^2), and the extrapolation factor 1 + s / d
        of its relative velocity. The weight is zero for a column of a surface the fluid particle lies behind
        (around a convex edge), which does not face it.
        """
        normal = self.boundary_particles_reordered[j, i_b].normal
        surface_pos = self.boundary_particles_reordered[j, i_b].pos + self._boundary_offset * normal
        dist = (self.particles_reordered[i, i_b].pos - surface_pos).dot(normal)
        # The fluid nearest to the surface stands for the fluid down to half a particle spacing below it, and the
        # column starts at that edge. A wider gap between the edge and the surface (fluid that is not pressed against
        # the wall) is filled by the column as well, with the same extrapolated velocity, else the wall transmits only
        # part of the shear stress across it. A narrower one (fluid pressed closer than half a spacing, as against a
        # volume map) starts the column below the surface, else the column and the fluid particles both stand for the
        # fluid near the surface and the wall transmits too much (1.34 of mu gamma with the first layer at a quarter
        # spacing, 0.93 with the column starting at the edge, offline lattice computation).
        gap = self.boundary_particles_fluid_gap[j, i_b] * (self._support_radius / FLUID_GAP_UNITS)
        band = gs.qd_float(0.0)
        if gap < self._support_radius:
            band = gap - 0.5 * self._particle_size
        length = self._support_radius + band
        n_ghosts = qd.static(self._n_wall_ghosts)
        depth = -band + (i_q + 0.5) / n_ghosts * length
        x_ig = self.particles_reordered[i, i_b].pos - (surface_pos - depth * normal)
        if qd.static(self._wall_viscosity_model == "boundary_pairs"):
            # benchmark control (SPH_SOLVER_GOAL.md N6, A2): the boundary particle itself acts as a fluid neighbor
            # moving with the wall, with its sampling volume, and without the extrapolation of a ghost column
            x_ig = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        r2 = x_ig.norm_sqr()
        weight = gs.qd_float(0.0)
        factor = gs.qd_float(1.0)
        if dist > 0.0 and r2 > self._df_eps * self._support_radius**2:
            # A fluid particle pressed against the wall gets the friction of a particle at a twentieth of the support
            # radius, which bounds the extrapolation factor.
            dist_eff = qd.max(dist, 0.05 * self._support_radius)
            column_volume = self.boundary_particles_reordered[j, i_b].area * length / n_ghosts
            if qd.static(self._wall_viscosity_model == "boundary_pairs"):
                column_volume = self.boundary_particles_reordered[j, i_b].volume
            weight = (
                column_volume * (-self.cubic_kernel_derivative(x_ig).dot(x_ig)) / (r2 + 0.01 * self._support_radius**2)
            )
            if qd.static(self._wall_viscosity_model != "boundary_pairs"):
                factor = 1.0 + depth / dist_eff
        return x_ig, weight, factor

    @qd.func
    def _func_wall_viscosity_tensor(self, i, j, i_b):
        """Return the wall viscosity tensor of fluid particle i due to boundary particle j, in N s / m.

        Each ghost point of the wall column of j (see _func_wall_ghost) acts on particle i through the pair dissipation
        of the fluid (SPH_SOLVER_DESIGN.md section 3.3), with the directional correction of particle i, and with its
        relative velocity (1 + s / d) (v_i - v_wall). The extrapolation holds for the tangential velocity alone: at an
        impermeable no-slip wall the normal velocity of an incompressible flow grows quadratically with the distance, so
        its viscous coupling to the wall vanishes to leading order, and the pressure carries the impermeability. The
        tensor is therefore projected on the tangent plane of the wall at the boundary particle. It is symmetric
        positive semi-definite.
        """
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        i_m = self.particles_reordered[i, i_b].material
        mu_i = self.materials[i_m].dynamic_viscosity
        correction = self.particles_viscosity[i, i_b].viscosity_correction
        for i_q in range(qd.static(self._n_wall_ghosts)):
            x_ig, weight, factor = self._func_wall_ghost(i, j, i_q, i_b)
            if weight > 0.0:
                r2 = x_ig.norm_sqr()
                scale = qd.max(self._func_viscosity_basis(x_ig / qd.sqrt(r2)).dot(correction), 0.0)
                tensor += (
                    2.0 * (3.0 + 2.0) * mu_i * self._particle_volume * scale * weight * factor / r2
                ) * x_ig.outer_product(x_ig)
        normal = self.boundary_particles_reordered[j, i_b].normal
        tangent_proj = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
        return tangent_proj @ tensor @ tangent_proj

    @qd.func
    def _func_add_viscosity_moment(self, x, weight, ret: qd.template()):
        """Add a neighbor at x_ij with the given moment weight to the moments of the viscosity operator of a particle."""
        basis = self._func_viscosity_basis(x / x.norm())
        ret.moment += weight
        # The upper triangle only: the Gram matrix is symmetric, and its lower triangle is filled once at the end (see
        # _kernel_compute_viscosity_geometry), which halves the work of every neighbor and ghost point.
        for a in qd.static(range(N_VISCOSITY_BASIS)):
            for b in qd.static(range(a, N_VISCOSITY_BASIS)):
                ret.gram[a, b] += weight * basis[a] * basis[b]

    @qd.func
    def _task_compute_viscosity_moment(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        r2 = x_ij.norm_sqr()
        if r2 > self._df_eps * self._support_radius**2:
            weight = (
                self._particle_volume
                * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
                * r2
                / (r2 + 0.01 * self._support_radius**2)
            )
            self._func_add_viscosity_moment(x_ij, weight, ret)

    @qd.func
    def _task_compute_boundary_viscosity_moment(self, i, j, ret: qd.template(), i_b):
        for i_q in range(qd.static(self._n_wall_ghosts)):
            x_ig, weight, _ = self._func_wall_ghost(i, j, i_q, i_b)
            if weight > 0.0:
                self._func_add_viscosity_moment(x_ig, weight * x_ig.norm_sqr(), ret)

    @qd.func
    def _task_compute_boundary_viscosity_geometry(self, i, j, ret: qd.template(), i_b):
        tensor = self._func_wall_viscosity_tensor(i, j, i_b)
        i_l = self.boundary_particles_reordered[j, i_b].link_idx
        # ret stacks the tensors of the slots, rows 3 k to 3 k + 2 for slot k
        for k in qd.static(range(N_BOUNDARY_SLOTS)):
            if self.particles_boundary_slots[i, i_b, k].link_idx == i_l:
                for a in qd.static(range(3)):
                    for b in qd.static(range(3)):
                        ret[3 * k + a, b] += tensor[a, b]

    @qd.func
    def _task_wall_shear_moment(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        x_n = x_ij.dot(ret.normal)
        ret.moment += k_ij * x_n * (x_ij.dot(ret.tangent_1) ** 2 + x_ij.dot(ret.tangent_2) ** 2)
        i_m = self.particles_reordered[j, i_b].material
        ret.deficit += self.materials[i_m].rest_volume * self.cubic_kernel_derivative(x_ij)

    @qd.func
    def _func_wall_shear_scale(self, i_p, i_b, wall_tensor):
        """Return the factor of the wall viscosity of particle i_p that makes its operator exact for a shear along the wall.

        A linear shear u = gamma zeta t along a wall, zeta the distance from it and t a tangent, has no viscous force.
        The fluid pairs of a particle near the wall, whose neighbors on the wall side are missing, leave the force
        gamma sum_j k_ij (x_ij . n)(x_ij . t)^2 along t, and the wall must cancel it with its force gamma zeta t^T W t:
        the factor is their ratio, averaged over two tangents, for the nearest wall of the particle, where the missing
        neighbors lie behind that wall alone (the gradient of the kernel sum within 25 degrees of its normal; 1 in a
        corner, where the test shear does not vanish on the other wall). A wall that transmits a linear shear exactly
        does not slip, whatever the distance at which the fluid rests from it, which the boundary offset and the
        pressure set. The factor is bounded to [0.25, 4].
        """
        scale = gs.qd_float(1.0)
        k_near = -1
        clearance = self._support_radius
        for k in range(N_BOUNDARY_SLOTS):
            if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                if self.particles_boundary_slots[i_p, i_b, k].clearance < clearance:
                    clearance = self.particles_boundary_slots[i_p, i_b, k].clearance
                    k_near = k
        if k_near != -1:
            normal = self.particles_boundary_slots[i_p, i_b, k_near].clearance_normal
            helper = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if qd.abs(normal[0]) > 0.9:
                helper = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
            tangent_1 = normal.cross(helper).normalized()
            tangent_2 = normal.cross(tangent_1)
            ret = qd.Struct(
                normal=normal,
                tangent_1=tangent_1,
                tangent_2=tangent_2,
                moment=gs.qd_float(0.0),
                deficit=qd.Vector.zero(gs.qd_float, 3),
            )
            self._func_for_all_fluid_neighbors(i_p, ret, self._task_wall_shear_moment, i_b)
            zeta = qd.max(clearance, 0.05 * self._support_radius)
            wall = zeta * (tangent_1.dot(wall_tensor @ tangent_1) + tangent_2.dot(wall_tensor @ tangent_2))
            # the missing neighbors lie behind the nearest wall alone (not in a corner, where another wall of the link
            # takes a part of them and the test shear does not vanish on it)
            is_flat = ret.deficit.dot(normal) > 0.9 * ret.deficit.norm()
            if wall > gs.EPS and ret.moment < 0.0 and is_flat:
                raw = -ret.moment / wall
                scale = qd.min(qd.max(raw, 0.25), 4.0)
                if qd.static(self._correction_diagnostics):
                    self.envs_diagnostics[i_b, 5] += 1.0
                    if raw < 0.25:
                        self.envs_diagnostics[i_b, 7] += 1.0
                    if raw > 4.0:
                        self.envs_diagnostics[i_b, 8] += 1.0
                    self.envs_diagnostics[i_b, 9] += scale
                    self.envs_diagnostics[i_b, 10] += scale * scale
            elif qd.static(self._correction_diagnostics):
                self.envs_diagnostics[i_b, 6] += 1.0
        return scale

    @qd.func
    def _task_count_clipped_pairs(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        r2 = x_ij.norm_sqr()
        if r2 > self._df_eps * self._support_radius**2:
            basis = self._func_viscosity_basis(x_ij / qd.sqrt(r2))
            raw = 0.5 * basis.dot(
                self.particles_viscosity[i, i_b].viscosity_correction
                + self.particles_viscosity[j, i_b].viscosity_correction
            )
            ret.n_pairs += 1.0
            if raw < 0.0:
                ret.n_clipped += 1.0

    @qd.kernel
    def _kernel_viscosity_diagnostics(self):
        """Count the fluid pairs of the viscosity operator and those whose corrected weight is clipped at zero."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(n_pairs=gs.qd_float(0.0), n_clipped=gs.qd_float(0.0))
                self._func_for_all_fluid_neighbors(i_p, ret, self._task_count_clipped_pairs, i_b)
                self.envs_diagnostics[i_b, 3] += ret.n_pairs
                self.envs_diagnostics[i_b, 4] += ret.n_clipped

    @qd.kernel
    def _kernel_reset_diagnostics(self):
        for i_b, i_d in qd.ndrange(self._B, N_DIAGNOSTICS):
            self.envs_diagnostics[i_b, i_d] = 0.0

    def _accumulate_diagnostics(self, stats):
        """Add the activation diagnostics of the substep to the stats (see SPHOptions.correction_diagnostics)."""
        values = qd_to_torch(self.envs_diagnostics)
        for i_d, (name, is_sum) in enumerate(DIAGNOSTICS):
            column = values[:, i_d]
            if is_sum:
                setattr(stats, name, getattr(stats, name) + float(column.sum()))
            else:
                setattr(stats, name, max(getattr(stats, name), float(column.max())))

    @qd.kernel
    def _kernel_compute_viscosity_geometry(self):
        """Compute the directional correction of the viscosity operator and the wall viscosity tensors of every particle.

        The pair dissipation of SPH_SOLVER_DESIGN.md section 3.3 reproduces the Laplacian of a quadratic velocity field
        only if the fourth moment tensor of its weights over the neighbors, sum_j w_ij u_ij u_ij u_ij u_ij with
        w_ij = V_j (-grad W_ij . x_ij) r_ij^2 / (r_ij^2 + eta^2) and u_ij = x_ij / r_ij, is isotropic with the trace 3 of
        its continuum value, the ghost fluid of the wall columns (see _func_wall_ghost) completing the neighborhood. A
        kernel support of two particle spacings holds too few neighbors for this: on a cubic lattice the operator
        reproduces 0.70 of the Laplacian of a shear flow and 1.19 of that of an extension, and a relaxed arrangement
        keeps a part of this anisotropy. The weights are therefore corrected by a function s_i(u) of the direction in
        the 15 even spherical harmonics up to degree 4, which span the fourth moment tensors: its coefficients solve
        the moment equations sum_j w_ij s_i(u_ij) Y_k(u_ij) = 3 rhohat_i delta_k0, with a Tikhonov term of a thousandth
        of the mean diagonal, since a lattice may offer fewer than 15 independent directions. The pair weight is the
        mean of s_i and s_j, clamped at zero, so the operator stays symmetric, positive semi-definite and central, and
        keeps its momentum conservation and its null modes.

        The normalized density rhohat_i in the target keeps the correction of a particle at the free surface at its
        bulk magnitude, since its missing neighbors carry no velocity difference. There the neighborhood is a half
        space, whose anisotropy the correction must not undo, so it blends linearly to the isotropic scale
        3 rhohat_i / sum_j w_ij as rhohat_i drops from 0.95 to 0.85.

        The wall viscosity tensor of each boundary slot is the sum of the ghost column tensors of the link (see
        _func_wall_viscosity_tensor), and the wall force on the particle is -M (v_i - v_wall(x_i)) per slot.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(
                    moment=gs.qd_float(0.0),
                    gram=qd.Matrix.zero(gs.qd_float, N_VISCOSITY_BASIS, N_VISCOSITY_BASIS),
                )
                self._func_for_all_fluid_neighbors(i_p, ret, self._task_compute_viscosity_moment, i_b)
                if qd.static(self._has_boundary):
                    self._func_for_all_boundary_neighbors(i_p, ret, self._task_compute_boundary_viscosity_moment, i_b)
                density_norm = self.particles_reordered[i_p, i_b].density_norm
                correction = qd.Vector.zero(gs.qd_float, N_VISCOSITY_BASIS)
                correction[0] = 1.0
                # benchmark controls (SPH_SOLVER_GOAL.md N6, A1): "none" keeps the uncorrected pair coefficient,
                # "isotropic" only rescales it to the trace of the continuum moment
                if qd.static(self._correction_diagnostics):
                    self.envs_diagnostics[i_b, 0] += 1.0
                if ret.moment > self._df_eps and qd.static(self._viscosity_correction != "none"):
                    correction[0] = 3.0 * density_norm / ret.moment
                    blend = qd.min(qd.max((density_norm - 0.85) / 0.1, 0.0), 1.0)
                    if qd.static(self._viscosity_correction == "isotropic"):
                        blend = 0.0
                    elif qd.static(self._correction_diagnostics):
                        if blend == 0.0:
                            self.envs_diagnostics[i_b, 2] += 1.0
                        elif blend < 1.0:
                            self.envs_diagnostics[i_b, 1] += 1.0
                    if blend > 0.0:
                        regularization = 1e-3 * ret.gram.trace() / N_VISCOSITY_BASIS
                        gram = ret.gram + regularization * qd.Matrix.identity(gs.qd_float, N_VISCOSITY_BASIS)
                        for a in qd.static(range(N_VISCOSITY_BASIS)):
                            for b in qd.static(range(a)):
                                gram[a, b] = gram[b, a]
                        target = qd.Vector.zero(gs.qd_float, N_VISCOSITY_BASIS)
                        target[0] = 3.0 * density_norm
                        correction = blend * self._func_solve_spd(gram, target) + (1.0 - blend) * correction
                self.particles_viscosity[i_p, i_b].viscosity_correction = correction

        if qd.static(self._has_boundary):
            # The boundary neighbors of a particle, each with the 8 ghost points of its wall column, are shared among
            # N_NEIGHBOR_SHARES threads (see _kernel_pressure_solve_iteration), whose slot tensors a loop over the
            # particles adds in a fixed order.
            for i_s, i_p, i_b in qd.ndrange(N_NEIGHBOR_SHARES, self._n_particles, self._B):
                if self.particles_reordered[i_p, i_b].active:
                    tensors = qd.Matrix.zero(gs.qd_float, 3 * N_BOUNDARY_SLOTS, 3)
                    self._func_for_boundary_neighbor_share(
                        i_p, i_s, tensors, self._task_compute_boundary_viscosity_geometry, i_b
                    )
                    for k in qd.static(range(N_BOUNDARY_SLOTS)):
                        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                        for a in qd.static(range(3)):
                            for b in qd.static(range(3)):
                                tensor[a, b] = tensors[3 * k + a, b]
                        self.particles_wall_tensor_shares[i_s, i_p, i_b, k] = tensor

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_reordered[i_p, i_b].active:
                    wall_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for k in qd.static(range(N_BOUNDARY_SLOTS)):
                        slot_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                        for i_s in qd.static(range(N_NEIGHBOR_SHARES)):
                            slot_tensor += self.particles_wall_tensor_shares[i_s, i_p, i_b, k]
                        self.particles_boundary_slots[i_p, i_b, k].wall_tensor = slot_tensor
                        wall_tensor += slot_tensor
                    if qd.static(self._wall_shear_consistency):
                        scale = self._func_wall_shear_scale(i_p, i_b, wall_tensor)
                        wall_tensor *= scale
                        for k in range(N_BOUNDARY_SLOTS):
                            self.particles_boundary_slots[i_p, i_b, k].wall_tensor *= scale
                    self.particles_viscosity[i_p, i_b].wall_tensor = wall_tensor
        else:
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                self.particles_viscosity[i_p, i_b].wall_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def _func_solve_spd(self, mat, rhs):
        """Solve the symmetric positive definite system mat x = rhs of size N_VISCOSITY_BASIS by Cholesky factorization."""
        n = qd.static(N_VISCOSITY_BASIS)
        lower = qd.Matrix.zero(gs.qd_float, n, n)
        for c in qd.static(range(n)):
            diag = mat[c, c]
            for k in qd.static(range(c)):
                diag -= lower[c, k] * lower[c, k]
            diag = qd.sqrt(qd.max(diag, gs.EPS))
            lower[c, c] = diag
            for r in qd.static(range(c + 1, n)):
                value = mat[r, c]
                for k in qd.static(range(c)):
                    value -= lower[r, k] * lower[c, k]
                lower[r, c] = value / diag
        sol = qd.Vector.zero(gs.qd_float, n)
        for r in qd.static(range(n)):
            value = rhs[r]
            for k in qd.static(range(r)):
                value -= lower[r, k] * sol[k]
            sol[r] = value / lower[r, r]
        for r_ in qd.static(range(n)):
            r = qd.static(n - 1 - r_)
            value = sol[r]
            for k in qd.static(range(r + 1, n)):
                value -= lower[k, r] * sol[k]
            sol[r] = value / lower[r, r]
        return sol

    @qd.func
    def _func_link_vel_at_pos(self, i_l, i_b, pos):
        """Return the rigid velocity of link i_l at a point of space, at the current substep."""
        return self.links_vel[i_l, i_b] + self.links_ang[i_l, i_b].cross(pos - self.links_root_com[i_l, i_b])

    @qd.func
    def _task_init_viscosity_solve(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        ret.image += self._substep_dt * k_ij * v_ij.dot(x_ij) * x_ij
        ret.block += self._substep_dt * k_ij * x_ij.outer_product(x_ij)
        # The diagonal and the off-diagonal blocks of the pair both have the norm h k_ij r_ij^2.
        ret.row_norm += 2.0 * self._substep_dt * k_ij * x_ij.norm_sqr()

    @qd.kernel
    def _kernel_init_viscosity_solve(self, is_first_pass: bool):
        """Start the conjugate gradient solve of the viscosity from the current velocity.

        The right-hand side velocity is v* plus the velocity change of the last density solve (SPH_SOLVER_DESIGN.md
        section 5.2). is_first_pass marks the first coupling pass of the substep, whose initial residual (the viscous
        impulse of the substep) scales the tolerance of the agreement between viscosity and pressure.
        """
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                self.envs_viscosity[i_b].residual_norm2 = 0.0
                self.envs_viscosity[i_b].residual_norm2_init = 0.0
                self.envs_viscosity[i_b].rhs_norm2 = 0.0
                if is_first_pass:
                    self.envs_viscosity[i_b].gravity_norm2 = 0.0
                    self.envs_viscosity[i_b].viscous_stiffness_bound = 1.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                terms = qd.Vector.zero(gs.qd_float, 4)
                if self.particles_reordered[i_p, i_b].active:
                    i_m = self.particles_reordered[i_p, i_b].material
                    mass = self.materials[i_m].mass
                    vel_rhs = (
                        self.particles_viscosity[i_p, i_b].vel_star + self.particles_viscosity[i_p, i_b].pressure_vel
                    )
                    ret = qd.Struct(
                        image=mass * self.particles_reordered[i_p, i_b].vel,
                        block=mass * qd.Matrix.identity(gs.qd_float, 3),
                        rhs=mass * vel_rhs,
                        row_norm=gs.qd_float(0.0),
                    )
                    self._func_for_all_fluid_neighbors(i_p, ret, self._task_init_viscosity_solve, i_b)
                    # The wall acts on the particle alone (see _kernel_compute_viscosity_geometry), so it only adds to
                    # the diagonal block, and the wall velocity is known in the solve, so its part goes to the
                    # right-hand side.
                    wall_block = self._substep_dt * self.particles_viscosity[i_p, i_b].wall_tensor
                    ret.block += wall_block
                    ret.image += wall_block @ self.particles_reordered[i_p, i_b].vel
                    ret.row_norm += wall_block.trace()
                    if qd.static(self._has_boundary):
                        for k in range(N_BOUNDARY_SLOTS):
                            if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                                ret.rhs += self._substep_dt * (
                                    self.particles_boundary_slots[i_p, i_b, k].wall_tensor
                                    @ self._func_slot_wall_vel(i_p, i_b, k)
                                )
                    precond_inv = ret.block.inverse()
                    residual = ret.rhs - ret.image
                    residual_precond = precond_inv @ residual
                    self.particles_viscosity[i_p, i_b].precond_inv = precond_inv
                    self.particles_viscosity[i_p, i_b].residual = residual
                    self.particles_viscosity[i_p, i_b].residual_precond = residual_precond
                    self.particles_viscosity[i_p, i_b].direction = residual_precond
                    terms[0] = residual.dot(residual_precond)
                    terms[1] = ret.rhs.dot(precond_inv @ ret.rhs)
                    # The velocity change of gravity over the substep, in the norm of the mass, so that the scale does
                    # not shrink with the viscosity.
                    terms[2] = mass * (self._substep_dt * self._gravity[i_b]).norm_sqr()
                    terms[3] = 1.0 + ret.row_norm / mass
                self.particles_reduction[i_p, i_b] = terms

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                total, largest = self._func_chunk_sum(i_c, i_b)
                self.envs_viscosity[i_b].residual_norm2 += total[0]
                self.envs_viscosity[i_b].rhs_norm2 += total[1]
                if is_first_pass:
                    self.envs_viscosity[i_b].gravity_norm2 += total[2]
                    qd.atomic_max(self.envs_viscosity[i_b].viscous_stiffness_bound, largest)

        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                self.envs_viscosity[i_b].residual_norm2_init = self.envs_viscosity[i_b].residual_norm2
                if is_first_pass:
                    self.envs_viscosity[i_b].viscous_norm2 = self.envs_viscosity[i_b].residual_norm2

    @qd.func
    def _task_apply_viscosity_operator(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        d_ij = self.particles_viscosity[i, i_b].direction - self.particles_viscosity[j, i_b].direction
        ret += self._substep_dt * k_ij * d_ij.dot(x_ij) * x_ij

    @qd.func
    def _func_viscosity_tolerance2(self, i_b):
        """Return the squared tolerance on the preconditioned residual norm of the viscosity solve of env i_b.

        The tolerance is relative to the initial residual, which is the viscous impulse of the substep, with a floor
        relative to the momentum that single precision resolves.
        """
        return qd.max(
            self._visc_max_error**2 * self.envs_viscosity[i_b].residual_norm2_init,
            1e-12 * self.envs_viscosity[i_b].rhs_norm2,
        )

    @qd.kernel
    def _kernel_viscosity_solve_iteration(self) -> float:
        """Run one conjugate gradient iteration of the viscosity solve and return the worst relative residual over envs.

        The relative residual is the residual norm over its tolerance, so the solve has converged once it is at most 1.
        An env that has converged keeps its velocity.
        """
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                self.envs_viscosity[i_b].direction_norm2 = 0.0
                self.envs_viscosity[i_b].residual_norm2_next = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
                terms = qd.Vector.zero(gs.qd_float, 4)
                if self.particles_reordered[i_p, i_b].active and is_iterating:
                    direction = self.particles_viscosity[i_p, i_b].direction
                    i_m = self.particles_reordered[i_p, i_b].material
                    image = self.materials[i_m].mass * direction
                    self._func_for_all_fluid_neighbors(i_p, image, self._task_apply_viscosity_operator, i_b)
                    if qd.static(self._has_boundary):
                        image += self._substep_dt * (self.particles_viscosity[i_p, i_b].wall_tensor @ direction)
                    self.particles_viscosity[i_p, i_b].direction_image = image
                    terms[0] = direction.dot(image)
                self.particles_reduction[i_p, i_b] = terms

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                total, _ = self._func_chunk_sum(i_c, i_b)
                self.envs_viscosity[i_b].direction_norm2 += total[0]

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
                terms = qd.Vector.zero(gs.qd_float, 4)
                if self.particles_reordered[i_p, i_b].active and is_iterating:
                    step = gs.qd_float(0.0)
                    if self.envs_viscosity[i_b].direction_norm2 > 0.0:
                        step = self.envs_viscosity[i_b].residual_norm2 / self.envs_viscosity[i_b].direction_norm2
                    self.particles_reordered[i_p, i_b].vel += step * self.particles_viscosity[i_p, i_b].direction
                    residual = (
                        self.particles_viscosity[i_p, i_b].residual
                        - step * self.particles_viscosity[i_p, i_b].direction_image
                    )
                    residual_precond = self.particles_viscosity[i_p, i_b].precond_inv @ residual
                    self.particles_viscosity[i_p, i_b].residual = residual
                    self.particles_viscosity[i_p, i_b].residual_precond = residual_precond
                    terms[0] = residual.dot(residual_precond)
                self.particles_reduction[i_p, i_b] = terms

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                total, _ = self._func_chunk_sum(i_c, i_b)
                self.envs_viscosity[i_b].residual_norm2_next += total[0]

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
                if self.particles_reordered[i_p, i_b].active and is_iterating:
                    beta = self.envs_viscosity[i_b].residual_norm2_next / self.envs_viscosity[i_b].residual_norm2
                    self.particles_viscosity[i_p, i_b].direction = (
                        self.particles_viscosity[i_p, i_b].residual_precond
                        + beta * self.particles_viscosity[i_p, i_b].direction
                    )

        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                if self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b):
                    self.envs_viscosity[i_b].residual_norm2 = self.envs_viscosity[i_b].residual_norm2_next
                qd.atomic_max(
                    max_rel_residual,
                    qd.sqrt(
                        self.envs_viscosity[i_b].residual_norm2 / qd.max(self._func_viscosity_tolerance2(i_b), 1e-30)
                    ),
                )
        return max_rel_residual

    @qd.kernel
    def _kernel_viscosity_residual(self) -> float:
        """Return the worst relative residual over envs of the viscosity solve as it stands, without iterating."""
        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                qd.atomic_max(
                    max_rel_residual,
                    qd.sqrt(
                        self.envs_viscosity[i_b].residual_norm2 / qd.max(self._func_viscosity_tolerance2(i_b), 1e-30)
                    ),
                )
        return max_rel_residual

    @qd.func
    def _func_cross_matrix(self, y):
        """Return the matrix Y of the cross product by y, Y v = y x v."""
        return qd.Matrix([[0.0, -y[2], y[1]], [y[2], 0.0, -y[0]], [-y[1], y[0], 0.0]], dt=gs.qd_float)

    @qd.kernel
    def _kernel_project_viscosity_residual(self):
        """Remove the net force and torque of the residual of the viscosity solve by a rigid velocity correction.

        The viscous pair forces are central and opposite, so the exact solution of (M + h D + h W) v = M v_rhs + h W v_w
        changes the momentum of the fluid by the wall impulse alone, -h W (v - v_w), whose reaction the links take. A
        solve stopped at its tolerance leaves the residual r = rhs - A v, and the fluid gains sum_i r_i and
        sum_i x_i x r_i on top, so that the momentum of a closed fluid-rigid system drifts at the order of the
        tolerance. Rigid velocities a + w x (x_i - c) are null modes of D, so adding one changes the net residual by
        -sum_i (m_i + h W_i) (a + w x (x_i - c)) only. The correction solving this 6x6 system for the net residual force
        and torque about the center of mass c of the fluid leaves both at zero and changes the velocities at the order
        of the tolerance.
        """
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                self.envs_viscosity_mass[i_b] = 0.0
                self.envs_viscosity_center[i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.envs_viscosity_rigid_system[i_b] = qd.Matrix.zero(gs.qd_float, 6, 6)
                self.envs_viscosity_rigid_rhs[i_b] = qd.Vector.zero(gs.qd_float, 6)

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                mass_sum = gs.qd_float(0.0)
                moment = qd.Vector.zero(gs.qd_float, 3)
                for i_k in range(REDUCTION_CHUNK):
                    i_p = i_c * REDUCTION_CHUNK + i_k
                    if i_p < self._n_particles:
                        if self.particles_reordered[i_p, i_b].active:
                            mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                            mass_sum += mass
                            moment += mass * self.particles_reordered[i_p, i_b].pos
                self.envs_viscosity_mass[i_b] += mass_sum
                self.envs_viscosity_center[i_b] += moment

        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                if self.envs_viscosity_mass[i_b] > 0.0:
                    self.envs_viscosity_center[i_b] /= self.envs_viscosity_mass[i_b]

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                system = qd.Matrix.zero(gs.qd_float, 6, 6)
                rhs = qd.Vector.zero(gs.qd_float, 6)
                for i_k in range(REDUCTION_CHUNK):
                    i_p = i_c * REDUCTION_CHUNK + i_k
                    if i_p < self._n_particles:
                        if self.particles_reordered[i_p, i_b].active:
                            mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                            block = mass * qd.Matrix.identity(gs.qd_float, 3) + self._substep_dt * (
                                self.particles_viscosity[i_p, i_b].wall_tensor
                            )
                            arm = self._func_cross_matrix(
                                self.particles_reordered[i_p, i_b].pos - self.envs_viscosity_center[i_b]
                            )
                            residual = self.particles_viscosity[i_p, i_b].residual
                            # J^T (M + h W) J and J^T r for the rigid velocity J (a, w) = a + w x y = a - Y w
                            coupling = -block @ arm
                            angular = arm @ coupling
                            for r in qd.static(range(3)):
                                for c in qd.static(range(3)):
                                    system[r, c] += block[r, c]
                                    system[r, 3 + c] += coupling[r, c]
                                    system[3 + c, r] += coupling[r, c]
                                    system[3 + r, 3 + c] += angular[r, c]
                            torque = arm @ residual
                            for r in qd.static(range(3)):
                                rhs[r] += residual[r]
                                rhs[3 + r] += torque[r]
                self.envs_viscosity_rigid_system[i_b] += system
                self.envs_viscosity_rigid_rhs[i_b] += rhs

        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                correction = qd.Vector.zero(gs.qd_float, 6)
                if self.envs_viscosity_mass[i_b] > 0.0:
                    system = self.envs_viscosity_rigid_system[i_b]
                    system += 1e-6 * system.trace() / 6.0 * qd.Matrix.identity(gs.qd_float, 6)
                    correction = self._func_solve_spd6(system, self.envs_viscosity_rigid_rhs[i_b])
                self.envs_viscosity_rigid_correction[i_b] = correction
                if qd.static(self._correction_diagnostics):
                    linear = qd.Vector([correction[0], correction[1], correction[2]], dt=gs.qd_float)
                    angular = qd.Vector([correction[3], correction[4], correction[5]], dt=gs.qd_float)
                    self.envs_diagnostics[i_b, 11] = qd.max(self.envs_diagnostics[i_b, 11], linear.norm())
                    self.envs_diagnostics[i_b, 12] = qd.max(self.envs_diagnostics[i_b, 12], angular.norm())

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    correction = self.envs_viscosity_rigid_correction[i_b]
                    arm = self.particles_reordered[i_p, i_b].pos - self.envs_viscosity_center[i_b]
                    linear = qd.Vector([correction[0], correction[1], correction[2]], dt=gs.qd_float)
                    angular = qd.Vector([correction[3], correction[4], correction[5]], dt=gs.qd_float)
                    dvel = linear + angular.cross(arm)
                    mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                    block = mass * qd.Matrix.identity(gs.qd_float, 3) + self._substep_dt * (
                        self.particles_viscosity[i_p, i_b].wall_tensor
                    )
                    self.particles_reordered[i_p, i_b].vel += dvel
                    self.particles_viscosity[i_p, i_b].residual -= block @ dvel

    @qd.func
    def _func_solve_spd6(self, mat, rhs):
        """Solve the symmetric positive definite system mat x = rhs of size 6 by Cholesky factorization."""
        lower = qd.Matrix.zero(gs.qd_float, 6, 6)
        for c in qd.static(range(6)):
            diag = mat[c, c]
            for k in qd.static(range(c)):
                diag -= lower[c, k] * lower[c, k]
            diag = qd.sqrt(qd.max(diag, gs.EPS))
            lower[c, c] = diag
            for r in qd.static(range(c + 1, 6)):
                value = mat[r, c]
                for k in qd.static(range(c)):
                    value -= lower[r, k] * lower[c, k]
                lower[r, c] = value / diag
        sol = qd.Vector.zero(gs.qd_float, 6)
        for r in qd.static(range(6)):
            value = rhs[r]
            for k in qd.static(range(r)):
                value -= lower[r, k] * sol[k]
            sol[r] = value / lower[r, r]
        for r_ in qd.static(range(6)):
            r = qd.static(5 - r_)
            value = sol[r]
            for k in qd.static(range(r + 1, 6)):
                value -= lower[k, r] * sol[k]
            sol[r] = value / lower[r, r]
        return sol

    def _viscosity_solve(self, is_first_pass):
        """Solve the implicit viscosity of the substep from the current velocity, in place.

        The solve ends with a rigid velocity correction that cancels the net force and torque of its residual (see
        _kernel_project_viscosity_residual). Returns the number of iterations and the final relative residual (at most
        1 once converged).
        """
        self._kernel_init_viscosity_solve(is_first_pass)
        rel_residual = self._kernel_viscosity_residual()
        n_iters = 0
        while rel_residual > 1.0 and n_iters < self._visc_max_iters:
            rel_residual = self._kernel_viscosity_solve_iteration()
            n_iters += 1
        if self._viscosity_residual_projection:
            self._kernel_project_viscosity_residual()
        return n_iters, rel_residual

    @qd.func
    def _task_apply_viscosity_operator_to_pressure_delta(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        d_ij = self.particles_viscosity[i, i_b].pressure_vel_delta - self.particles_viscosity[j, i_b].pressure_vel_delta
        ret += self._substep_dt * k_ij * d_ij.dot(x_ij) * x_ij

    @qd.kernel
    def _kernel_coupling_residual(self) -> float:
        """Return the worst relative disagreement over envs between the viscosity and the last density solve.

        A density solve that changed the pressure velocity by delta leaves the residual -h D delta in the viscosity
        system of the combined velocity (SPH_SOLVER_DESIGN.md section 5.2). It is measured relative to the larger of the
        viscous and the gravity impulses of the substep, with the floor of the viscosity solve, so the coupling has
        converged once the result is at most 1. The density solves resolve the velocity only to eta dx / h, eta their
        tolerance on the density error (a velocity error of that size moves the particles by eta dx over the substep),
        and those of the later passes reduce their error tenfold (see _pressure_solve), so the coupling has converged
        as well once the mass-weighted rms of delta is at most a tenth of it: without gravity and with a weak viscous
        impulse, the disagreement would otherwise stall at the changes the pressure clamp makes within the tolerance of
        the density solves. The full eta dx / h let a stirred cup stop its passes early enough to misread its viscous
        dissipation by 12 %.
        """
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                self.envs_viscosity[i_b].coupling_norm2 = 0.0
                self.envs_viscosity[i_b].delta_norm2 = 0.0
                self.envs_viscosity[i_b].mass = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                terms = qd.Vector.zero(gs.qd_float, 4)
                if self.particles_reordered[i_p, i_b].active:
                    residual = qd.Vector.zero(gs.qd_float, 3)
                    self._func_for_all_fluid_neighbors(
                        i_p, residual, self._task_apply_viscosity_operator_to_pressure_delta, i_b
                    )
                    if qd.static(self._has_boundary):
                        residual += self._substep_dt * (
                            self.particles_viscosity[i_p, i_b].wall_tensor
                            @ self.particles_viscosity[i_p, i_b].pressure_vel_delta
                        )
                    terms[0] = residual.dot(self.particles_viscosity[i_p, i_b].precond_inv @ residual)
                    mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                    terms[1] = mass * self.particles_viscosity[i_p, i_b].pressure_vel_delta.norm_sqr()
                    terms[2] = mass
                self.particles_reduction[i_p, i_b] = terms

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if self.envs_coupling_active[i_b]:
                total, _ = self._func_chunk_sum(i_c, i_b)
                self.envs_viscosity[i_b].coupling_norm2 += total[0]
                self.envs_viscosity[i_b].delta_norm2 += total[1]
                self.envs_viscosity[i_b].mass += total[2]

        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                impulse_norm2 = qd.max(self.envs_viscosity[i_b].viscous_norm2, self.envs_viscosity[i_b].gravity_norm2)
                tolerance2 = qd.max(self._visc_max_error**2 * impulse_norm2, 1e-12 * self.envs_viscosity[i_b].rhs_norm2)
                rel_residual = qd.sqrt(self.envs_viscosity[i_b].coupling_norm2 / qd.max(tolerance2, 1e-30))
                vel_resolution = 0.1 * self._df_max_error_den * self._particle_size / self._substep_dt
                rel_delta = qd.sqrt(
                    self.envs_viscosity[i_b].delta_norm2
                    / qd.max(self.envs_viscosity[i_b].mass * vel_resolution**2, 1e-30)
                )
                self.envs_coupling_rel[i_b] = qd.min(rel_residual, rel_delta)
                qd.atomic_max(max_rel_residual, self.envs_coupling_rel[i_b])
        return max_rel_residual

    @qd.kernel
    def _kernel_store_coupling_pressure(self, is_first_pass: bool):
        """Record the pressure a coupling pass starts from, which the viscosity solve of the pass reads.

        The first pass starts from zero pressure, since its viscosity solve sees no pressure velocity.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    pressure_in = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                    if is_first_pass:
                        pressure_in = 0.0
                    self.particles_viscosity[i_p, i_b].pressure_in = pressure_in

    @qd.kernel
    def _kernel_accelerate_coupling(self, i_pass: int):
        """Extrapolate the pressure of the next coupling pass by Chebyshev acceleration, and its pressure velocity.

        A coupling pass maps the pressure it starts from to the pressure of its density solve. With the set of compressed
        particles fixed, this map is a Richardson step on the pressure system G A^-1 G^T of the coupled viscosity and
        pressure, preconditioned by the density solve G M^-1 G^T. The eigenvalues of the preconditioned system lie in
        [1 / L, 1], where L bounds the eigenvalues of M^-1 A (Gershgorin disks of the viscosity system), since
        M / L <= A^-1 <= M^-1 in the ordering of symmetric matrices. Chebyshev semi-iteration on that interval converges
        at the rate (sqrt(L) - 1) / (sqrt(L) + 1) instead of 1 - 1 / L, and reduces to the plain iteration as L tends to
        1 for weakly viscous fluids. i_pass counts the passes since the first one or the last restart. The pressure is
        clamped to stay nonnegative.
        """
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                if i_pass == 0:
                    bound = self.envs_viscosity[i_b].viscous_stiffness_bound
                    self.envs_viscosity[i_b].chebyshev_step = 2.0 / (1.0 + 1.0 / bound)
                    self.envs_viscosity[i_b].chebyshev_ratio2 = ((1.0 - 1.0 / bound) / (1.0 + 1.0 / bound)) ** 2
                    self.envs_viscosity[i_b].chebyshev_weight = 1.0
                elif i_pass == 1:
                    self.envs_viscosity[i_b].chebyshev_weight = 1.0 / (
                        1.0 - 0.5 * self.envs_viscosity[i_b].chebyshev_ratio2
                    )
                else:
                    self.envs_viscosity[i_b].chebyshev_weight = 1.0 / (
                        1.0
                        - 0.25 * self.envs_viscosity[i_b].chebyshev_ratio2 * self.envs_viscosity[i_b].chebyshev_weight
                    )

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    pressure_in = self.particles_viscosity[i_p, i_b].pressure_in
                    pressure_in_prev = self.particles_viscosity[i_p, i_b].pressure_in_prev
                    if i_pass == 0:
                        pressure_in_prev = pressure_in
                    correction = self.envs_viscosity[i_b].chebyshev_step * (
                        self.particles_reordered[i_p, i_b].pressure_density2_norm_p - pressure_in
                    )
                    pressure = (
                        self.envs_viscosity[i_b].chebyshev_weight * (pressure_in - pressure_in_prev + correction)
                        + pressure_in_prev
                    )
                    self.particles_viscosity[i_p, i_b].pressure_in_prev = pressure_in
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p = qd.max(pressure, 0.0)

        # The pressure velocity of the extrapolated pressure, which the viscosity solve of the next pass reads
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    self._func_compute_pressure_acc(i_p, i_b, False)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    self.particles_viscosity[i_p, i_b].pressure_vel = (
                        self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc
                    )

    @qd.kernel
    def _kernel_remove_pressure_vel(self):
        """Remove the velocity change of the last density solve, so that the next one starts from the viscous velocity."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    self.particles_reordered[i_p, i_b].vel -= self.particles_viscosity[i_p, i_b].pressure_vel

    # ------------------------------------------------------------------------------------
    # ------------------------------------ fluid load ------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_accumulate_boundary_load(self, is_step_start: bool):
        """Add the fluid impulse of the substep on every rigid link, from the final pressures and velocities.

        The pressure impulse of the divergence and the density solves (SPH_SOLVER_DESIGN.md section 3.2) and the viscous
        impulse of each boundary slot act at the fluid particle (see _kernel_compute_density_and_factor and
        _kernel_compute_viscosity_geometry). is_step_start resets the accumulation, which spans the substeps of one
        scene step.
        """
        if is_step_start:
            for i_l, i_b in qd.ndrange(self.links_fluid_impulse.shape[0], self._B):
                self.links_fluid_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.links_fluid_angular_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                for i_q in qd.static(range(N_LOAD_PARTS)):
                    self.links_fluid_part_impulse[i_l, i_b, i_q] = qd.Vector.zero(gs.qd_float, 3)
                    self.links_fluid_part_angular_impulse[i_l, i_b, i_q] = qd.Vector.zero(gs.qd_float, 3)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                kappa = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                if qd.static(not self._df_skip_div):
                    kappa += self.particles_reordered[i_p, i_b].pressure_density2_norm_d
                i_m = self.particles_reordered[i_p, i_b].material
                mass_kappa = self.materials[i_m].mass * kappa
                pos_i = self.particles_reordered[i_p, i_b].pos
                for k in range(N_BOUNDARY_SLOTS):
                    i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                    if i_l != -1:
                        arm = pos_i - self.links_com[i_l, i_b]
                        pressure = self._substep_dt * mass_kappa * self.particles_boundary_slots[i_p, i_b, k].grad
                        viscous = qd.Vector.zero(gs.qd_float, 3)
                        if qd.static(self._has_viscosity):
                            vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                            viscous = self._substep_dt * (
                                self.particles_boundary_slots[i_p, i_b, k].wall_tensor @ vel_rel
                            )
                        clearance = self.particles_boundary_slots[i_p, i_b, k].clearance_impulse
                        impulse = pressure + viscous + clearance
                        self.links_fluid_impulse[i_l, i_b] += impulse
                        self.links_fluid_angular_impulse[i_l, i_b] += arm.cross(impulse)
                        self.links_fluid_part_impulse[i_l, i_b, 0] += pressure
                        self.links_fluid_part_impulse[i_l, i_b, 1] += viscous
                        self.links_fluid_part_impulse[i_l, i_b, 2] += clearance
                        self.links_fluid_part_angular_impulse[i_l, i_b, 0] += arm.cross(pressure)
                        self.links_fluid_part_angular_impulse[i_l, i_b, 1] += arm.cross(viscous)
                        self.links_fluid_part_angular_impulse[i_l, i_b, 2] += arm.cross(clearance)

    @qd.func
    def _task_compute_viscous_dissipation(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        # each pair is visited from both of its particles
        ret += 0.5 * k_ij * v_ij.dot(x_ij) ** 2

    @qd.kernel
    def _kernel_accumulate_viscous_dissipation(self, is_step_start: bool):
        """Add the kinetic energy the viscosity dissipates over the substep, h v^T D v at the final velocities.

        D is the dissipation operator of the implicit viscosity solve (SPH_SOLVER_DESIGN.md section 3.3), the walls
        included with their velocity relative to the fluid. The backward Euler step dissipates in addition
        (v - v_rhs)^T M (v - v_rhs) / 2, which vanishes to first order in the substep and is not included.
        """
        if is_step_start:
            for i_b in range(self._B):
                self.envs_viscous_dissipation[i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            terms = qd.Vector.zero(gs.qd_float, 4)
            if self.particles_reordered[i_p, i_b].active:
                power = gs.qd_float(0.0)
                self._func_for_all_fluid_neighbors(i_p, power, self._task_compute_viscous_dissipation, i_b)
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                            power += vel_rel.dot(self.particles_boundary_slots[i_p, i_b, k].wall_tensor @ vel_rel)
                terms[0] = self._substep_dt * power
            self.particles_reduction[i_p, i_b] = terms

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            total, _ = self._func_chunk_sum(i_c, i_b)
            self.envs_viscous_dissipation[i_b] += total[0]

    @qd.kernel
    def _kernel_enforce_wall_clearance(self):
        """Keep every fluid particle at least a quarter of a particle spacing in front of the rigid surfaces near it.

        A single layer of boundary particles stops the fluid only through the density constraint, which a particle of
        the free surface, short of fluid neighbors, satisfies only once it comes very close to the boundary particles.
        Hitting a wall, it can cross their layer and get trapped inside the geom. The guard runs after the pressure and
        viscosity solves, as a unilateral velocity constraint with a soft position recovery: a particle closer to the
        surface of a link than the clearance (along the normal of the boundary particle it faces, see
        _task_compute_boundary_density_and_factor) must not approach it, and is moved away by a hundredth of its
        clearance deficit per substep. The recovery moves the position, not the velocity: the velocity constraint only
        removes kinetic energy, while an outward recovery velocity kicked the particles of the first layer that rest
        closer than the clearance at every substep, which kept water at rest in a cup moving (SPH_SOLVER_STATUS.md
        finding 48). The reaction of the corrective impulse acts on the link at the fluid particle, so momentum stays
        balanced, and enters its load readout and, for a coupled link, its impulse of the substep. Carrying the
        velocities to the recovered positions changes the angular momentum of the fluid by sum_i m_i dx_i x v_i, which
        a rigid rotation of the fluid about its center of mass cancels.
        """
        for i_b in range(self._B):
            self.envs_clearance_corrections[i_b] = 0
            self.envs_clearance_mass[i_b] = 0.0
            self.envs_clearance_center[i_b] = qd.Vector.zero(gs.qd_float, 3)
            self.envs_clearance_inertia[i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.envs_clearance_angular[i_b] = qd.Vector.zero(gs.qd_float, 3)
        clearance_min = 0.25 * self._particle_size
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                mass = self.materials[i_m].mass
                for k in range(N_BOUNDARY_SLOTS):
                    i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                    if i_l != -1:
                        normal = self.particles_boundary_slots[i_p, i_b, k].clearance_normal
                        clearance = self.particles_boundary_slots[i_p, i_b, k].clearance
                        vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                        dvel = gs.qd_float(0.0)
                        if clearance < clearance_min:
                            # benchmark controls (SPH_SOLVER_GOAL.md N6, A5): the position recovery and the velocity
                            # constraint can be disabled separately
                            if qd.static(self._clearance_position):
                                shift = 0.01 * (clearance_min - clearance) * normal
                                self.particles_reordered[i_p, i_b].pos += shift
                                self.envs_clearance_angular[i_b] += mass * shift.cross(
                                    self.particles_reordered[i_p, i_b].vel
                                )
                                if qd.static(self._correction_diagnostics):
                                    self.envs_diagnostics[i_b, 13] += shift.norm()
                                    self.envs_diagnostics[i_b, 16] -= mass * self._gravity[i_b].dot(shift)
                            dvel = -vel_rel.dot(normal)
                        if dvel > 0.0 and qd.static(self._clearance_velocity):
                            self.particles_reordered[i_p, i_b].vel += dvel * normal
                            self.particles_boundary_slots[i_p, i_b, k].clearance_impulse = -mass * dvel * normal
                            self.envs_clearance_corrections[i_b] += 1
                            if qd.static(self._correction_diagnostics):
                                self.envs_diagnostics[i_b, 14] += mass * dvel

        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            if qd.static(self._clearance_rotation):
                mass_sum = gs.qd_float(0.0)
                moment = qd.Vector.zero(gs.qd_float, 3)
                for i_k in range(REDUCTION_CHUNK):
                    i_p = i_c * REDUCTION_CHUNK + i_k
                    if i_p < self._n_particles:
                        if self.particles_reordered[i_p, i_b].active:
                            mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                            mass_sum += mass
                            moment += mass * self.particles_reordered[i_p, i_b].pos
                self.envs_clearance_mass[i_b] += mass_sum
                self.envs_clearance_center[i_b] += moment
        for i_b in range(self._B):
            if self.envs_clearance_mass[i_b] > 0.0:
                self.envs_clearance_center[i_b] /= self.envs_clearance_mass[i_b]
        for i_c, i_b in qd.ndrange(self._n_reduction_chunks, self._B):
            inertia = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i_k in range(REDUCTION_CHUNK):
                i_p = i_c * REDUCTION_CHUNK + i_k
                if i_p < self._n_particles:
                    if self.particles_reordered[i_p, i_b].active:
                        mass = self.materials[self.particles_reordered[i_p, i_b].material].mass
                        arm = self.particles_reordered[i_p, i_b].pos - self.envs_clearance_center[i_b]
                        inertia += mass * (arm.norm_sqr() * qd.Matrix.identity(gs.qd_float, 3) - arm.outer_product(arm))
            self.envs_clearance_inertia[i_b] += inertia
        for i_b in range(self._B):
            omega = qd.Vector.zero(gs.qd_float, 3)
            if self.envs_clearance_mass[i_b] > 0.0:
                inertia = self.envs_clearance_inertia[i_b]
                inertia += 1e-6 * inertia.trace() * qd.Matrix.identity(gs.qd_float, 3)
                omega = -(inertia.inverse() @ self.envs_clearance_angular[i_b])
            self.envs_clearance_angular[i_b] = omega
            if qd.static(self._correction_diagnostics):
                self.envs_diagnostics[i_b, 15] = qd.max(self.envs_diagnostics[i_b, 15], omega.norm())
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if qd.static(self._clearance_rotation):
                if self.particles_reordered[i_p, i_b].active:
                    arm = self.particles_reordered[i_p, i_b].pos - self.envs_clearance_center[i_b]
                    self.particles_reordered[i_p, i_b].vel += self.envs_clearance_angular[i_b].cross(arm)

        if qd.static(self._has_rigid_coupling):
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_reordered[i_p, i_b].active:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    for k in range(N_BOUNDARY_SLOTS):
                        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                        if i_l != -1:
                            i_c = self.links_coupled_idx[i_l]
                            impulse = self.particles_boundary_slots[i_p, i_b, k].clearance_impulse
                            if i_c != -1 and impulse.norm_sqr() > 0.0:
                                torque = (pos_i - self.links_root_com[i_l, i_b]).cross(impulse)
                                for j in qd.static(range(3)):
                                    qd.atomic_add(self.coupled_impulse_den[6 * i_c + j, i_b], torque[j])
                                    qd.atomic_add(self.coupled_impulse_den[6 * i_c + 3 + j, i_b], impulse[j])

    # ------------------------------------------------------------------------------------
    # ---------------------------------- rigid coupling ----------------------------------
    # ------------------------------------------------------------------------------------
    # The fluid stage of a substep runs inside the substep of the rigid solver, between its forward dynamics and its
    # constraint solve (SPH_SOLVER_DESIGN.md section 4): the pressure and viscosity solves see the response of the coupled
    # links through their mobility, and their net impulse enters the smooth force of the rigid solver, so that the
    # constraint solve and the integration of the rigid solver take the fluid load of the same substep.

    @qd.kernel
    def _kernel_set_coupled_links(
        self,
        links_coupled_idx: qd.types.ndarray(),
        coupled_links_idx: qd.types.ndarray(),
        coupled_links_extent: qd.types.ndarray(),
    ):
        for i_l in range(self.links_coupled_idx.shape[0]):
            self.links_coupled_idx[i_l] = links_coupled_idx[i_l]
        for i_c in range(self.coupled_links_idx.shape[0]):
            self.coupled_links_idx[i_c] = coupled_links_idx[i_c]
            self.coupled_links_extent[i_c] = coupled_links_extent[i_c]

    @qd.func
    def _func_chain_spatial_vel(
        self,
        i_c: int,
        i_b: int,
        vec: qd.Tensor,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_config: qd.template(),
    ):
        """Return the spatial velocity (angular and linear part) of coupled link i_c for the dof velocities vec."""
        ang = qd.Vector.zero(gs.qd_float, 3)
        lin = qd.Vector.zero(gs.qd_float, 3)
        i_l = self.coupled_links_idx[i_c]
        for _ in range(self.links_coupled_idx.shape[0]):
            if i_l != -1:
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    ang += dyn_state.dofs.cdof_ang[i_d, i_b] * vec[i_d, i_b]
                    lin += dyn_state.dofs.cdof_vel[i_d, i_b] * vec[i_d, i_b]
                i_l = dyn_info.links.parent_idx[I_l]
        return ang, lin

    @qd.func
    def _func_chain_add_wrench(
        self,
        i_c: int,
        i_b: int,
        torque: qd.types.vector(3),
        force: qd.types.vector(3),
        vec: qd.Tensor,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_config: qd.template(),
    ):
        """Add to the dof forces vec the generalized force of a wrench on coupled link i_c (torque about the center of
        mass of its kinematic tree)."""
        i_l = self.coupled_links_idx[i_c]
        for _ in range(self.links_coupled_idx.shape[0]):
            if i_l != -1:
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    vec[i_d, i_b] += dyn_state.dofs.cdof_ang[i_d, i_b].dot(torque) + dyn_state.dofs.cdof_vel[
                        i_d, i_b
                    ].dot(force)
                i_l = dyn_info.links.parent_idx[I_l]

    @qd.kernel
    def _kernel_init_rigid_coupling(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
    ):
        """Compute the mobility of the coupled links and take their spatial velocities at the start of the substep.

        Column (c, s) of the mobility is the spatial acceleration of every coupled link for a unit wrench component s
        on link c, solved with the mass matrix factor of the forward dynamics of the substep.
        """
        n_coupled = self.coupled_links_idx.shape[0]
        n_dofs = self.dofs_coupling_rhs.shape[0]
        for i_b in range(self._B):
            for i_col in range(6 * n_coupled):
                i_c = i_col // 6
                i_s = i_col % 6
                for i_d in range(n_dofs):
                    self.dofs_coupling_rhs[i_d, i_b] = 0.0
                torque = qd.Vector(
                    [qd.cast(i_s == 0, gs.qd_float), qd.cast(i_s == 1, gs.qd_float), qd.cast(i_s == 2, gs.qd_float)]
                )
                force = qd.Vector(
                    [qd.cast(i_s == 3, gs.qd_float), qd.cast(i_s == 4, gs.qd_float), qd.cast(i_s == 5, gs.qd_float)]
                )
                self._func_chain_add_wrench(
                    i_c, i_b, torque, force, self.dofs_coupling_rhs, dyn_state, dyn_info, rigid_config
                )
                func_solve_mass_batch(
                    i_b, self.dofs_coupling_rhs, self.dofs_coupling_sol, dyn_state, rigid_info, rigid_config
                )
                for i_r in range(n_coupled):
                    ang, lin = self._func_chain_spatial_vel(
                        i_r, i_b, self.dofs_coupling_sol, dyn_state, dyn_info, rigid_config
                    )
                    for k in qd.static(range(3)):
                        self.coupled_mobility[6 * i_r + k, i_col, i_b] = ang[k]
                        self.coupled_mobility[6 * i_r + 3 + k, i_col, i_b] = lin[k]

        for i_c, i_b in qd.ndrange(n_coupled, self._B):
            i_l = self.coupled_links_idx[i_c]
            for k in qd.static(range(3)):
                self.coupled_vel[6 * i_c + k, i_b] = dyn_state.links.cd_ang[i_l, i_b][k]
                self.coupled_vel[6 * i_c + 3 + k, i_b] = dyn_state.links.cd_vel[i_l, i_b][k]
            for k in range(6):
                self.coupled_vel_base[6 * i_c + k, i_b] = self.coupled_vel[6 * i_c + k, i_b]
                self.coupled_impulse_div[6 * i_c + k, i_b] = 0.0
                self.coupled_impulse_visc[6 * i_c + k, i_b] = 0.0
                self.coupled_impulse_den[6 * i_c + k, i_b] = 0.0
                self.coupled_vel_delta[6 * i_c + k, i_b] = 0.0

    @qd.func
    def _func_set_coupled_links_vel(self, i_c, i_b):
        """Copy the spatial velocity the fluid sees for coupled link i_c into the velocity fields of the links."""
        i_l = self.coupled_links_idx[i_c]
        self.links_ang[i_l, i_b] = qd.Vector(
            [self.coupled_vel[6 * i_c, i_b], self.coupled_vel[6 * i_c + 1, i_b], self.coupled_vel[6 * i_c + 2, i_b]]
        )
        self.links_vel[i_l, i_b] = qd.Vector(
            [self.coupled_vel[6 * i_c + 3, i_b], self.coupled_vel[6 * i_c + 4, i_b], self.coupled_vel[6 * i_c + 5, i_b]]
        )

    @qd.kernel
    def _kernel_set_coupled_vel_star(
        self, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()
    ):
        """Set the coupled velocities to their value after the divergence impulse, the smooth acceleration and the
        velocity change of the constraints (see repeat_for_rigid_constraints)."""
        for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
            ang, lin = self._func_chain_spatial_vel(
                i_c, i_b, dyn_state.dofs.acc_smooth, dyn_state, dyn_info, rigid_config
            )
            for k in qd.static(range(3)):
                self.coupled_vel[6 * i_c + k, i_b] += (
                    self._substep_dt * ang[k] + self.coupled_vel_constraint[6 * i_c + k, i_b]
                )
                self.coupled_vel[6 * i_c + 3 + k, i_b] += (
                    self._substep_dt * lin[k] + self.coupled_vel_constraint[6 * i_c + 3 + k, i_b]
                )
            for k in range(6):
                self.coupled_vel_star[6 * i_c + k, i_b] = self.coupled_vel[6 * i_c + k, i_b]
            self._func_set_coupled_links_vel(i_c, i_b)

    @qd.kernel
    def _kernel_set_coupled_vel_base(self, with_viscosity: bool):
        """Set the coupled velocities to the base of the density solve: v* plus the response to the viscous impulse."""
        n_coupled_6 = 6 * self.coupled_links_idx.shape[0]
        for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
            if self.envs_coupling_active[i_b]:
                vel = self.coupled_vel_star[i_r, i_b]
                if with_viscosity:
                    for i_col in range(n_coupled_6):
                        vel += self.coupled_mobility[i_r, i_col, i_b] * self.coupled_impulse_visc[i_col, i_b]
                self.coupled_vel_base[i_r, i_b] = vel
                self.coupled_vel[i_r, i_b] = vel
        for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
            if self.envs_coupling_active[i_b]:
                self._func_set_coupled_links_vel(i_c, i_b)

    @qd.func
    def _func_add_coupled_wrench(self, i_p, i_b, mass_kappa):
        """Add the reaction of the boundary pressure of fluid particle i_p to the wrenches of the coupled links."""
        pos_i = self.particles_reordered[i_p, i_b].pos
        for k in range(N_BOUNDARY_SLOTS):
            i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
            if i_l != -1:
                i_c = self.links_coupled_idx[i_l]
                if i_c != -1:
                    force = mass_kappa * self.particles_boundary_slots[i_p, i_b, k].grad
                    torque = (pos_i - self.links_root_com[i_l, i_b]).cross(force)
                    for j in qd.static(range(3)):
                        self.coupled_wrench[6 * i_c + j, i_b] += torque[j]
                        self.coupled_wrench[6 * i_c + 3 + j, i_b] += force[j]

    @qd.func
    def _func_coupled_point_acc(self, i_p, i_b, k):
        """Return the acceleration of the link of boundary slot k of fluid particle i_p at the particle, due to the
        pressure of the current iteration (zero for a link that is not coupled)."""
        res = qd.Vector.zero(gs.qd_float, 3)
        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
        i_c = self.links_coupled_idx[i_l]
        if i_c != -1:
            ang = qd.Vector(
                [self.coupled_acc[6 * i_c, i_b], self.coupled_acc[6 * i_c + 1, i_b], self.coupled_acc[6 * i_c + 2, i_b]]
            )
            lin = qd.Vector(
                [
                    self.coupled_acc[6 * i_c + 3, i_b],
                    self.coupled_acc[6 * i_c + 4, i_b],
                    self.coupled_acc[6 * i_c + 5, i_b],
                ]
            )
            res = lin + ang.cross(self.particles_reordered[i_p, i_b].pos - self.links_root_com[i_l, i_b])
        return res

    @qd.kernel
    def _kernel_solve_coupled_viscosity(self) -> float:
        """Take the wall viscous impulse of the pass on the coupled links and solve their implicit velocities.

        The wall dissipation of the slots (see _kernel_compute_viscosity_geometry) is quadratic in the spatial
        velocities xi of the coupled links: the wrench of the fluid on them is f - K xi, with K the sum over the slots of
        G^T M G and f that of G^T M v_i, G mapping a spatial velocity to the velocity at the fluid particle. The
        impulse the links take is the reaction of the one the fluid took in its viscosity solve, h (f - K xi_f), at the
        velocities xi_f the fluid saw, so that the exchange balances linear and angular momentum exactly at every
        pass, converged or not. The backward Euler step of the links, xi = xi* + Lambda (I_den + h (f - K xi)), is the
        dense system (I + h Lambda K) xi = xi* + Lambda I_den + h Lambda f, solved per environment by Gaussian
        elimination (its matrix is similar to a symmetric positive definite one). Its solution differs from the
        velocities xi* + Lambda (I_den + I_visc) that the impulses give by a correction, which the fluid adds to the
        velocities it sees in the next pass (see _kernel_apply_coupled_vel_delta), so that a light link stays stable
        under a strong viscous load. The passes have converged once xi_f agrees with the implicit solution. Returns the
        largest disagreement over envs, as the norm of the linear velocity plus the size of the link times the norm of
        the angular velocity, relative to the same norm of the solution, with a floor at the velocity gravity gives in
        a substep.
        """
        n_coupled = self.coupled_links_idx.shape[0]
        n_coupled_6 = 6 * n_coupled
        for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
            if self.envs_coupling_active[i_b]:
                self.coupled_visc_force[i_r, i_b] = 0.0
                for j in range(6):
                    self.coupled_visc_stiffness[i_r, j, i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_coupling_active[i_b]:
                if self.particles_reordered[i_p, i_b].active:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    vel_i = self.particles_reordered[i_p, i_b].vel
                    for k in range(N_BOUNDARY_SLOTS):
                        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                        if i_l != -1:
                            i_c = self.links_coupled_idx[i_l]
                            if i_c != -1:
                                tensor = self.particles_boundary_slots[i_p, i_b, k].wall_tensor
                                r = pos_i - self.links_root_com[i_l, i_b]
                                # G = [-[r]x, I] maps (angular, linear) to the point velocity, G^T y = (r x y, y)
                                cross_r = qd.Matrix([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
                                force = tensor @ vel_i
                                torque = r.cross(force)
                                block_aa = -(cross_r @ tensor @ cross_r)
                                block_al = cross_r @ tensor
                                for a in qd.static(range(3)):
                                    self.coupled_visc_force[6 * i_c + a, i_b] += torque[a]
                                    self.coupled_visc_force[6 * i_c + 3 + a, i_b] += force[a]
                                    for b in qd.static(range(3)):
                                        self.coupled_visc_stiffness[6 * i_c + a, b, i_b] += block_aa[a, b]
                                        self.coupled_visc_stiffness[6 * i_c + a, 3 + b, i_b] += block_al[a, b]
                                        self.coupled_visc_stiffness[6 * i_c + 3 + a, b, i_b] += block_al[b, a]
                                        self.coupled_visc_stiffness[6 * i_c + 3 + a, 3 + b, i_b] += tensor[a, b]

        max_change = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_coupling_active[i_b]:
                h = self._substep_dt
                # system matrix and right-hand side, Lambda K being block-structured by the column link
                for i_r in range(n_coupled_6):
                    rhs = self.coupled_vel_star[i_r, i_b]
                    for i_col in range(n_coupled_6):
                        rhs += self.coupled_mobility[i_r, i_col, i_b] * (
                            self.coupled_impulse_den[i_col, i_b] + h * self.coupled_visc_force[i_col, i_b]
                        )
                    self.coupled_system[i_r, n_coupled_6, i_b] = rhs
                    for i_col in range(n_coupled_6):
                        i_c_col = i_col // 6
                        value = gs.qd_float(0.0)
                        for j in range(6):
                            value += (
                                self.coupled_mobility[i_r, 6 * i_c_col + j, i_b]
                                * self.coupled_visc_stiffness[6 * i_c_col + j, i_col % 6, i_b]
                            )
                        self.coupled_system[i_r, i_col, i_b] = h * value + qd.cast(i_r == i_col, gs.qd_float)
                for i_k in range(n_coupled_6):
                    pivot = self.coupled_system[i_k, i_k, i_b]
                    for i_r in range(i_k + 1, n_coupled_6):
                        ratio = self.coupled_system[i_r, i_k, i_b] / pivot
                        for i_col in range(i_k, n_coupled_6 + 1):
                            self.coupled_system[i_r, i_col, i_b] -= ratio * self.coupled_system[i_k, i_col, i_b]
                for i_k_ in range(n_coupled_6):
                    i_k = n_coupled_6 - 1 - i_k_
                    value = self.coupled_system[i_k, n_coupled_6, i_b]
                    for i_col in range(i_k + 1, n_coupled_6):
                        value -= self.coupled_system[i_k, i_col, i_b] * self.coupled_system[i_col, n_coupled_6, i_b]
                    value = value / self.coupled_system[i_k, i_k, i_b]
                    self.coupled_system[i_k, n_coupled_6, i_b] = value
                # viscous impulse h (f - K xi_f) on each link, with its own block of K, at the velocities the fluid saw
                for i_r in range(n_coupled_6):
                    i_c = i_r // 6
                    value = self.coupled_visc_force[i_r, i_b]
                    for j in range(6):
                        value -= self.coupled_visc_stiffness[i_r, j, i_b] * self.coupled_vel[6 * i_c + j, i_b]
                    self.coupled_impulse_visc[i_r, i_b] = h * value
                # implicit correction: the implicit solution minus the velocities the impulses give
                for i_r in range(n_coupled_6):
                    vel = self.coupled_vel_star[i_r, i_b]
                    for i_col in range(n_coupled_6):
                        vel += self.coupled_mobility[i_r, i_col, i_b] * (
                            self.coupled_impulse_den[i_col, i_b] + self.coupled_impulse_visc[i_col, i_b]
                        )
                    self.coupled_vel_delta[i_r, i_b] = self.coupled_system[i_r, n_coupled_6, i_b] - vel
                change = gs.qd_float(0.0)
                vel_scale = self._substep_dt * self._gravity[i_b].norm()
                for i_c in range(n_coupled):
                    extent = self.coupled_links_extent[i_c]
                    ang_change, lin_change = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                    ang, lin = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                    for k in qd.static(range(3)):
                        ang[k] = self.coupled_system[6 * i_c + k, n_coupled_6, i_b]
                        lin[k] = self.coupled_system[6 * i_c + 3 + k, n_coupled_6, i_b]
                        ang_change[k] = ang[k] - self.coupled_vel[6 * i_c + k, i_b]
                        lin_change[k] = lin[k] - self.coupled_vel[6 * i_c + 3 + k, i_b]
                    change = qd.max(change, lin_change.norm() + extent * ang_change.norm())
                    vel_scale = qd.max(vel_scale, lin.norm() + extent * ang.norm())
                self.envs_coupled_vel_change[i_b] = change / qd.max(vel_scale, gs.EPS)
                qd.atomic_max(max_change, self.envs_coupled_vel_change[i_b])
        return max_change

    @qd.kernel
    def _kernel_apply_coupled_vel_delta(self):
        """Add the implicit correction of the last pass to the coupled velocities that the next viscosity solve sees."""
        for i_r, i_b in qd.ndrange(6 * self.coupled_links_idx.shape[0], self._B):
            if self.envs_coupling_active[i_b]:
                self.coupled_vel[i_r, i_b] += self.coupled_vel_delta[i_r, i_b]
        for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
            if self.envs_coupling_active[i_b]:
                self._func_set_coupled_links_vel(i_c, i_b)

    @qd.kernel
    def _kernel_apply_coupled_impulse(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
    ):
        """Add the net impulse of the fluid of the substep to the smooth force and acceleration of the rigid solver."""
        n_coupled = self.coupled_links_idx.shape[0]
        n_dofs = self.dofs_coupling_rhs.shape[0]
        inv_dt = 1.0 / self._substep_dt
        for i_b in range(self._B):
            for i_d in range(n_dofs):
                self.dofs_coupling_rhs[i_d, i_b] = 0.0
            for i_c in range(n_coupled):
                torque = qd.Vector.zero(gs.qd_float, 3)
                force = qd.Vector.zero(gs.qd_float, 3)
                for j in qd.static(range(3)):
                    torque[j] = inv_dt * (
                        self.coupled_impulse_div[6 * i_c + j, i_b]
                        + self.coupled_impulse_visc[6 * i_c + j, i_b]
                        + self.coupled_impulse_den[6 * i_c + j, i_b]
                    )
                    force[j] = inv_dt * (
                        self.coupled_impulse_div[6 * i_c + 3 + j, i_b]
                        + self.coupled_impulse_visc[6 * i_c + 3 + j, i_b]
                        + self.coupled_impulse_den[6 * i_c + 3 + j, i_b]
                    )
                self._func_chain_add_wrench(
                    i_c, i_b, torque, force, self.dofs_coupling_rhs, dyn_state, dyn_info, rigid_config
                )
            func_solve_mass_batch(
                i_b, self.dofs_coupling_rhs, self.dofs_coupling_sol, dyn_state, rigid_info, rigid_config
            )
            for i_d in range(n_dofs):
                self.dofs_fluid_force[i_d, i_b] = self.dofs_coupling_rhs[i_d, i_b]
                self.dofs_fluid_acc[i_d, i_b] = self.dofs_coupling_sol[i_d, i_b]
                dyn_state.dofs.force[i_d, i_b] += self.dofs_coupling_rhs[i_d, i_b]
                dyn_state.dofs.qf_smooth[i_d, i_b] += self.dofs_coupling_rhs[i_d, i_b]
                dyn_state.dofs.acc_smooth[i_d, i_b] += self.dofs_coupling_sol[i_d, i_b]

    @qd.kernel
    def _kernel_constraint_velocity_change(
        self,
        dyn_state: array_class.DynState,
        constraint_state: array_class.ConstraintState,
        dyn_info: array_class.DynInfo,
        rigid_config: qd.template(),
    ) -> float:
        """Take the velocity change of the coupled links by the constraint solve of the substep, h S (qacc -
        acc_smooth), as the external velocity change of a repeated fluid stage. qacc is the acceleration the constraint
        solver just found, which the rigid solver copies to the dofs only after the repetitions. Returns the largest
        change of it over envs since the stage last ran, in the norm of _kernel_solve_coupled_viscosity, relative to
        the link velocities, with a floor at the velocity gravity gives in a substep."""
        n_coupled = self.coupled_links_idx.shape[0]
        n_dofs = self.dofs_coupling_rhs.shape[0]
        for i_d, i_b in qd.ndrange(n_dofs, self._B):
            self.dofs_coupling_rhs[i_d, i_b] = constraint_state.qacc[i_d, i_b] - dyn_state.dofs.acc_smooth[i_d, i_b]
        for i_b in range(self._B):
            self.envs_constraint_residual[i_b] = 0.0
        for i_b in range(self._B):
            change = gs.qd_float(0.0)
            vel_scale = self._substep_dt * self._gravity[i_b].norm()
            for i_c in range(n_coupled):
                ang, lin = self._func_chain_spatial_vel(
                    i_c, i_b, self.dofs_coupling_rhs, dyn_state, dyn_info, rigid_config
                )
                extent = self.coupled_links_extent[i_c]
                ang_change = qd.Vector.zero(gs.qd_float, 3)
                lin_change = qd.Vector.zero(gs.qd_float, 3)
                ang_vel = qd.Vector.zero(gs.qd_float, 3)
                lin_vel = qd.Vector.zero(gs.qd_float, 3)
                for k in qd.static(range(3)):
                    ang_change[k] = self._substep_dt * ang[k] - self.coupled_vel_constraint[6 * i_c + k, i_b]
                    lin_change[k] = self._substep_dt * lin[k] - self.coupled_vel_constraint[6 * i_c + 3 + k, i_b]
                    self.coupled_vel_constraint[6 * i_c + k, i_b] = self._substep_dt * ang[k]
                    self.coupled_vel_constraint[6 * i_c + 3 + k, i_b] = self._substep_dt * lin[k]
                    ang_vel[k] = self.coupled_vel[6 * i_c + k, i_b]
                    lin_vel[k] = self.coupled_vel[6 * i_c + 3 + k, i_b]
                change = qd.max(change, lin_change.norm() + extent * ang_change.norm())
                vel_scale = qd.max(vel_scale, lin_vel.norm() + extent * ang_vel.norm())
            self.envs_constraint_residual[i_b] = change / qd.max(vel_scale, gs.EPS)
        max_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            qd.atomic_max(max_residual, self.envs_constraint_residual[i_b])
        return max_residual

    @qd.kernel
    def _kernel_undo_coupled_impulse(self, dyn_state: array_class.DynState):
        """Take back the fluid impulse the last fluid stage put in the smooth force and acceleration of the dofs."""
        for i_d, i_b in qd.ndrange(self.dofs_fluid_force.shape[0], self._B):
            dyn_state.dofs.force[i_d, i_b] -= self.dofs_fluid_force[i_d, i_b]
            dyn_state.dofs.qf_smooth[i_d, i_b] -= self.dofs_fluid_force[i_d, i_b]
            dyn_state.dofs.acc_smooth[i_d, i_b] -= self.dofs_fluid_acc[i_d, i_b]

    @qd.kernel
    def _kernel_save_readouts(self, is_restore: bool):
        """Save the load and dissipation readouts at the start of a fluid stage, or restore them for a repetition."""
        for i_l, i_b in qd.ndrange(self.links_fluid_impulse.shape[0], self._B):
            if is_restore:
                self.links_fluid_impulse[i_l, i_b] = self.links_fluid_impulse_saved[i_l, i_b]
                self.links_fluid_angular_impulse[i_l, i_b] = self.links_fluid_angular_impulse_saved[i_l, i_b]
            else:
                self.links_fluid_impulse_saved[i_l, i_b] = self.links_fluid_impulse[i_l, i_b]
                self.links_fluid_angular_impulse_saved[i_l, i_b] = self.links_fluid_angular_impulse[i_l, i_b]
            for i_q in qd.static(range(N_LOAD_PARTS)):
                if is_restore:
                    self.links_fluid_part_impulse[i_l, i_b, i_q] = self.links_fluid_part_impulse_saved[i_l, i_b, i_q]
                    self.links_fluid_part_angular_impulse[i_l, i_b, i_q] = self.links_fluid_part_angular_impulse_saved[
                        i_l, i_b, i_q
                    ]
                else:
                    self.links_fluid_part_impulse_saved[i_l, i_b, i_q] = self.links_fluid_part_impulse[i_l, i_b, i_q]
                    self.links_fluid_part_angular_impulse_saved[i_l, i_b, i_q] = self.links_fluid_part_angular_impulse[
                        i_l, i_b, i_q
                    ]
        if qd.static(self._has_viscosity):
            for i_b in range(self._B):
                if is_restore:
                    self.envs_viscous_dissipation[i_b] = self.envs_viscous_dissipation_saved[i_b]
                else:
                    self.envs_viscous_dissipation_saved[i_b] = self.envs_viscous_dissipation[i_b]

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def cubic_kernel(self, r_norm):
        """
        Cubic spline smoothing kernel.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q**2
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = 2 * k * (1.0 - q) ** 3
        return res

    @qd.func
    def cubic_kernel_derivative(self, r):
        """
        Derivative of cubic spline smoothing kernel.
        """
        res = qd.Vector.zero(gs.qd_float, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
            else:
                res = -6.0 * k * (1.0 - q) ** 2 * grad_q
        return res

    @qd.func
    def cubic_kernel_plane_integral(self, dist):
        """
        Integral of the cubic spline smoothing kernel over the plane at distance dist from its center.

        Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr. Its own integral along the plane normal equals
        half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
        forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        q = qd.abs(dist) / h
        if q <= 1.0:
            q2 = q**2
            q3 = q2 * q
            if q <= 0.5:
                res = 0.0875 - (1.2 * q3 * q2 - 1.5 * q2**2 + 0.5 * q2)
            else:
                res = 0.1 - (q2 - 2.0 * q3 + 1.5 * q2**2 - 0.4 * q3 * q2)
            res = res * 16.0 / h
        return res

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    @property
    def is_rigid_coupled(self):
        """Whether the fluid stage of a substep runs inside the substep of the rigid solver (see rigid coupling)."""
        return self.is_active and self._has_rigid_coupling

    def substep_coupled_with_rigid(self, f):
        """Run the fluid stage of a substep from the substep of the rigid solver, after its forward dynamics."""
        self._n_constraint_passes = 0
        self._kernel_save_readouts(False)
        self._run_fluid_stage(f)

    def repeat_for_rigid_constraints(self, f):
        """Repeat the fluid stage of the substep if the constraint solve of the rigid solver changed the coupled links.

        Called by the rigid solver after its constraint solve (SPH_SOLVER_DESIGN.md section 6). The constraint solve
        changes the velocity of the coupled links by h S (qacc - acc_smooth), which the fluid stage did not see. When it
        differs from the change the stage held, by more than `max_constraint_error` of the link velocities, the fluid
        impulse is taken back from the rigid solver, the readouts are restored and the stage runs again with that change
        as an external velocity change of the links; the rigid solver then solves its constraints again. The change is
        kept as the guess of the next substep, so a steady contact repeats nothing. Returns whether the stage ran again.
        """
        rigid_solver = self.sim.rigid_solver
        stats = self._solve_stats
        residual = self._kernel_constraint_velocity_change(
            rigid_solver.dyn_state,
            rigid_solver.constraint_solver.constraint_state,
            rigid_solver.dyn_info,
            rigid_solver.rigid_config,
        )
        if residual <= self._max_constraint_error:
            stats.max_constraint_residual = max(stats.max_constraint_residual, residual)
            return False
        if self._n_constraint_passes >= self._max_constraint_iters:
            stats.n_unconverged_constraint += 1
            stats.max_constraint_residual = max(stats.max_constraint_residual, residual)
            return False
        self._n_constraint_passes += 1
        stats.n_constraint_passes += 1
        self._kernel_undo_coupled_impulse(rigid_solver.dyn_state)
        self._kernel_save_readouts(True)
        self._run_fluid_stage(f, is_repetition=True)
        return True

    def substep_pre_coupling(self, f):
        if self.is_active and not self._has_rigid_coupling:
            self._run_fluid_stage(f)

    def _run_fluid_stage(self, f, is_repetition=False):
        """Run the boundary update, the divergence solve, the explicit forces and the coupled viscosity and density
        solves of a substep (SPH_SOLVER_DESIGN.md section 4), and the transfer of the fluid impulse to the coupled
        links."""
        # f indexes the substep in the memory of the simulator, so the first substep of a step is counted globally.
        is_step_start = self._sim.cur_substep_global % self._sim.substeps == 0
        if is_step_start and not is_repetition:
            self._solve_stats = SPHSolveStats()
        stats = self._solve_stats
        if not is_repetition:
            stats.n_substeps += 1

        rigid_solver = self.sim.rigid_solver
        if self._has_boundary:
            self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
        if self._has_rigid_coupling:
            self._kernel_init_rigid_coupling(
                rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info, rigid_solver.rigid_config
            )
        self._kernel_reorder_particles(f)
        self._kernel_activate_envs()
        self._kernel_compute_density_and_factor(f)
        if self._isotropic_pressure:
            self._kernel_compute_pressure_correction()
            if self._isotropic_density:
                self._kernel_compute_corrected_density()
        energy = self._kernel_kinetic_energy() if self._energy_audit else 0.0
        if not self._df_skip_div:
            n_iters, residual = self._pressure_solve(is_divergence=True)
            stats.n_divergence_iters += n_iters
            stats.n_unconverged_divergence += residual > 1.0
            stats.max_divergence_residual = max(stats.max_divergence_residual, residual)
            if self._energy_audit:
                energy, energy_prev = self._kernel_kinetic_energy(), energy
                stats.ke_change_divergence += energy - energy_prev
        self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
        if self._energy_audit:
            energy, energy_prev = self._kernel_kinetic_energy(), energy
            stats.ke_change_forces += energy - energy_prev
        if self._has_rigid_coupling:
            self._kernel_set_coupled_vel_star(rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_config)

        # Coupled viscosity and density solves, alternated until they agree (SPH_SOLVER_DESIGN.md section 5.2).
        n_passes = 1 if not self._has_viscosity else self._max_coupling_iters
        if self._correction_diagnostics:
            self._kernel_reset_diagnostics()
        if self._has_viscosity:
            self._kernel_compute_viscosity_geometry()
            if self._correction_diagnostics:
                self._kernel_viscosity_diagnostics()
            stats.coupling_history.append([])
        i_chebyshev, previous_coupling_residual = 0, math.inf
        for i_pass in range(n_passes):
            rigid_change = 0.0
            n_visc_iters = 0
            if self._has_viscosity:
                if self._has_rigid_coupling and i_pass > 0:
                    self._kernel_apply_coupled_vel_delta()
                n_iters, residual = self._viscosity_solve(is_first_pass=i_pass == 0)
                n_visc_iters = n_iters
                stats.n_viscosity_iters += n_iters
                stats.n_unconverged_viscosity += residual > 1.0
                stats.max_viscosity_residual = max(stats.max_viscosity_residual, residual)
                if self._has_rigid_coupling:
                    rigid_change = self._kernel_solve_coupled_viscosity()
                if i_pass > 0:
                    self._kernel_remove_pressure_vel()
                self._kernel_store_coupling_pressure(i_pass == 0)
            if self._has_rigid_coupling:
                self._kernel_set_coupled_vel_base(self._has_viscosity)
            # The density solves of the first two coupling passes are accelerated. A substep that needs more passes,
            # under a strong viscosity-pressure coupling, continues with plain solves, since the coupling iteration,
            # accelerated itself, may stall on accelerated density solves (SPH_SOLVER_STATUS.md finding 15).
            n_iters, residual = self._pressure_solve(
                is_divergence=False, keep_pressure=i_pass > 0, accelerate=i_pass < 2
            )
            stats.n_density_iters += n_iters
            stats.n_unconverged_density += residual > 1.0
            stats.max_density_residual = max(stats.max_density_residual, residual)
            stats.n_coupling_passes += 1
            if self._has_viscosity:
                # The rigid velocities converge with the fluid ones: their change over the pass, relative to their
                # magnitude, has the tolerance of the viscosity solve.
                fluid_residual = self._kernel_coupling_residual()
                coupling_residual = max(fluid_residual, rigid_change / self._visc_max_error)
                self._kernel_update_coupling_active()
                stats.coupling_history[-1].append(
                    (fluid_residual, rigid_change / self._visc_max_error, n_iters, n_visc_iters)
                )
                gs.logger.debug(
                    f"SPH coupling pass {i_pass}: viscosity {stats.n_viscosity_iters} iterations in total, density "
                    f"residual {residual:.3g}, coupling residual {coupling_residual:.3g}"
                )
                if coupling_residual <= 1.0 or i_pass == n_passes - 1:
                    stats.n_unconverged_coupling += coupling_residual > 1.0
                    stats.max_coupling_residual = max(stats.max_coupling_residual, coupling_residual)
                    break
                # The Chebyshev acceleration assumes a fixed linear map, which the clamp of the pressure and the response
                # of the coupled links may break; a pass that raises the disagreement restarts it from a plain step.
                if i_pass > 0 and coupling_residual > previous_coupling_residual:
                    i_chebyshev = 0
                self._kernel_accelerate_coupling(i_chebyshev)
                i_chebyshev += 1
                previous_coupling_residual = coupling_residual

        self._kernel_activate_envs()
        if self._energy_audit:
            energy, energy_prev = self._kernel_kinetic_energy(), energy
            stats.ke_change_coupled += energy - energy_prev
        if self._has_boundary:
            self._kernel_enforce_wall_clearance()
            stats.n_clearance_corrections += int(qd_to_torch(self.envs_clearance_corrections).sum())
            if self._energy_audit:
                energy, energy_prev = self._kernel_kinetic_energy(), energy
                stats.ke_change_clearance += energy - energy_prev
        if self._has_rigid_coupling:
            self._kernel_apply_coupled_impulse(
                rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info, rigid_solver.rigid_config
            )
        if self._has_viscosity:
            self._kernel_accumulate_viscous_dissipation(is_step_start)
        if self._correction_diagnostics:
            self._accumulate_diagnostics(stats)
        if self._has_boundary:
            self._kernel_accumulate_boundary_load(is_step_start)
            self._solve_stats.n_boundary_slot_overflows += int(qd_to_torch(self.envs_boundary_slot_overflows).sum())

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            energy = self._kernel_kinetic_energy() if self._energy_audit else 0.0
            if self._position_drift:
                self._drift_correction()
            self._kernel_advect_position(f)
            if self._energy_audit:
                self._solve_stats.ke_change_bounds += self._kernel_kinetic_energy() - energy
            self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        """Restore the particles, their warm starts and micro-rotations, and the constraint guess of the coupled links,
        in the environments envs_idx (all by default), so that the simulation continues as from the saved state."""
        if self.is_active:
            envs_mask = torch.zeros((self._B,), dtype=gs.tc_int, device=gs.device)
            envs_mask[self._scene._sanitize_envs_idx(envs_idx)] = 1
            self._kernel_set_state(f, state.pos, state.vel, state.active, state.pressure, state.angular_vel, envs_mask)
            if self._has_rigid_coupling:
                self._kernel_set_coupled_vel_constraint(state.coupled_vel_constraint, envs_mask)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
        pressure: qd.types.ndarray(),
        angular_vel: qd.types.ndarray(),
        envs_mask: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if envs_mask[i_b]:
                for j in qd.static(range(3)):
                    self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                    self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
                    if qd.static(self._micropolar):
                        self.particles[i_p, i_b].angular_vel[j] = angular_vel[i_b, i_p, j]
                self.particles[i_p, i_b].active = active[i_b, i_p]
                self.particles[i_p, i_b].pressure_density2_norm_p = pressure[i_b, i_p, 0]
                self.particles[i_p, i_b].pressure_density2_norm_d = pressure[i_b, i_p, 1]

    @qd.kernel
    def _kernel_set_coupled_vel_constraint(self, vel: qd.types.ndarray(), envs_mask: qd.types.ndarray()):
        for i_r, i_b in qd.ndrange(self.coupled_vel_constraint.shape[0], self._B):
            if envs_mask[i_b]:
                self.coupled_vel_constraint[i_r, i_b] = vel[i_b, i_r]

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active, state.pressure, state.angular_vel)
            if self._has_rigid_coupling:
                self._kernel_get_coupled_vel_constraint(state.coupled_vel_constraint)
        else:
            state = None
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
        pressure: qd.types.ndarray(),
        angular_vel: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
                if qd.static(self._micropolar):
                    angular_vel[i_b, i_p, j] = self.particles[i_p, i_b].angular_vel[j]
            active[i_b, i_p] = self.particles[i_p, i_b].active
            pressure[i_b, i_p, 0] = self.particles[i_p, i_b].pressure_density2_norm_p
            pressure[i_b, i_p, 1] = self.particles[i_p, i_b].pressure_density2_norm_d

    @qd.kernel
    def _kernel_get_coupled_vel_constraint(self, vel: qd.types.ndarray()):
        for i_r, i_b in qd.ndrange(self.coupled_vel_constraint.shape[0], self._B):
            vel[i_b, i_r] = self.coupled_vel_constraint[i_r, i_b]

    @property
    def n_coupled_dofs(self):
        """Number of spatial velocity components of the coupled links (6 per link), 0 without coupled links."""
        return 6 * len(self._coupled_links_idx) if self._has_rigid_coupling else 0

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles[i_p, i_b].active

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        material_idx: qd.i32,
        pos: qd.types.ndarray(),
        material: array_class.SPHMaterialInfo,
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].material = material_idx
            self.particles[i_p, i_b].pressure_density2_norm_p = 0.0
            self.particles[i_p, i_b].pressure_density2_norm_d = 0.0
            if qd.static(self._micropolar):
                self.particles[i_p, i_b].angular_vel = qd.Vector.zero(gs.qd_float, 3)

        self.materials[material_idx].rest_volume = self._particle_volume
        self.materials[material_idx].mass = self._particle_volume * material.rest_density
        self.materials[material_idx].rest_density = material.rest_density
        self.materials[material_idx].dynamic_viscosity = material.rest_density * material.viscosity
        self.materials[material_idx].surface_tension = material.surface_tension
        self.materials[material_idx].vorticity = material.vorticity
        self.materials[material_idx].angular_viscosity = material.angular_viscosity
        self.materials[material_idx].inverse_microinertia = 1.0 / material.microinertia

    # ----------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)
            self.particles[i_p, i_b].acc.fill(0.0)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
            self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].active = actives[i_b_, i_p_]

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles[i_p, i_b].active

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            i_m = self.particles[i_p, i_b].material
            mass[i_b_] += self.materials[i_m].mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def n_boundary_particles(self):
        return self._n_boundary_particles

    @property
    def solve_stats(self):
        """Iteration counts and residuals of the solves of the last step (see SPHSolveStats)."""
        return self._solve_stats

    def get_links_fluid_wrench(self, envs_idx=None):
        """Return the force and the torque of the fluid on every rigid link, averaged over the substeps of the last step.

        The force is in world frame, in N. The torque is about the center of mass of the link, in world frame, in N m.
        Both hold the pressure and viscous load of the fluid alone, without gravity, actuators or contacts. For a link
        driven along a prescribed trajectory, they are the load its actuators carry against the fluid. Links away from
        the fluid get zero. The tensors have shape (n_envs, n_links, 3), without the leading dimension for a scene that
        is not batched.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_links = self.sim.rigid_solver.n_links
        if not self._has_boundary:
            zeros = torch.zeros((len(envs_idx), n_links, 3), dtype=gs.tc_float, device=gs.device)
            force, torque = zeros, zeros.clone()
        else:
            inv_step_dt = 1.0 / self._sim.dt
            force = qd_to_torch(self.links_fluid_impulse, envs_idx, transpose=True) * inv_step_dt
            torque = qd_to_torch(self.links_fluid_angular_impulse, envs_idx, transpose=True) * inv_step_dt
        if self._scene.n_envs == 0:
            force, torque = force[0], torque[0]
        return force, torque

    def get_links_fluid_wrench_parts(self, envs_idx=None):
        """Return the parts of get_links_fluid_wrench due to the pressure, the viscosity and the wall clearance guard.

        A dict {"pressure": (force, torque), "viscous": (force, torque), "clearance": (force, torque)}, in the frames,
        units and shapes of get_links_fluid_wrench, whose sum the parts are. The pressure part is the boundary term of
        the pressure solves (BWJ23 eq. (33)), the viscous part the impulse of the wall viscosity tensors of the boundary
        slots, the clearance part the reaction of the wall clearance guard (see _kernel_enforce_wall_clearance). A
        sheared fluid carries no shear stress through its pressure, so the pressure part of the torque of a
        tangentially moving wall measures the shear stress that the discrete pressure carries; the pressure part of a
        drag also holds the form drag of the flow (SPH_SOLVER_GOAL.md, N4 and N6).
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_links = self.sim.rigid_solver.n_links
        parts = {}
        if self._has_boundary:
            # fields of shape (n_links, n_envs, N_LOAD_PARTS) of 3-vectors
            inv_step_dt = 1.0 / self._sim.dt
            impulse = qd_to_torch(self.links_fluid_part_impulse)[:, envs_idx]
            angular_impulse = qd_to_torch(self.links_fluid_part_angular_impulse)[:, envs_idx]
        for i_q, name in enumerate(LOAD_PARTS):
            if not self._has_boundary:
                force = torch.zeros((len(envs_idx), n_links, 3), dtype=gs.tc_float, device=gs.device)
                torque = force.clone()
            else:
                force = impulse[:, :, i_q].transpose(0, 1) * inv_step_dt
                torque = angular_impulse[:, :, i_q].transpose(0, 1) * inv_step_dt
            if self._scene.n_envs == 0:
                force, torque = force[0], torque[0]
            parts[name] = (force, torque)
        return parts

    @qd.func
    def _task_pressure_virial(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        kappa_j = self.particles_reordered[j, i_b].pressure_density2_norm_p
        if qd.static(not self._df_skip_div):
            kappa_j += self.particles_reordered[j, i_b].pressure_density2_norm_d
        weight = self._func_pressure_pair_weight(i, j, x_ij, i_b)
        ret.virial += (
            0.5
            * weight
            * (ret.kappa_i + kappa_j)
            * self.materials[i_m].rest_volume
            * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
        )

    @qd.func
    def _task_pressure_virial_boundary(self, i, j, ret: qd.template(), i_b):
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret.virial += (
            0.5
            * ret.kappa_i
            * self.boundary_particles_reordered[j, i_b].volume
            * (-self.cubic_kernel_derivative(x_ib).dot(x_ib))
        )

    @qd.kernel
    def _kernel_get_particles_pressure(self, pressure: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            value = gs.qd_float(0.0)
            if self.particles[i_p, i_b].active:
                i_r = self.particles[i_p, i_b].reordered_idx
                kappa = self.particles_reordered[i_r, i_b].pressure_density2_norm_p
                if qd.static(not self._df_skip_div):
                    kappa += self.particles_reordered[i_r, i_b].pressure_density2_norm_d
                scale = qd.Struct(virial=gs.qd_float(0.0), kappa_i=kappa)
                self._func_for_all_fluid_neighbors(i_r, scale, self._task_pressure_virial, i_b)
                if qd.static(self._has_boundary):
                    self._func_for_all_boundary_neighbors(i_r, scale, self._task_pressure_virial_boundary, i_b)
                i_m = self.particles_reordered[i_r, i_b].material
                value = self.materials[i_m].rest_density * scale.virial / 3.0
            pressure[i_b, i_p] = value

    def get_particles_pressure(self, envs_idx=None):
        """Return the pressure of every fluid particle at the end of the last substep, in Pa.

        The pressure solves give each particle a multiplier kappa_i (the pressure over the rest density, of the density
        and the divergence solves), whose pair force -m_i sum_j V_j (kappa_i + kappa_j) grad W_ij takes the gradient
        of a linear pressure field scaled by C_i = -sum_j V_j grad W_ij . x_ij / 3, the boundary samples included: 1
        in the bulk to within a percent, less where neighbors are missing, at the free surface. The returned pressure
        is the virial of the pair forces on the particle (Irving and Kirkwood), half of each pair to each of its two
        particles: rho_0 / 3 sum_j V_j c_ij (kappa_i + kappa_j) / 2 (-grad W_ij . x_ij), c_ij the pair weight of the
        isotropic pressure (1 without it), which is rho_0 C_i kappa_i in a uniform field. Within a support radius of a rigid surface, the boundary force of a particle takes its own
        multiplier alone (Bender et al. 2023), half the force of a pair of equal multipliers, so that a boundary sample
        adds rho_0 / 3 V_b kappa_i / 2 (-grad W_ib . x_ib); with the full kappa_i the readout exceeded the hydrostatic
        pressure by 1.6 times at a floor (SPH_SOLVER_STATUS.md finding 29).
        The tensor has shape (n_envs, n_particles) in the particle order of the solver, without the leading dimension
        for a scene that is not batched; inactive particles get zero.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        pressure = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_particles_pressure(pressure)
        pressure = pressure[envs_idx]
        if self._scene.n_envs == 0:
            pressure = pressure[0]
        return pressure

    def get_viscous_dissipation(self, envs_idx=None):
        """Return the kinetic energy the viscosity of the fluid dissipated over the substeps of the last step, in J.

        It is the dissipation of the implicit viscosity operator at the final velocities of each substep, the no-slip
        walls included (see _kernel_accumulate_viscous_dissipation). The tensor has shape (n_envs,), and is a scalar for
        a scene that is not batched. It is zero for an inviscid fluid.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        if not self._has_viscosity:
            dissipation = torch.zeros((len(envs_idx),), dtype=gs.tc_float, device=gs.device)
        else:
            dissipation = qd_to_torch(self.envs_viscous_dissipation, envs_idx)
        if self._scene.n_envs == 0:
            dissipation = dissipation[0]
        return dissipation

    @property
    def _n_wrench_rows(self):
        """Number of rows of the wrenches of the coupled links, at least one for kernels that also reset scalars."""
        return max(6 * len(self._coupled_links_idx), 1) if self._has_rigid_coupling else 1

    @property
    def boundary_particle_radius(self):
        return self._boundary_spacing / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
