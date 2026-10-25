import math

import numpy as np

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers import LegacyCoupler
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils import array_class
from genesis.utils.misc import tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud

from .base_solver import GravityMixin, Solver, TimeBasedMixin


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
        self._df_min_div_iters = options.min_divergence_solver_iterations
        self._df_min_den_iters = options.min_density_solver_iterations
        self._df_skip_div = options.skip_divergence_solver
        self._df_warm_start = options.warm_start
        self._micropolar = options.micropolar
        self._df_eps = 1e-5
        # A particle with fewer neighbors sits at a free surface, where the divergence estimate is unreliable.
        self._df_min_div_neighbors = 20

        # Implicit position-based fluids (IPBF) parameters
        self._ipbf_compliance = options.compliance
        self._ipbf_damping_compliance = options.damping_compliance
        self._ipbf_has_damping = options.damping_compliance is not None
        # Distance between the damped and the solved positions over which the damping fades out, as in IPBF.
        self._ipbf_damping_dist = 60.0 * self._support_radius

        self._particle_volume = options.particle_volume_factor * self._particle_size**3

        # Boundary particles sampled on the surface of the rigid geoms. The spacing is the one of the regular triangle
        # sampling of Akinci et al. 2012 boundaries, denser than the fluid so that the fluid cannot leak through.
        self._boundary_spacing = 0.75 * self._particle_size
        # Upper bound on the number of boundary particles, which grows with the area of the simulation domain.
        self._max_n_boundary_particles = 5_000_000
        self._boundary_offset = options.boundary_offset
        self._n_boundary_particles = 0
        self._has_boundary = False

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
        particle_state_dtypes = dict(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # non-pressure acceleration
            material=gs.qd_int,  # material index
            reordered_idx=gs.qd_int,  # TODO: see above.
            active=gs.qd_bool,
            # Micro-rotation (Bender et al. 2017, "Turbulent Micropolar SPH Fluids with Foam"), allocated on demand.
            **(dict(angular_vel=gs.qd_vec3, angular_acc=gs.qd_vec3) if self._micropolar else {}),
            # rho / rho_0 = sum_j V_j W_ij
            density_norm=gs.qd_float,
            # number of neighbors within the support radius
            n_neighbors=gs.qd_int,
        )
        if self._pressure_solver == "IPBF":
            struct_particle_state = qd.types.struct(
                **particle_state_dtypes,
                # position at the beginning of the substep
                pos_prev=gs.qd_vec3,
                # Newton step of the current iteration
                delta_pos=gs.qd_vec3,
                # position solved with the damping compliance, see [_kernel_ipbf_update_velocity]
                pos_damped=gs.qd_vec3,
            )
        else:
            struct_particle_state = qd.types.struct(
                **particle_state_dtypes,
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
            )

        # static material info
        struct_material_info = qd.types.struct(
            rest_volume=gs.qd_float,
            mass=gs.qd_float,
            rest_density=gs.qd_float,  # mass / rest_volume
            viscosity=gs.qd_float,
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

        # Per-environment reductions of the DFSPH and IPBF solver iterations.
        self.envs_n_active = qd.field(gs.qd_int, shape=(self._B,))
        self.envs_solver_error = qd.field(gs.qd_float, shape=(self._B,))

        if self._has_boundary:
            # Boundary particles, sampled on the rigid geoms and moved with their link. They carry the static
            # sampling data as well, so that their hashed (reordered) copy is self-contained.
            struct_boundary_particle_state = qd.types.struct(
                pos_local=gs.qd_vec3,  # position in link frame
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

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active:
            self.sh.build(self._B)

            # Sample the boundary particles of the rigid geoms coupled with the fluid, in link frame.
            boundaries_pos_local, boundaries_link_idx, boundaries_is_fixed, boundaries_active = [], [], [], []
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
                            else:
                                verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                                faces = geom.init_faces
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
                            if geom.type == gs.GEOM_TYPE.PLANE:
                                normals_local = surface_normal
                            pos_local = pos_local - self._boundary_offset * normals_local

                            # A fixed link never moves, so the samples outside the boundary area can be dropped.
                            if link.is_fixed:
                                pos_world = gu.transform_by_trans_quat(pos_local, link_pos, link_quat)
                                is_inside = np.all((pos_world > area_lower) & (pos_world < area_upper), axis=1)
                                pos_local = pos_local[is_inside]

                            if geom.active_envs_mask is None:
                                active = np.ones((self._B, len(pos_local)), dtype=gs.np_int)
                            else:
                                envs_active = tensor_to_array(geom.active_envs_mask, dtype=gs.np_int)
                                active = np.tile(envs_active[:, None], (1, len(pos_local)))
                            boundaries_pos_local.append(pos_local)
                            boundaries_link_idx.append(np.full(len(pos_local), link.idx, dtype=gs.np_int))
                            boundaries_is_fixed.append(np.full(len(pos_local), link.is_fixed, dtype=gs.np_int))
                            boundaries_active.append(active)
            self._n_boundary_particles = sum(len(pos_local) for pos_local in boundaries_pos_local)
            self._has_boundary = self._n_boundary_particles > 0
            if self._has_boundary:
                gs.logger.info(f"SPH boundary sampled with ~<{self._n_boundary_particles}>~ particles.")
                self.sh_boundary = gu.SpatialHasher(cell_size=self.sh.cell_size, grid_res=self.sh.grid_res)
                self.sh_boundary.build(self._B)

            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            if self._has_boundary:
                self._kernel_add_boundary_particles(
                    np.concatenate(boundaries_pos_local),
                    np.concatenate(boundaries_link_idx),
                    np.concatenate(boundaries_is_fixed),
                    np.concatenate(boundaries_active, axis=1),
                )
                self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
                self._kernel_compute_boundary_volume()

            # The explicit viscosity term in [_task_compute_non_pressure_forces] is only stable below this timestep.
            viscosity_max = max(entity.material.viscosity for entity in self.entities)
            stable_dt = 0.01 * self._support_radius**2 / viscosity_max if viscosity_max > 0.0 else float("inf")
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep "
                    f"({stable_dt:.4g}) for SPH solver. "
                    "Please consider reducing the timestep, liquid viscosity, or increasing the particle size."
                )

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

        # Viscosity, using the 2 * (d + 2) Laplacian approximation in 3D with a regularized denominator.
        v_ij = (self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel).dot(x_ij)
        rho_j = self.particles_reordered[j, i_b].density_norm * self.materials[i_m_j].rest_density
        grad_W = self.cubic_kernel_derivative(x_ij)
        ret.acc += (
            2.0
            * (3.0 + 2.0)
            * self.materials[i_m_i].viscosity
            * (mass_j / rho_j)
            * v_ij
            / (dist**2 + 0.01 * self._support_radius**2)
            * grad_W
        )

        if qd.static(self._micropolar):
            # The micro-rotation field only couples particles of the same material.
            if i_m_i == i_m_j:
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
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

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
                if qd.static(self._micropolar):
                    self.particles_reordered[i_p, i_b].angular_vel += (
                        self._substep_dt * self.particles_reordered[i_p, i_b].angular_acc
                    )

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
        link_idx: qd.types.ndarray(),
        is_fixed: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            for i in qd.static(range(3)):
                self.boundary_particles[i_bp, i_b].pos_local[i] = pos_local[i_bp, i]
            self.boundary_particles[i_bp, i_b].link_idx = link_idx[i_bp]
            self.boundary_particles[i_bp, i_b].is_fixed = qd.cast(is_fixed[i_bp], gs.qd_bool)
            self.boundary_particles[i_bp, i_b].active = qd.cast(active[i_b, i_bp], gs.qd_bool)

    @qd.kernel
    def _kernel_update_boundary_particles(self, links_state: array_class.LinksState):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_l = self.boundary_particles[i_bp, i_b].link_idx
                pos = gu.qd_transform_by_trans_quat(
                    self.boundary_particles[i_bp, i_b].pos_local, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].pos = pos
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
    def _task_compute_boundary_density_and_factor(self, i, j, ret: qd.template(), i_b):
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret.density_norm += V_b * self.cubic_kernel(x_ib.norm())
        if qd.static(self._pressure_solver == "DFSPH"):
            # BWJ23 eq. (32): boundary particles carry no constraint of their own, so they only enter the gradient of
            # the constraint of particle i, and not the sum of squared neighbor gradients.
            ret.grad_p_i += V_b * self.cubic_kernel_derivative(x_ib)
        ret.n_neighbors += 1

    @qd.func
    def _task_compute_boundary_density_change(self, i, j, ret: qd.template(), i_b):
        V_b = self.boundary_particles_reordered[j, i_b].volume
        v_ib = self.particles_reordered[i, i_b].vel - self.boundary_particles_reordered[j, i_b].vel
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret += V_b * v_ib.dot(self.cubic_kernel_derivative(x_ib))

    @qd.func
    def _task_compute_boundary_pressure_acc(self, i, j, ret: qd.template(), i_b):
        # BWJ23 eq. (33): a single term of the fluid particle pressure, with no pressure value at the boundary.
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret.acc -= ret.p_rho2_i * V_b * self.cubic_kernel_derivative(x_ib)

    @qd.func
    def _task_compute_boundary_predicted_source(self, i, j, ret: qd.template(), i_b):
        # BWJ23 eq. (34): the pressure acceleration of a boundary particle is zero.
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret += V_b * self.particles_reordered[i, i_b].pressure_acc.dot(self.cubic_kernel_derivative(x_ib))

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
        if qd.static(self._pressure_solver == "DFSPH"):
            grad_p_j = -V_j * self.cubic_kernel_derivative(x_ij)
            ret.sum_grad_p_k += grad_p_j.norm_sqr()
            ret.grad_p_i -= grad_p_j
        ret.n_neighbors += 1

    @qd.kernel
    def _kernel_compute_density_and_factor(self, f: qd.i32):
        for i_b in range(self._B):
            self.envs_n_active[i_b] = 0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                ret = qd.Struct(
                    density_norm=self.materials[i_m].rest_volume * self.cubic_kernel(0.0),
                    sum_grad_p_k=gs.qd_float(0.0),
                    grad_p_i=qd.Vector.zero(gs.qd_float, 3),
                    n_neighbors=0,
                )
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_and_factor,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        ret,
                        self._task_compute_boundary_density_and_factor,
                        i_b,
                    )

                if qd.static(self._pressure_solver == "DFSPH"):
                    sum_grad_p_k = ret.sum_grad_p_k + ret.grad_p_i.norm_sqr()
                    factor = gs.qd_float(0.0)
                    if sum_grad_p_k > self._df_eps:
                        factor = 1.0 / sum_grad_p_k
                    self.particles_reordered[i_p, i_b].alpha_factor_adj = factor

                self.particles_reordered[i_p, i_b].density_norm = ret.density_norm
                self.particles_reordered[i_p, i_b].n_neighbors = ret.n_neighbors
                self.envs_n_active[i_b] += 1

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
            ret.acc -= p_sum * self.materials[i_m_j].rest_volume * self.cubic_kernel_derivative(x_ij)

    @qd.func
    def _func_compute_pressure_acc(self, i_p, i_b, is_divergence: qd.template()):
        ret = qd.Struct(
            acc=qd.Vector.zero(gs.qd_float, 3),
            p_rho2_i=self._func_pressure_density2_norm(i_p, i_b, is_divergence),
            is_divergence=is_divergence,
        )
        self.sh.for_all_neighbors(
            i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_pressure_acc, i_b
        )

        if qd.static(self._has_boundary):
            if qd.abs(ret.p_rho2_i) > self._df_eps:
                self.sh_boundary.for_all_neighbors_of_pos(
                    i_p,
                    self.particles_reordered[i_p, i_b].pos,
                    -1,
                    self.boundary_particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_boundary_pressure_acc,
                    i_b,
                )

        self.particles_reordered[i_p, i_b].pressure_acc = ret.acc

    @qd.func
    def _task_compute_density_change(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        ret += self.materials[i_m].rest_volume * v_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_init_pressure_solve(self, is_divergence: qd.template()):
        inv_dt = 1.0 / self._substep_dt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                density_change = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    density_change,
                    self._task_compute_density_change,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        density_change,
                        self._task_compute_boundary_density_change,
                        i_b,
                    )

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
                            p_rho2 = 0.5 * qd.min(self.particles_reordered[i_p, i_b].pressure_density2_norm_d, 0.5 * inv_dt)
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                else:
                    density_adv = self.particles_reordered[i_p, i_b].density_norm + self._substep_dt * density_change
                    self.particles_reordered[i_p, i_b].density_adv_norm = density_adv

                    p_rho2 = -qd.min(1.0 - density_adv, 0.0) * factor * inv_dt * inv_dt
                    if qd.static(self._df_warm_start):
                        p_rho2 = 0.0
                        if density_adv > 1.0:
                            # The clamp bounds p * dt^2, which is what limits the warm-started velocity change.
                            p_rho2 = 0.5 * qd.min(
                                self.particles_reordered[i_p, i_b].pressure_density2_norm_p, 0.00025 * inv_dt * inv_dt
                            )
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2

    @qd.func
    def _task_compute_predicted_source(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        a_ij = self.particles_reordered[i, i_b].pressure_acc - self.particles_reordered[j, i_b].pressure_acc
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        ret += self.materials[i_m].rest_volume * a_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_pressure_solve_iteration(self, is_divergence: qd.template()) -> float:
        """
        Run one Jacobi iteration of the divergence or density solve and return the worst average error over envs.

        The error is the average density deviation (normalized by the rest density) for the density solve, and the
        average normalized density change rate for the divergence solve.
        """
        # dt for the divergence solve, dt^2 for the density solve.
        dt_pow = self._substep_dt
        if qd.static(not is_divergence):
            dt_pow = self._substep_dt * self._substep_dt

        for i_b in range(self._B):
            self.envs_solver_error[i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self._func_compute_pressure_acc(i_p, i_b, is_divergence)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                predicted_source = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    predicted_source,
                    self._task_compute_predicted_source,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        predicted_source,
                        self._task_compute_boundary_predicted_source,
                        i_b,
                    )

                self.particles_reordered[i_p, i_b].predicted_source = predicted_source

                source = -self.particles_reordered[i_p, i_b].density_adv_norm
                if qd.static(not is_divergence):
                    source = 1.0 + source
                residual = source - dt_pow * predicted_source
                error = -qd.min(residual, 0.0)

                p_rho2 = self._func_pressure_density2_norm(i_p, i_b, is_divergence)
                factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                p_rho2 = qd.max(p_rho2 - 0.5 * residual * factor / dt_pow, 0.0)
                if qd.static(is_divergence):
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                    if self.particles_reordered[i_p, i_b].n_neighbors < self._df_min_div_neighbors:
                        error = 0.0
                else:
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2

                self.envs_solver_error[i_b] += error

        max_avg_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            n_active = qd.max(self.envs_n_active[i_b], 1)
            qd.atomic_max(max_avg_error, self.envs_solver_error[i_b] / n_active)
        return max_avg_error

    @qd.kernel
    def _kernel_apply_pressure_acc(self, is_divergence: qd.template()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self._func_compute_pressure_acc(i_p, i_b, is_divergence)
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc

    def _pressure_solve(self, is_divergence):
        """
        Run the divergence solve or the density solve and apply the resulting pressure accelerations to velocities.

        Iterations continue until the average error of every environment drops below the tolerance, within the
        min / max iteration bounds.
        """
        if is_divergence:
            n_iters_min, n_iters_max = self._df_min_div_iters, self._df_max_div_iters
            # The divergence error is a rate, so its tolerance is the density tolerance per substep.
            max_error = self._df_max_error_div / self._substep_dt
        else:
            n_iters_min, n_iters_max = self._df_min_den_iters, self._df_max_den_iters
            max_error = self._df_max_error_den

        self._kernel_init_pressure_solve(is_divergence)

        n_iters, avg_error, is_converged = 0, 0.0, False
        while (not is_converged or n_iters < n_iters_min) and n_iters < n_iters_max:
            avg_error = self._kernel_pressure_solve_iteration(is_divergence)
            is_converged = avg_error <= max_error
            n_iters += 1

        gs.logger.debug(
            f"DFSPH {'divergence' if is_divergence else 'density'} solve: {n_iters} iterations, avg error {avg_error:.4g}"
        )

        self._kernel_apply_pressure_acc(is_divergence)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- IPBF -----------------------------------------
    # ------------------------------------------------------------------------------------
    # Implicit position-based fluids (Diaz et al. 2025, "Implicit Position-Based Fluids"). Each iteration takes one
    # Newton step per particle on the variational form of implicit Euler, holding its neighbors fixed as in vertex block
    # descent, and applies half of it to all particles at once (relaxed Jacobi). The energy is the inertial term
    # 1 / (2 dt^2) ||x - y||_M^2 plus sum_j V_j k / 2 C_j^2, with y the inertial positions, k the bulk modulus and
    # C_j = max(rho_j / rho_0 - 1, 0) the density constraints clamped at the free surface. Dividing it by k V, the rest
    # volume V being shared by all particles, leaves the weight alpha rho_0 / dt^2 on the inertial term of a particle,
    # where alpha = 1 / k is the compliance. Boundary particles carry no constraint of their own, so they only enter the
    # constraint of the fluid particles around them.

    @qd.kernel
    def _kernel_ipbf_predict_position(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                self.particles_reordered[i_p, i_b].pos_prev = pos
                self.particles_reordered[i_p, i_b].pos = self.boundary.impose_pos(
                    pos + self._substep_dt * self.particles_reordered[i_p, i_b].vel
                )

    @qd.func
    def _task_ipbf_compute_newton_system(self, i, j, ret: qd.template(), i_b):
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        V_i = self.materials[i_m_i].rest_volume
        V_j = self.materials[i_m_j].rest_volume
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        grad_w = self.cubic_kernel_derivative(x_ij)
        hess_w = self.cubic_kernel_hessian(x_ij)

        # Derivatives of the constraint of particle i, which its own neighbor loop is the only one to know.
        ret.grad_c_i += V_j * grad_w
        ret.hess_c_i += V_j * hess_w

        # Derivatives of the constraint of particle j with respect to x_i. The Gauss-Newton term dC_j^T dC_j enters the
        # Hessian of every neighbor, clamped constraints included, which makes it positive definite as soon as the
        # neighbors span all directions. Leaving it out for the clamped constraints lets particles with few active
        # constraints around them take steps large enough to inject energy.
        grad_c_j = V_i * grad_w
        ret.hess += grad_c_j.outer_product(grad_c_j)
        c_j = self.particles_reordered[j, i_b].density_norm - 1.0
        if c_j > 0.0:
            ret.force -= c_j * grad_c_j
            # The row sums of the squared entries of a symmetric matrix are the squared norms of its columns.
            ret.hess_diag += c_j * V_i * qd.sqrt((hess_w * hess_w) @ qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float))

    @qd.func
    def _task_ipbf_compute_boundary_newton_system(self, i, j, ret: qd.template(), i_b):
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret.grad_c_i += V_b * self.cubic_kernel_derivative(x_ib)
        ret.hess_c_i += V_b * self.cubic_kernel_hessian(x_ib)

    @qd.kernel
    def _kernel_ipbf_solve_iteration(self) -> float:
        """
        Run one relaxed Jacobi iteration of the IPBF solver and return the worst average density error over envs.

        The density error is the average positive deviation of the density from the rest density, normalized by the
        rest density, at the positions the iteration starts from. [_kernel_compute_density_and_factor] must have
        evaluated the densities at these positions.
        """
        inv_dt2 = 1.0 / (self._substep_dt * self._substep_dt)

        for i_b in range(self._B):
            self.envs_solver_error[i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(
                    grad_c_i=qd.Vector.zero(gs.qd_float, 3),
                    hess_c_i=qd.Matrix.zero(gs.qd_float, 3, 3),
                    force=qd.Vector.zero(gs.qd_float, 3),
                    hess=qd.Matrix.zero(gs.qd_float, 3, 3),
                    hess_diag=qd.Vector.zero(gs.qd_float, 3),
                )
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_ipbf_compute_newton_system,
                    i_b,
                )
                if qd.static(self._has_boundary):
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        ret,
                        self._task_ipbf_compute_boundary_newton_system,
                        i_b,
                    )

                ret.hess += ret.grad_c_i.outer_product(ret.grad_c_i)
                c_i = self.particles_reordered[i_p, i_b].density_norm - 1.0
                if c_i > 0.0:
                    ret.force -= c_i * ret.grad_c_i
                    ret.hess_diag += c_i * qd.sqrt(
                        (ret.hess_c_i * ret.hess_c_i) @ qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)
                    )
                    self.envs_solver_error[i_b] += c_i

                # The constraint curvature C_j d2C_j / dx_i2 is indefinite, so it is replaced by the diagonal of the
                # norms of its columns (Andrews et al. 2017, "Geometric Stiffness for Real-time Constrained Multibody
                # Dynamics").
                for k in qd.static(range(3)):
                    ret.hess[k, k] += ret.hess_diag[k]

                pos = self.particles_reordered[i_p, i_b].pos
                pos_inertial = self.boundary.impose_pos(
                    self.particles_reordered[i_p, i_b].pos_prev
                    + self._substep_dt * self.particles_reordered[i_p, i_b].vel
                )
                i_m = self.particles_reordered[i_p, i_b].material
                rest_density = self.materials[i_m].rest_density
                eye = qd.Matrix.identity(gs.qd_float, 3)

                inertia = self._ipbf_compliance * rest_density * inv_dt2
                delta_pos = qd.Vector.zero(gs.qd_float, 3)
                # A positive diagonal makes the Hessian positive definite even when the few neighbors of a particle
                # leave its Gauss-Newton term singular. A zero diagonal means that no constraint is active, so the force
                # vanishes.
                if ret.hess_diag.min() + inertia > gs.EPS:
                    delta_pos = (ret.hess + inertia * eye).inverse() @ (ret.force - inertia * (pos - pos_inertial))
                self.particles_reordered[i_p, i_b].delta_pos = delta_pos

                if qd.static(self._ipbf_has_damping):
                    inertia_damped = self._ipbf_damping_compliance * rest_density * inv_dt2
                    delta_pos_damped = (ret.hess + inertia_damped * eye).inverse() @ (
                        ret.force - inertia_damped * (pos - pos_inertial)
                    )
                    self.particles_reordered[i_p, i_b].pos_damped = self.boundary.impose_pos(
                        pos + 0.5 * delta_pos_damped
                    )

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].pos = self.boundary.impose_pos(
                    self.particles_reordered[i_p, i_b].pos + 0.5 * self.particles_reordered[i_p, i_b].delta_pos
                )

        max_avg_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            n_active = qd.max(self.envs_n_active[i_b], 1)
            qd.atomic_max(max_avg_error, self.envs_solver_error[i_b] / n_active)
        return max_avg_error

    def _ipbf_solve(self, f):
        """
        Move the particles from their inertial positions toward the rest density with the IPBF iterations.

        Iterations continue until the average density error of every environment drops below the tolerance, within
        the min / max iteration bounds of the density solver.
        """
        n_iters, avg_error, is_converged = 0, 0.0, False
        while (not is_converged or n_iters < self._df_min_den_iters) and n_iters < self._df_max_den_iters:
            self._kernel_compute_density_and_factor(f)
            avg_error = self._kernel_ipbf_solve_iteration()
            is_converged = avg_error <= self._df_max_error_den
            n_iters += 1

        gs.logger.debug(f"IPBF solve: {n_iters} iterations, avg error {avg_error:.4g}")

    @qd.kernel
    def _kernel_ipbf_update_velocity(self, f: qd.i32):
        inv_dt = 1.0 / self._substep_dt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                pos_prev = self.particles_reordered[i_p, i_b].pos_prev
                vel = (pos - pos_prev) * inv_dt

                if qd.static(self._ipbf_has_damping):
                    # Eq. (17) of IPBF: an incompressible solve that has not converged injects kinetic energy, so the
                    # kinetic energy of a particle is brought toward the one it reaches in the last iteration solved
                    # with the damping compliance, all the more as both positions are close.
                    pos_damped = self.particles_reordered[i_p, i_b].pos_damped
                    dist = (pos - pos_damped).norm()
                    vel_sqr = vel.norm_sqr()
                    vel_damped_sqr = ((pos_damped - pos_prev) * inv_dt).norm_sqr()
                    if dist < self._ipbf_damping_dist and vel_damped_sqr < vel_sqr:
                        weight = 1.0 - dist / self._ipbf_damping_dist
                        vel *= qd.sqrt(1.0 - weight * (vel_sqr - vel_damped_sqr) / vel_sqr)

                self.particles_reordered[i_p, i_b].vel = vel

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
    def cubic_kernel_hessian(self, r):
        """
        Hessian of cubic spline smoothing kernel.

        The Hessian of a radial kernel is W'' along r and W' / |r| across. Both are finite and equal at the kernel
        center, where the direction of r is undefined.
        """
        res = qd.Matrix.zero(gs.qd_float, 3, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if q <= 1.0:
            d2w = 12.0 * k * (3.0 * q - 1.0) / h**2
            dw_over_r = 6.0 * k * (3.0 * q - 2.0) / h**2
            if q > 0.5:
                d2w = 12.0 * k * (1.0 - q) / h**2
                dw_over_r = -6.0 * k * (1.0 - q) ** 2 / (q * h**2)
            dir_r = qd.Vector.zero(gs.qd_float, 3)
            if r_norm > 1e-5:
                dir_r = r / r_norm
            res = dw_over_r * qd.Matrix.identity(gs.qd_float, 3) + (d2w - dw_over_r) * dir_r.outer_product(dir_r)
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

    def substep_pre_coupling(self, f):
        if self.is_active:
            # The rigid solver runs its substep before this one, so the fluid sees the boundary at the end of the
            # substep, with the velocity it reaches there.
            if self._has_boundary:
                self._kernel_update_boundary_particles(self.sim.rigid_solver.dyn_state.links)
            self._kernel_reorder_particles(f)
            self._kernel_compute_density_and_factor(f)
            if self._pressure_solver == "IPBF":
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_ipbf_predict_position(f)
                # The neighbors are searched around the inertial positions, from which the iterations only move the
                # particles by small corrections.
                self._kernel_copy_from_reordered(f)
                self._kernel_reorder_particles(f)
                self._ipbf_solve(f)
            else:
                if not self._df_skip_div:
                    self._pressure_solve(is_divergence=True)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._pressure_solve(is_divergence=False)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            if self._pressure_solver == "IPBF":
                self._kernel_ipbf_update_velocity(f)
            else:
                self._kernel_advect_position(f)
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
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
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
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            active[i_b, i_p] = self.particles[i_p, i_b].active

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
            if qd.static(self._pressure_solver == "DFSPH"):
                self.particles[i_p, i_b].pressure_density2_norm_p = 0.0
                self.particles[i_p, i_b].pressure_density2_norm_d = 0.0
            if qd.static(self._micropolar):
                self.particles[i_p, i_b].angular_vel = qd.Vector.zero(gs.qd_float, 3)

        self.materials[material_idx].rest_volume = self._particle_volume
        self.materials[material_idx].mass = self._particle_volume * material.rest_density
        self.materials[material_idx].rest_density = material.rest_density
        self.materials[material_idx].viscosity = material.viscosity
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
