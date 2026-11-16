# Tests for SPH simulation - initial density and pressure validation
#
# Background:
# SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
# To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
# Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization, which lead to large
# spurious pressure forces and can cause the simulation to become unstable or diverge.

import pytest
import numpy as np
import torch

import genesis as gs
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose


def create_sph_scene(show_viewer, particle_size=0.01, pressure_solver="WCSPH"):
    """Create a basic SPH scene with regular sampler."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            use_legacy_solver=True,
            pressure_solver=pressure_solver,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build()

    return scene, liquid


@pytest.fixture(scope="function")
def sph_scene(show_viewer):
    """Fixture to create a basic SPH scene with regular sampler."""
    return create_sph_scene(show_viewer)


def get_sph_density_pressure(scene, n_particles):
    """
    Helper function to compute and retrieve density and pressure from SPH solver.

    This triggers the density computation kernel and returns the computed values.

    Parameters
    ----------
    scene : gs.Scene
        The built scene containing SPH solver.
    n_particles : int
        Number of particles to retrieve.

    Returns
    -------
    densities : np.ndarray
        Array of density values for all particles.
    pressures : np.ndarray
        Array of pressure values for all particles.
    """
    sph_solver = scene.sph_solver

    # Reorder particles into spatial grid cells for efficient neighbor queries,
    # then compute density. This mirrors what happens in substep_pre_coupling.
    # We must reorder first because _kernel_compute_rho operates on particles_reordered.
    sph_solver._kernel_reorder_particles(0)
    sph_solver._kernel_compute_rho(0)

    # Get density values from reordered particles (where _kernel_compute_rho stores results)
    densities = sph_solver.particles_reordered.rho.to_numpy()[:n_particles, 0]

    # Get material properties for pressure computation
    rho0 = sph_solver.particles_info[0].rest_density
    stiffness = sph_solver.particles_info[0].stiffness
    exponent = sph_solver.particles_info[0].exponent

    # Compute pressure using the same formula as the solver:
    # p = stiffness * ((max(rho, rho0) / rho0)^exponent - 1)
    densities_clamped = np.maximum(densities, rho0)
    pressures = stiffness * (np.power(densities_clamped / rho0, exponent) - 1.0)

    return densities, pressures


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_density_regular_sampler(show_viewer, pressure_solver, tol):
    # A regular sampler places particles at uniform spacing: interior particles see a consistent near-rest
    # density while boundary particles sit lower (fewer neighbors inside the kernel support), so the mean lands
    # below the rest density with a bounded spread.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rest_density

    # Get initial density values
    densities, _ = get_sph_density_pressure(scene, n_particles)

    # Test 1: Mean density should be within reasonable bounds
    # For regular sampler, mean is typically 70-90% of rho0 due to boundary effects
    density_mean = densities.mean()
    assert 0.5 * rho0 < density_mean < rho0, (
        f"Mean initial density {density_mean:.2f} is outside expected range "
        f"[{0.5 * rho0:.2f}, {rho0:.2f}] for rest density {rho0}"
    )

    # Test 2: Density standard deviation should be bounded
    # Regular sampler produces relatively uniform density distribution
    density_std = densities.std()
    assert density_std < 0.1 * rho0, (
        f"Density standard deviation {density_std:.2f} is too high "
        f"(expected < {0.1 * rho0:.2f} for rest density {rho0})"
    )

    # Test 3: No extreme density values
    assert densities.min() > 0.3 * rho0, (
        f"Minimum density {densities.min():.2f} is too low (expected > {0.3 * rho0:.2f})"
    )
    assert densities.max() < 1.5 * rho0, (
        f"Maximum density {densities.max():.2f} is too high (expected < {1.5 * rho0:.2f})"
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_pressure_regular_sampler(show_viewer, pressure_solver, tol):
    # Regular sampling yields densities at or below the rest density, and the equation of state clamps density
    # to the rest density, so the initial pressure must be exactly zero. A nonzero initial pressure would inject
    # spurious forces that destabilize the simulation.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rest_density
    stiffness = sph_solver.particles_info[0].stiffness

    # Get initial pressure values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Test: For regular sampler, initial density should be <= rho0
    # which means pressure should be 0 (clamped)
    # All particles should have near-zero pressure initially
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero. Mean: {pressures.mean():.6f}, "
        f"Max: {pressures.max():.6f}. "
        f"This indicates density > rho0 for some particles, which shouldn't happen "
        f"with regular sampler."
    )

    # Also verify that density is indeed <= rho0 for most particles
    fraction_above_rho0 = (densities > rho0).sum() / n_particles
    assert fraction_above_rho0 < 0.01, (
        f"{fraction_above_rho0 * 100:.1f}% of particles have density > rho0. "
        f"Regular sampler should produce density <= rho0 for almost all particles."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_simulation_stability_regular_sampler(show_viewer, pressure_solver, tol):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles

    # Get initial positions for comparison
    initial_pos = liquid.get_particles_pos().clone()

    # Run simulation for a few steps
    num_steps = 5
    max_velocities = []
    for step in range(num_steps):
        scene.step()
        pos = liquid.get_particles_pos()
        vel = liquid.get_particles_vel()
        assert not pos.isnan().any(), f"NaN values detected in positions at step {step}"
        assert not pos.isinf().any(), f"Inf values detected in positions at step {step}"
        assert not vel.isnan().any(), f"NaN values detected in velocities at step {step}"
        assert not vel.isinf().any(), f"Inf values detected in velocities at step {step}"

        # Track maximum velocity
        max_vel = vel.norm(dim=-1).max()
        max_velocities.append(max_vel)

    # Maximum velocity should be bounded (not exploding)
    # For stable simulation with gravity, velocities should be reasonable
    max_velocity_overall = max(max_velocities)
    assert max_velocity_overall < 10.0, (
        f"Maximum velocity {max_velocity_overall:.2f} exceeds reasonable bounds. This may indicate instability."
    )

    # Get final positions
    final_pos = liquid.get_particles_pos()

    # Check particles haven't moved too far (not exploding)
    displacement = (final_pos - initial_pos).norm(dim=-1).max()
    assert displacement < 1.0, (
        f"Maximum displacement {displacement:.4f} is too large after {num_steps} steps. This may indicate instability."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
@pytest.mark.parametrize("particle_size", [0.01, 0.02])
def test_density_consistency_different_particle_sizes(show_viewer, pressure_solver, particle_size, tol):
    # The SPH kernel support scales with the particle size, so the density distribution normalized by the rest
    # density must be independent of the particle size.
    scene, _ = create_sph_scene(
        show_viewer,
        particle_size=particle_size,
        pressure_solver=pressure_solver,
    )

    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rest_density

    # Get density values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Normalized metrics should be similar regardless of particle size
    normalized_mean = densities.mean() / rho0
    normalized_std = densities.std() / rho0

    # Mean density should be in similar range (50-100% of rho0)
    assert 0.5 < normalized_mean < 1.0, (
        f"Normalized mean density {normalized_mean:.3f} is outside expected range [0.5, 1.0] "
        f"for particle_size={particle_size}"
    )

    # Standard deviation should be bounded
    assert normalized_std < 0.1, (
        f"Normalized density std {normalized_std:.3f} is too high for particle_size={particle_size}"
    )

    # Pressure should still be zero
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero for particle_size={particle_size}. Mean: {pressures.mean():.6f}"
    )


# =============================================================================
# DFSPH (Divergence-Free SPH) Pressure Solver Tests
# =============================================================================


# TODO: Add more tests for validating the underlying physical behavior of DFSPH.
@pytest.mark.required
def test_dfsph_simulation_builds_and_runs(show_viewer):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver="DFSPH")

    assert scene.sph_solver is not None
    assert scene.sph_solver.n_particles > 0

    # Run a few simulation steps
    for _ in range(3):
        scene.step()

    pos = liquid.get_particles_pos()
    assert pos.shape[0] == scene.sph_solver.n_particles


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_pressure_carries_column_weight(show_viewer, pressure_solver):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            use_legacy_solver=True,
            pressure_solver=pressure_solver,
            lower_bound=(-0.15, -0.15, 0.0),
            upper_bound=(0.15, 0.15, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, 0.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.3, 0.3, 0.5),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    scene.build()

    for _ in range(100):
        scene.step()

    sph_solver = scene.sph_solver
    pressures = qd_to_numpy(sph_solver.particles_reordered.p, transpose=True)[0]
    heights = qd_to_numpy(sph_solver.particles_reordered.pos, transpose=True)[0, :, 2]

    # The pressure at the bottom of a settled column carries the weight of the whole column, which is exactly
    # the particle count times the particle weight spread over the column footprint. The tolerance absorbs the
    # residual sloshing of the column and two boundary effects that pull in opposite directions: the wall
    # pressure force from the floor relieves part of the compression that the WCSPH equation of state reads,
    # while the positive-part clamp of the DFSPH density solve rectifies velocity noise into extra pressure.
    weight_per_area = sph_solver.n_particles * sph_solver.particles_info[0].mass * 9.81 / (0.3 * 0.3)
    p_bottom = pressures[heights < 0.05].mean()
    assert 0.4 * weight_per_area < p_bottom < 1.5 * weight_per_area

    # The pressure must decrease with height, following the hydrostatic gradient. The factors are lower bounds
    # on the ratio of the hydrostatic depths at the band centers, with slack for pressure fluctuations.
    p_middle = pressures[(heights >= 0.2) & (heights < 0.25)].mean()
    p_top = pressures[heights >= 0.3].mean()
    assert p_bottom > 1.5 * p_middle
    assert p_middle > 2.0 * p_top


def create_viscous_ball(
    show_viewer,
    viscosity,
    n_envs=0,
    micropolar=False,
    max_viscosity_error=1e-3,
    density_correction="velocity",
    viscosity_integration="implicit",
):
    """A ball of viscous liquid at zero gravity, far from the bounds of the domain, for the default solver."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.01,
            particle_volume_factor=1.0,
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
            micropolar=micropolar,
            max_viscosity_error=max_viscosity_error,
            density_correction=density_correction,
            viscosity_integration=viscosity_integration,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=0.06,
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
            viscosity=viscosity,
            surface_tension=0.0,
        ),
    )
    scene.build(n_envs=n_envs)
    return scene, liquid


def get_momenta(liquid, mass):
    pos = tensor_to_array(liquid.get_particles_pos()).astype(np.float64)
    vel = tensor_to_array(liquid.get_particles_vel()).astype(np.float64)
    return mass * vel.sum(axis=0), mass * np.cross(pos, vel).sum(axis=0), 0.5 * mass * (vel**2).sum()


@pytest.mark.required
def test_implicit_viscosity_conserves_momentum(show_viewer):
    # The implicit viscosity is a sum of central pair forces, so a ball of liquid alone keeps its linear and angular
    # momentum while the viscosity dissipates its shear, and its rigid rotation is a null mode of the dissipation. The
    # rigid correction of the residual keeps this exact at a loose tolerance of the viscosity solve as well.
    scene, liquid = create_viscous_ball(show_viewer, viscosity=1e-2, max_viscosity_error=0.05)
    mass = scene.sph_solver.particle_volume * 1000.0
    pos = liquid.get_particles_pos()
    omega, shear = 5.0, 3.0
    vel = torch.stack((-omega * pos[:, 1] + shear * pos[:, 2] + 0.1, omega * pos[:, 0], torch.zeros_like(pos[:, 0])), 1)
    liquid.set_particles_vel(vel)
    momentum_0, angular_0, energy_0 = get_momenta(liquid, mass)
    for _ in range(50):
        scene.step()
    momentum, angular, energy = get_momenta(liquid, mass)
    assert_allclose(momentum, momentum_0, atol=2e-5 * np.linalg.norm(momentum_0))
    assert_allclose(angular, angular_0, atol=2e-5 * np.linalg.norm(angular_0))
    assert energy < energy_0
    assert scene.sph_solver.solve_stats.n_unconverged_viscosity == 0

    # rigid rotation and translation dissipate nothing
    scene.reset()
    pos = liquid.get_particles_pos()
    vel = torch.stack((-omega * pos[:, 1] + 0.1, omega * pos[:, 0], torch.zeros_like(pos[:, 0])), 1)
    liquid.set_particles_vel(vel)
    scene.step()
    dissipation = float(scene.sph_solver.get_viscous_dissipation())
    assert dissipation < 1e-3 * get_momenta(liquid, mass)[2]


@pytest.mark.required
def test_explicit_viscosity_dissipates_strain(show_viewer):
    viscosity = 1e-3
    scene, liquid = create_viscous_ball(show_viewer, viscosity=viscosity, viscosity_integration="explicit")
    mass = scene.sph_solver.particle_volume * 1000.0
    pos = liquid.get_particles_pos()
    omega, shear = 5.0, 3.0
    vel = torch.stack((-omega * pos[:, 1] + shear * pos[:, 2] + 0.1, omega * pos[:, 0], torch.zeros_like(pos[:, 0])), 1)
    liquid.set_particles_vel(vel)
    momentum_0, angular_0, energy_0 = get_momenta(liquid, mass)
    n_steps = 50
    for _ in range(n_steps):
        scene.step()
    momentum, angular, energy = get_momenta(liquid, mass)
    assert_allclose(momentum, momentum_0, atol=2e-5 * np.linalg.norm(momentum_0))
    assert_allclose(angular, angular_0, atol=2e-3 * np.linalg.norm(angular_0))
    # The simple shear is a pure strain of rate shear / 2 plus a rigid rotation, so the viscosity dissipates
    # 2 * mu * |E|^2 = mu * shear^2 per unit volume, the strain decaying little over the horizon.
    volume = 4.0 / 3.0 * np.pi * 0.06**3
    duration = n_steps * scene.dt
    assert_allclose(energy_0 - energy, 1000.0 * viscosity * shear**2 * volume * duration, rtol=0.25)

    # rigid rotation and translation dissipate nothing
    scene.reset()
    pos = liquid.get_particles_pos()
    vel = torch.stack((-omega * pos[:, 1] + 0.1, omega * pos[:, 0], torch.zeros_like(pos[:, 0])), 1)
    liquid.set_particles_vel(vel)
    energy_0 = get_momenta(liquid, mass)[2]
    scene.step()
    assert energy_0 - get_momenta(liquid, mass)[2] < 1e-4 * energy_0


@pytest.mark.required
def test_position_drift_correction_keeps_momentum(show_viewer):
    # The drift correction moves the particles without changing their velocities, and a rigid rotation of the fluid
    # restores the angular momentum the displacements change, so a ball of liquid keeps both momenta.
    scene, liquid = create_viscous_ball(show_viewer, viscosity=1e-3, density_correction="position")
    mass = scene.sph_solver.particle_volume * 1000.0
    pos = liquid.get_particles_pos()
    vel = torch.stack((-5.0 * pos[:, 1] + 3.0 * pos[:, 2] + 0.1, 5.0 * pos[:, 0], torch.zeros_like(pos[:, 0])), 1)
    liquid.set_particles_vel(vel)
    momentum_0, angular_0, _ = get_momenta(liquid, mass)
    for _ in range(50):
        scene.step()
    momentum, angular, _ = get_momenta(liquid, mass)
    assert_allclose(momentum, momentum_0, atol=2e-5 * np.linalg.norm(momentum_0))
    assert_allclose(angular, angular_0, atol=2e-5 * np.linalg.norm(angular_0))
    assert scene.sph_solver.solve_stats.n_unconverged_drift == 0


@pytest.mark.required
def test_dfsph_state_restores_warm_starts(show_viewer):
    # A restart from a saved state repeats the run up to the round-off of the GPU reductions, which a restart that lost
    # the pressure warm starts does not.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4),
        sph_options=gs.options.SPHOptions(
            particle_size=0.01,
            particle_volume_factor=1.0,
            lower_bound=(-0.2, -0.2, -0.01),
            upper_bound=(0.2, 0.2, 0.3),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    liquid = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.05), size=(0.1, 0.1, 0.1)),
        material=gs.materials.SPH.Liquid(sampler="regular", viscosity=1e-3, surface_tension=0.0),
    )
    scene.build(n_envs=2)
    for _ in range(20):
        scene.step()
    saved = scene.get_state()
    scene.step()
    pos_reference = tensor_to_array(liquid.get_particles_pos())
    scene.reset(saved, envs_idx=[1])
    scene.reset(saved, envs_idx=[0])
    scene.step()
    restart_error = np.abs(tensor_to_array(liquid.get_particles_pos()) - pos_reference).max()
    sph_state = next(state for state in saved.solvers_state if isinstance(state, SPHSolverState))
    sph_state.pressure.zero_()
    scene.reset(saved)
    scene.step()
    control_error = np.abs(tensor_to_array(liquid.get_particles_pos()) - pos_reference).max()
    assert restart_error < 1e-2 * control_error


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])
def test_boundary_offset_closes_reentrant_edges():
    # A floor (normal +z) meeting a wall (normal -x) at a reentrant edge along y: the boundary particles of both faces
    # move by the offset into the geom, and the samples of the edge extend each moved face to the line where the
    # moved planes meet, at x = offset and z = -offset.
    from genesis.engine.solvers.sph_solver import sample_reentrant_edges

    verts = np.array(
        [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)]
    )
    faces = np.array([(0, 1, 2), (0, 2, 3), (1, 5, 4), (1, 4, 2)])
    offset, spacing = 0.1, 0.05
    pos, normals, areas = sample_reentrant_edges(verts, faces, offset, spacing)
    is_floor = normals[:, 2] > 0.5
    is_wall = normals[:, 0] < -0.5
    assert np.all(is_floor | is_wall) and is_floor.sum() == is_wall.sum() == 20 * 2
    assert_allclose(pos[is_floor, 2], -offset, tol=1e-9)
    assert_allclose(pos[is_wall, 0], offset, tol=1e-9)
    assert_allclose(pos[is_floor, 0].max(), offset, tol=1e-9)
    assert_allclose(pos[is_wall, 2].min(), -offset, tol=1e-9)
    assert_allclose(areas.sum(), 2 * offset * 1.0, tol=1e-9)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])
def test_volume_map_of_half_space():
    # The boundary volume of the half-space z < 0 is a function of the height alone, 1 deep inside, 0 a support radius
    # above the surface, and the volumes at heights z and -z add up to the integral of the kernel, 1.
    from genesis.engine.solvers.sph_solver import compute_volume_map, half_space_volume, volume_map_to_distance

    support_radius, cell_size = 1.0, 0.125
    # The grid spans a support radius on either side of its central column, and of the heights checked.
    n_nodes = (17, 17, 49)
    lower = np.array([-1.0, -1.0, -3.0])
    volume = compute_volume_map(lambda x: x[:, 2], lower, cell_size, n_nodes, support_radius)
    heights = lower[2] + cell_size * np.arange(n_nodes[2])
    is_checked = np.abs(heights) <= 2.0
    column, heights = volume[8, 8, is_checked], heights[is_checked]
    assert_allclose(column[heights <= -support_radius], 1.0, tol=1e-6)
    assert_allclose(column[heights >= support_radius], 0.0, tol=1e-6)
    assert_allclose(column[heights == 0.0], 0.5, tol=1e-6)
    assert_allclose(column + column[::-1], 1.0, tol=1e-6)
    assert np.all(np.diff(column) <= 1e-6)
    # against the closed form, within the discretization of the solid on the grid
    assert_allclose(column, half_space_volume(heights, support_radius), tol=5e-3)
    is_within = np.abs(heights) < support_radius
    dist = volume_map_to_distance(half_space_volume(heights[is_within], support_radius), support_radius)
    assert_allclose(dist, heights[is_within], tol=1e-4)
