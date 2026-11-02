import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize(
    "n_envs, pressure_solver",
    [
        (0, "WCSPH"),
        (0, "DFSPH"),
        pytest.param(2, "WCSPH", marks=pytest.mark.slow),  # ~150s
        pytest.param(2, "DFSPH", marks=pytest.mark.slow),  # ~300s
    ],
)
def test_rigid_flotation_follows_density_ratio(n_envs, pressure_solver, show_viewer):
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
            lower_bound=(-0.25, -0.25, 0.0),
            upper_bound=(0.25, 0.25, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.3),
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
            size=(0.5, 0.5, 0.5),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    light_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(-0.12, 0.0, 0.25),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    heavy_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.12, 0.0, 0.25),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=1500.0,
        ),
    )
    raft = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.12, 0.25),
            size=(0.1, 0.1, 0.04),
            euler=(30.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    scene.build(n_envs=n_envs)

    raft_rolls = []
    for _ in range(50):
        scene.step()
        raft_rolls.append(gu.quat_to_xyz(raft.get_quat(), rpy=True, degrees=True)[..., 0])
    raft_rolls = torch.stack(raft_rolls)

    # The static fluid pressure must push the light ball (rho well below the fluid rest density) up toward the
    # surface, while the heavy ball (rho well above) must keep sinking: flotation discriminates on the density
    # ratio, which is Archimedes' principle.
    light_z = light_ball.get_pos()[..., 2]
    heavy_z = heavy_ball.get_pos()[..., 2]
    assert (light_z > 0.28).all(), f"Light ball must rise under buoyancy, got z={light_z}"
    assert (heavy_z < 0.22).all(), f"Heavy ball must sink, got z={heavy_z}"
    # The pressure on the tilted raft acts off its center of mass, so the buoyancy torque rocks it about its roll
    # axis. That torque reaches the rigid body through the coupling wrench alone.
    assert (raft_rolls.amax(dim=0) - raft_rolls.amin(dim=0) > 2.0).all(), f"Raft must rock, got rolls={raft_rolls}"


@pytest.mark.parametrize("n_envs", [0, pytest.param(2, marks=pytest.mark.required)])
def test_coupling_force_wakes_hibernated_link(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            use_legacy_solver=True,
            lower_bound=(-0.55, -0.25, 0.0),
            upper_bound=(0.55, 0.25, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.5, 1.0),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.35, 0.0, 0.15),
            size=(0.3, 0.4, 0.3),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.35, 0.0, 0.06),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    # The island partition, which hibernation builds on, only engages when the scene holds at least two free
    # bodies; this bystander rests dry on the floor for the whole scenario.
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.06),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=2000.0,
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(15):
        scene.step()

    # The dry resting ball must have hibernated, otherwise the scenario would validate flotation instead of the
    # wake-up on coupling forces.
    is_ball_hibernated = qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated, transpose=True)
    assert is_ball_hibernated[..., ball.links[0].idx].all()

    # Teleporting the fluid over the hibernated ball must wake it through the coupling forces alone: there is no
    # awake body around to collide with, so with the ball left asleep the fluid pressure would be silently
    # discarded and the ball would stay frozen on the floor instead of floating up.
    liquid.set_position((0.35, 0.0, 0.15))
    for _ in range(40):
        scene.step()
    ball_z = ball.get_pos()[..., 2]
    assert (ball_z > 0.15).all(), f"Submerged ball must wake up and float, got z={ball_z}"


@pytest.mark.required
def test_fluid_link_momentum_exchange(show_viewer):
    # A box thrown into a ball of viscous liquid at zero gravity: pressure and wall viscosity exchange equal and opposite
    # impulses with the link, applied at the fluid particles, so the total linear and angular momentum stay constant.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.006,
            particle_volume_factor=1.0,
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=0.04),
        material=gs.materials.SPH.Liquid(sampler="regular", viscosity=1e-3, surface_tension=0.0),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.09, 0.015, 0.0), size=(0.03, 0.03, 0.03), euler=(0.0, 0.0, 20.0)),
        material=gs.materials.Rigid(rho=600.0),
    )
    scene.build()
    mass = scene.sph_solver.particle_volume * 1000.0
    box_mass = float(box.links[0].get_mass())
    box_inertia = box_mass * 0.03**2 / 6.0
    box.set_dofs_velocity((0.3, 0.0, 0.0, 0.0, 0.0, 0.0))

    def momenta():
        pos = tensor_to_array(liquid.get_particles_pos()).astype(np.float64)
        vel = tensor_to_array(liquid.get_particles_vel()).astype(np.float64)
        com = tensor_to_array(box.get_pos()).astype(np.float64)
        vel_box = tensor_to_array(box.get_vel()).astype(np.float64)
        ang_box = tensor_to_array(box.get_ang()).astype(np.float64)
        momentum = mass * vel.sum(axis=0) + box_mass * vel_box
        angular = mass * np.cross(pos, vel).sum(axis=0) + box_mass * np.cross(com, vel_box) + box_inertia * ang_box
        return momentum, angular, vel_box

    momentum_0, angular_0, vel_box_0 = momenta()
    for _ in range(150):
        scene.step()
    momentum, angular, vel_box = momenta()
    # the impact took most of the box momentum
    assert np.linalg.norm(vel_box) < 0.5 * np.linalg.norm(vel_box_0)
    assert_allclose(momentum, momentum_0, atol=1e-4 * np.linalg.norm(momentum_0))
    assert_allclose(angular, angular_0, atol=1e-3 * np.linalg.norm(angular_0))


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_fluid_load_readout_is_link_impulse(show_viewer):
    # A buoyant cube on a slide joint, held under water while the fluid settles, then released: the fluid load readout
    # is the impulse the link received, so with the weight it accounts for the momentum change of the link.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.01,
            particle_volume_factor=1.0,
            lower_bound=(-0.12, -0.12, -0.01),
            upper_bound=(0.12, 0.12, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.005), size=(0.24, 0.24, 0.01), fixed=True),
    )
    for x, y, sx, sy in (
        (0.105, 0.0, 0.01, 0.2),
        (-0.105, 0.0, 0.01, 0.2),
        (0.0, 0.105, 0.2, 0.01),
        (0.0, -0.105, 0.2, 0.01),
    ):
        scene.add_entity(morph=gs.morphs.Box(pos=(x, y, 0.06), size=(sx, sy, 0.12), fixed=True))
    cube = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="""
                <mujoco>
                    <worldbody>
                        <body name="cube" pos="0 0 0.04">
                            <joint name="slide" type="slide" axis="0 0 1"/>
                            <geom type="box" size="0.02 0.02 0.02" density="300"/>
                        </body>
                    </worldbody>
                </mujoco>
            """,
            default_armature=0.0,
        ),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.05), size=(0.19, 0.19, 0.1)),
        material=gs.materials.SPH.Liquid(sampler="regular", viscosity=1e-6, surface_tension=0.0),
    )
    scene.build()
    pos = liquid.get_particles_pos()
    liquid.set_particles_active(~((pos - torch.tensor((0.0, 0.0, 0.04), device=gs.device)).abs() < 0.025).all(dim=1))
    cube.set_dofs_kp((400.0,))
    cube.set_dofs_kv((20.0,))
    cube.control_dofs_position((0.0,))
    for _ in range(150):
        scene.step()
    link = cube.get_link("cube")
    mass = float(link.get_mass())
    cube.control_dofs_force((0.0,))
    vel_start = tensor_to_array(cube.get_dofs_velocity())[0]
    loads, pressure_loads = [], []
    for _ in range(20):
        scene.step()
        force, torque = (tensor_to_array(x)[link.idx] for x in scene.sph_solver.get_links_fluid_wrench())
        loads.append(force[2])
        # the pressure, viscous and clearance parts of the readout add up to it
        parts = scene.sph_solver.get_links_fluid_wrench_parts()
        part_force = sum(tensor_to_array(parts[name][0])[link.idx] for name in ("pressure", "viscous", "clearance"))
        part_torque = sum(tensor_to_array(parts[name][1])[link.idx] for name in ("pressure", "viscous", "clearance"))
        assert_allclose(part_force, force, rtol=1e-5, atol=1e-6 * mass * 9.81)
        assert_allclose(part_torque, torque, rtol=1e-5, atol=1e-6 * mass * 9.81 * 0.02)
        pressure_loads.append(tensor_to_array(parts["pressure"][0])[link.idx, 2])
    momentum_change = mass * (tensor_to_array(cube.get_dofs_velocity())[0] - vel_start)
    assert_allclose(momentum_change, (np.sum(loads) - 20 * mass * 9.81) * scene.dt, rtol=1e-2)
    assert np.mean(loads) > 0.5 * mass * 9.81
    # the buoyancy of the cube is carried by the pressure
    assert np.mean(pressure_loads) > 0.5 * mass * 9.81
