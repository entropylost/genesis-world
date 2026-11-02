from typing import Any, Literal

import numpy as np
from pydantic import PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, UnitVec4FType, Vec3FType

from .options import Options

############################ Top level: simulator and coupler ############################
"""
Simulator options specifies the global settings for the simulator and the coupler options specifies whether the
coupling between pairs of solvers is enabled.
"""


class SimOptions(Options):
    """
    Options configuring the top-level simulator.

    Note
    ----
    1. `SimOptions` specifies the global settings for the simulator. Some parameters exist both in `SimOptions` and
    `SolverOptions`. In this case, if such parameters are given in `SolverOptions`, it will override the one specified
    in `SimOptions` for this specific solver. For example, if `dt` is only given in `SimOptions`, it will be shared by
    all the solvers, while a solver given its own `dt` integrates over that interval instead, and the number of
    substeps of every solver follows from it.

    2. In differentiable mode, `substeps_local` must be divisible by `substeps`, as external command is input per
    `step`, but `substep`. If `requires_grad` is False, we can use arbitrary `substeps_local`.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    substeps : int, optional
        Number of substeps per simulation step, i.e. how many times each solver integrates per `scene.step()`. More
        substeps buy accuracy and stability, at a runtime cost that grows linearly with the count in the worst case
        though sub-linearly in practice. Setting both this and a solver `dt` that implies a different count raises an
        exception. Defaults to 1.
    substeps_local : int, optional
        Number of substeps stored in GPU memory. Defaults to None. This is used for differentiable mode.
    gravity : tuple, optional
        Gravity force in N/kg. Defaults to (0.0, 0.0, -9.81).
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    use_hydroelastic_contact : bool, optional
        Whether to use hydroelastic contact. Defaults to False.
    """

    dt: PositiveFloat = 1e-2
    substeps: PositiveInt = 1
    substeps_local: PositiveInt | None = None  # number of substeps stored in GPU memory
    gravity: Vec3FType = (0.0, 0.0, -9.81)
    floor_height: float = 0.0
    requires_grad: StrictBool = False

    _steps_local: int | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_substeps(cls, data: dict) -> dict:
        if data.get("substeps_local") is None:
            # use 1 to save gpu memory when not in differentiable mode
            data["substeps_local"] = data.get("substeps", 1) if data.get("requires_grad", False) else 1
        return data

    def model_post_init(self, context: Any) -> None:
        if self.requires_grad:
            if self.substeps_local % self.substeps != 0:
                gs.raise_exception("`substeps_local` must be divisible by `substeps` when `requires_grad` is True.")
            else:
                self._steps_local = int(self.substeps_local / self.substeps)
        else:
            self._steps_local = None


class BaseCouplerOptions(Options):
    """
    Base class for all coupler options.
    """

    pass


class LegacyCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling.

    Parameters
    ----------
    rigid_mpm : bool, optional
        Whether to enable coupling between rigid and MPM solvers. Defaults to True.
    rigid_sph : bool, optional
        Whether to enable coupling between rigid and SPH solvers. Defaults to True.
    rigid_pbd : bool, optional
        Whether to enable coupling between rigid and PBD solvers. Defaults to True.
    rigid_fem : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    mpm_sph : bool, optional
        Whether to enable coupling between MPM and SPH solvers. Defaults to True.
    mpm_pbd : bool, optional
        Whether to enable coupling between MPM and PBD solvers. Defaults to True.
    fem_mpm : bool, optional
        Whether to enable coupling between FEM and MPM solvers. Defaults to True.
    fem_sph : bool, optional
        Whether to enable coupling between FEM and SPH solvers. Defaults to True.
    """

    rigid_mpm: StrictBool = True
    rigid_sph: StrictBool = True
    rigid_pbd: StrictBool = True
    rigid_fem: StrictBool = True
    mpm_sph: StrictBool = True
    mpm_pbd: StrictBool = True
    fem_mpm: StrictBool = True
    fem_sph: StrictBool = True


class SAPCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling for the Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html

    Parameters
    ----------
    n_sap_iterations : int, optional
        Number of iterations for the SAP solver. Defaults to 5.
    n_pcg_iterations : int, optional
        Number of iterations for the Preconditioned Conjugate Gradient solver. Defaults to 100.
    n_linesearch_iterations : int, optional
        Max number of iterations for the line search solver. Defaults to 10.
    sap_convergence_atol : float, optional
        Absolute tolerance for SAP convergence. Defaults to 1e-6.
    sap_convergence_rtol : float, optional
        Relative tolerance for SAP convergence. Defaults to 1e-5.
    sap_taud : float, optional
        Dissipation time scale for SAP. Defaults to 0.1.
    sap_beta : float, optional
        Normal regularization parameter for SAP. Defaults to 1.0.
    sap_sigma : float, optional
        Friction regularization parameter for SAP. Defaults to 1e-3.
    pcg_threshold : float, optional
        Threshold for the Preconditioned Conjugate Gradient solver. Defaults to 1e-6.
    linesearch_ftol : float, optional
        Line search sufficient value close to zero for exact linesearch. Defaults to 1e-6.
    linesearch_max_step_size : float, optional
        Maximum step size for exact linesearch. Defaults to 1.5.
    hydroelastic_stiffness : float, optional
        Stiffness for hydroelastic contact. Defaults to 1e8.
    point_contact_stiffness : float, optional
        Stiffness for point contact. Defaults to 1e8.
    fem_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        TET would be the default choice for most cases.
        VERT would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_fem_self_tet_contact : bool, optional
        Whether to use tetrahedral based self-contact. Defaults to True.
    rigid_rigid_type : str, optional
        Type of contact between rigid bodies. Defaults to "tet". Can be "tet", "vert", or "none".
    rigid_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        Tet would be the default choice for most cases.
        Vert would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_rigid_fem_contact : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    """

    n_sap_iterations: PositiveInt = 5
    n_pcg_iterations: PositiveInt = 100
    n_linesearch_iterations: PositiveInt = 10
    sap_convergence_atol: PositiveFloat = 1e-6
    sap_convergence_rtol: PositiveFloat = 1e-5
    sap_taud: PositiveFloat = 0.1
    sap_beta: PositiveFloat = 1.0
    sap_sigma: PositiveFloat = 1e-3
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_ftol: PositiveFloat = 1e-6
    linesearch_max_step_size: PositiveFloat = 1.5
    hydroelastic_stiffness: PositiveFloat = 1e8
    point_contact_stiffness: PositiveFloat = 1e8
    fem_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_fem_self_tet_contact: StrictBool = True
    rigid_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_rigid_fem_contact: StrictBool = True
    rigid_rigid_contact_type: Literal["tet", "vert", "none"] = "tet"


class IPCCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the Incremental Potential Contact (IPC) coupler.

    Time step, gravity, and differentiable simulation mode are derived from ``SimOptions``
    (``dt``, ``gravity``, ``requires_grad``) and should not be set here.

    Parameters
    ----------
    Newton Solver Options
    ---------------------
    newton_max_iterations : int, optional
        Maximum iterations for Newton solver. Defaults to None (use libuipc default: 1024).
    newton_min_iterations : int, optional
        Minimum iterations for Newton solver. Defaults to None (use libuipc default: 1).
    newton_tolerance : float, optional
        Velocity tolerance for Newton solver convergence. Defaults to None (use libuipc default: 0.05).
    newton_ccd_tolerance : float, optional
        CCD (Continuous Collision Detection) tolerance for Newton solver. Defaults to None (use libuipc default: 1.0).
    newton_use_adaptive_tolerance : bool, optional
        Whether Newton solver should use adaptive tolerance. Defaults to None (use libuipc default: False).
    newton_translation_tolerance : float, optional
        Translation rate tolerance for Newton solver. Defaults to None (use libuipc default: 0.1).
    newton_semi_implicit_enable : bool, optional
        Whether to enable semi-implicit Newton solver mode. Defaults to None (use libuipc default: False).
    newton_semi_implicit_beta_tolerance : float, optional
        Beta tolerance for semi-implicit Newton solver. Defaults to None (use libuipc default: 1e-3).

    Line Search Options
    -------------------
    n_linesearch_iterations : int, optional
        Maximum iterations for line search. Defaults to None (use libuipc default: 8).
    linesearch_report_energy : bool, optional
        Whether to report energy during line search. Defaults to None (use libuipc default: False).

    Linear System Options
    ---------------------
    linear_system_solver : str, optional
        Linear system solver type. Options: 'linear_pcg', 'direct', etc. Defaults to None (use libuipc default: 'linear_pcg').
    linear_system_tolerance : float, optional
        Tolerance for linear system solver. Defaults to None (use libuipc default: 1e-3).

    Contact Options
    ---------------
    contact_enable : bool, optional
        Whether to enable contact detection. Defaults to None (use libuipc default: True).
    contact_d_hat : float, optional
        Contact distance threshold. Defaults to None (use libuipc default: 0.01).
    contact_friction_enable : bool, optional
        Whether to enable friction in contact. Defaults to None (use libuipc default: True).
    contact_resistance : float, optional
        Ground/default contact resistance/stiffness. It is used for ground contact pairs and
        as the per-entity fallback when a material does not define ``contact_resistance``.
        For ground pairs, it is combined with entity ``material.contact_resistance`` via
        geometric mean. Defaults to 1e9.
    contact_eps_velocity : float, optional
        Epsilon velocity for contact. Defaults to None (use libuipc default: 0.01).
    contact_constitution : str, optional
        Contact constitution model. Options: 'ipc', 'isometric'. Defaults to None (use libuipc default: 'ipc').

    Collision Detection Options
    ---------------------------
    collision_detection_method : str, optional
        Collision detection method. Options: 'linear_bvh', 'spatial_hash', etc. Defaults to None (use libuipc default: 'linear_bvh').

    CFL Options
    -----------
    cfl_enable : bool, optional
        Whether to enable CFL (Courant-Friedrichs-Lewy) condition. Defaults to None (use libuipc default: False).

    Sanity Check Options
    --------------------
    sanity_check_enable : bool, optional
        Whether to enable sanity checks. Defaults to None (use libuipc default: True).

    Genesis Coupling Options
    ------------------------
    constraint_strength_translation : float, optional
        Translation strength for IPC soft transform constraint coupling.
        Higher values create stiffer position coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    constraint_strength_rotation : float, optional
        Rotation strength for IPC soft transform constraint coupling.
        Higher values create stiffer orientation coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    enable_rigid_ground_contact : bool, optional
        Whether to enable ground contact in IPC system. When False, objects in IPC will not collide
        with the ground plane. Defaults to True.
    enable_rigid_rigid_contact : bool, optional
        Whether to enable contact detection between rigid bodies (ABD objects) in the IPC system.
        When False, only soft-soft and soft-rigid collisions are detected by IPC; rigid-rigid
        collisions within IPC are skipped. Defaults to True.
    two_way_coupling : bool, optional
        Whether to apply coupling forces/torques from IPC back to Genesis rigid bodies. Defaults to True.
    enable_rigid_dofs_sync : bool, optional
        Whether to synchronize the IPC reference DOF state with Genesis each step for
        external_articulation entities. When True, IPC gets tighter coupling with Genesis joint
        state but may amplify small divergences. When False, IPC uses its own DOF reference
        without per-step updates. Defaults to False.
    free_base_driven_by_ipc : bool, optional
        For external_articulation with non-fixed base: whether base link is fully driven by IPC physics.
        When False, base link uses SoftTransformConstraint controlled by Genesis. When True, base link
        is fully driven by IPC physics. Defaults to False.
    _show_ipc_gui : bool, optional
        [Dev/debug] Enable the libuipc built-in polyscope GUI viewer for inspecting the IPC scene.
        Defaults to False.
    """

    # Newton solver options (None = use libuipc default)
    newton_max_iterations: PositiveInt | None = None
    newton_min_iterations: PositiveInt | None = None
    newton_tolerance: PositiveFloat | None = None
    newton_ccd_tolerance: PositiveFloat | None = None
    newton_use_adaptive_tolerance: StrictBool | None = None
    newton_translation_tolerance: PositiveFloat | None = None
    newton_semi_implicit_enable: StrictBool | None = None
    newton_semi_implicit_beta_tolerance: PositiveFloat | None = None

    # Line search options (None = use libuipc default)
    n_linesearch_iterations: PositiveInt | None = None
    linesearch_report_energy: StrictBool | None = None

    # Linear system options (None = use libuipc default)
    linear_system_solver: Literal["linear_pcg", "direct"] | None = None
    linear_system_tolerance: PositiveFloat | None = None

    # Contact options
    contact_enable: StrictBool | None = None
    contact_d_hat: PositiveFloat | None = None
    contact_friction_enable: StrictBool | None = None
    contact_resistance: PositiveFloat = 1e9
    contact_eps_velocity: PositiveFloat | None = None
    contact_constitution: Literal["ipc", "isometric"] | None = None

    # Collision detection options
    collision_detection_method: Literal["linear_bvh", "spatial_hash"] | None = None

    # CFL options
    cfl_enable: StrictBool | None = None

    # Sanity check options
    sanity_check_enable: StrictBool | None = None

    # Genesis coupling options
    constraint_strength_translation: PositiveFloat = 100.0
    constraint_strength_rotation: PositiveFloat = 100.0
    enable_rigid_ground_contact: StrictBool = True
    enable_rigid_rigid_contact: StrictBool = True
    two_way_coupling: StrictBool = True
    enable_rigid_dofs_sync: StrictBool = False
    free_base_driven_by_ipc: StrictBool = False

    _show_ipc_gui: bool = PrivateAttr(default=False)

    def __init__(self, *, _show_ipc_gui: StrictBool = False, **data) -> None:
        super().__init__(**data)
        self._show_ipc_gui = bool(_show_ipc_gui)


############################ Solvers inside simulator ############################
"""
Parameters in these solver-specific options will override SimOptions if available.
"""


class TimeBasedMixin(Options):
    """
    A mixin adding the integration interval `dt` to the options of the solvers that integrate over one.

    Parameters
    ----------
    dt : float, optional
        The interval this solver integrates over, in seconds. It must divide `SimOptions.dt` an integer number of
        times, and that quotient is the number of substeps this solver runs per scene step. A shorter interval buys
        accuracy and stability where the motion is stiff or fast, at a runtime cost that grows linearly with the number
        of substeps in the worst case though sub-linearly in practice. Every active solver advances together, so an
        interval that disagrees with another solver's, or with `SimOptions.substeps`, raises an exception. If none, this
        solver integrates over the interval the other options settle on. Defaults to None.
    """

    dt: PositiveFloat | None = None


class GravityMixin(Options):
    """
    A mixin adding `gravity` to the options of the solvers that accelerate their bodies under it.

    Parameters
    ----------
    gravity : tuple, optional
        The acceleration applied to the bodies of this solver, in m/s^2. Each solver has its own, so one subsystem
        can be simulated weightless next to another. If none, the value carried by `SimOptions` is used. Defaults to
        None.
    """

    gravity: Vec3FType | None = None


class KinematicOptions(Options):
    """
    Options configuring the KinematicSolver (visualization-only solver).

    KinematicSolver is a lightweight solver for ghost/reference entities that only computes
    forward kinematics for visualization. No collision, physics integration, or constraint
    solving is performed.

    Parameters
    ----------
    batch_links_info : bool, optional
        Whether to batch link info. Automatically enabled for heterogeneous simulation. Defaults to False.
    batch_dofs_info : bool, optional
        Whether to batch DOF info. Automatically enabled for heterogeneous simulation. Defaults to False.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    """

    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False
    IK_max_targets: PositiveInt = 6


class ToolOptions(TimeBasedMixin):
    """
    Options configuring the ToolSolver.

    Note
    ----
    ToolEntity is a simplified form of RigidEntity. It supports one way tool->other coupling, but has *no* internal dynamics and can only be created from a single mesh. This is a temporal workaround for differentiable rigid-soft interaction. This solver will be removed once differentiable mode is supported by the RigidSolver.

    Parameters
    ----------
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    """

    floor_height: float | None = None


class RigidOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the RigidSolver.

    Parameters
    ----------
    enable_collision : bool, optional
        Whether to enable collision detection. Defaults to True.
    enable_joint_limit : bool, optional
        Whether to enable joint limit. Defaults to True.
    enable_self_collision : bool, optional
        Whether to enable self collision within each entity. Defaults to True.
    enable_neutral_collision : bool, optional
        Whether to enable self collision occurring in neutral configuration (qpos0) within each entity. Defaults to
        False.
    enable_adjacent_collision : bool, optional
        Whether to enable collision between successive parent-child body pairs within each entity. Defaults to False.
    disable_constraint: bool, optional
        Whether to disable all constraints. Defaults to False.
    max_collision_pairs : int, optional
        Maximum number of collision pairs. Defaults to 100.
    max_contacts : int, optional
        Maximum number of simultaneous contact points per environment that the constraint solver can handle, which
        determines the size of the contact constraint buffers (3 to 10 constraint rows per contact point depending on
        'friction_cone', 'enable_torsional_friction', and 'enable_rolling_friction'). Defaults to None.

        This limit applies to the final contact points after pruning, not to the candidate contact points that
        collision detection can emit (see 'max_collision_pairs'). Exceeding it at runtime halts the simulation with
        an error. None resolves it automatically: the pre-pruning worst case or, when contact pruning is enabled
        (see 'contact_pruning_tolerance'), 32 contact points per candidate link pair but no less than 512, whichever
        is smaller.
    integrator : gs.integrator, optional
        Integrator type. Current supported integrators are 'gs.integrator.Euler', 'gs.integrator.implicitfast' and
        'gs.integrator.approximate_implicitfast'. 'Euler' and 'implicitfast' are consistent with their Mujoco
        counterpart. 'approximate_implicitfast' is an even faster approximation of 'implicitfast', which avoid
        computing the inverse mass matrix twice by considering the first order correction terms of the implicit
        integration scheme systematically, including for computing the acceleration resulting from the constraints
        and external forces. Although this approximation is wrong in theory, it works reasonably well in practice.
        Defaults to 'approximate_implicitfast'.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    batch_links_info : bool, optional
        Whether the model parameters of a link, such as its mass or its inertia, are stored per environment rather
        than shared by the whole batch. Storing them per environment is what lets each environment carry its own
        values, which domain randomization needs, and what makes a per-environment write possible at all. It costs one
        copy of every link parameter per environment, in memory and in the bandwidth to read it, which slows down the
        memory-bound kernels. Automatically enabled for heterogeneous simulation. Defaults to False.
    batch_joints_info : bool, optional
        Whether the model parameters of a joint are stored per environment rather than shared by the whole batch,
        with the same tradeoff as `batch_links_info`. Automatically enabled for heterogeneous simulation. Defaults to
        False.
    batch_dofs_info : bool, optional
        Whether the model parameters of a degree of freedom are stored per environment rather than shared by the
        whole batch, with the same tradeoff as `batch_links_info`. Automatically enabled for heterogeneous simulation.
        Defaults to False.
    constraint_solver : gs.constraint_solver, optional
        Constraint solver type. Current supported constraint solvers are 'gs.constraint_solver.CG' (conjugate gradient)
        and 'gs.constraint_solver.Newton' (Newton's method). Defaults to 'Newton'.
    iterations : int, optional
        Maximum number of iterations of the constraint solver, which exits early once its tolerance is met. A batch of
        parallel environments waits for its slowest one on every step, so raising the bound buys accuracy on the steps
        whose hardest contacts never converge at the price of every such step. Defaults to 25.
    tolerance : float, optional
        Tolerance for the constraint solver. If None, resolved based on the floating-point precision selected via
        `gs.init(precision=...)`: 1e-5 for single precision ("32") and 1e-8 for double precision ("64"). Defaults
        to None.
    ls_iterations : int, optional
        Number of line search iterations for the constraint solver. Defaults to 50.
    ls_tolerance : float, optional
        Tolerance for the line search. Defaults to 1e-2.
    noslip_iterations : int, optional
        Number of iterations for the noslip solver. Defaults to 0 (disabled).
        noslip is a post-processing step after the main solver to suppress slip/drift.
        Recommended to set this value to 5 for manipulation tasks or when slip/drift is a big problem.
        This option should only be enabled if necessary because it is experimental and will slow down the simulation.
    noslip_tolerance : float, optional
        Tolerance for the noslip solver. Defaults to 1e-6.
    friction_cone : gs.friction_cone, optional
        Contact friction cone model, trading numerical robustness for physical accuracy. 'gs.friction_cone.pyramidal'
        (default) is robust and easy to solve; 'gs.friction_cone.elliptic' is the exact isotropic cone, harder to solve
        but paired with a high 'impratio' it holds resting stacks without slow tangential creep. See 'gs.friction_cone'
        for the description of each model. Unsupported with the noslip solver or differentiable simulation.
    contact_resolution : gs.contact_resolution, optional
        How a contact's normal force and friction force are resolved against each other.
        'gs.contact_resolution.signorini' bounds friction against the normal force the contact has developed, so sliding
        never inflates it and a body launched horizontally decelerates at mu * g instead of lifting off, at the cost of
        extra solver iterations. 'gs.contact_resolution.convex' poses the contact as a single convex program, which
        converges more predictably on stiff scenes but lets fast sliding buy normal force. See 'gs.contact_resolution'
        for the description of each model. Defaults to None, resolving to 'signorini' with the elliptic cone and the
        Newton solver, and 'convex' otherwise - the pyramidal cone's rows do not separate, and the conjugate gradient
        solver does not reach the fixed point. Always 'convex' when 'enable_mujoco_compatibility' is set.
    enable_torsional_friction : bool, optional
        Whether contacts also resist relative spin about their normal, with strength set per geometry by the material
        option 'friction_torsional' (see 'gs.materials.Rigid'). Enable it when spin resistance matters - a grasped
        object twisting in a gripper, a top spinning in place - motions a point contact transmits no torque against,
        so they persist indefinitely otherwise. The extra spin resistance slows down the constraint solve on every
        contact, including those where spin is irrelevant. Defaults to False.
    enable_rolling_friction : bool, optional
        Whether contacts also resist rolling, with strength set per geometry by the material option 'friction_rolling'
        (see 'gs.materials.Rigid'). Enable it when rolling resistance matters - a ball or wheel coasting to rest, a
        cylinder settling on a slope - motions a point contact otherwise never slows down. The extra rolling
        resistance slows down the constraint solve on every contact, more so than torsional friction (two extra axes),
        and requires 'enable_torsional_friction'. Defaults to False.
    impratio : float, optional
        Ratio of tangential (friction) to normal constraint impedance at contacts. Raising it above 1 stiffens
        friction so resting stacks and piles hold their pose under sustained shear, at the cost of a slower solve that
        turns numerically unstable once pushed too far - a stiffness-versus-stability tradeoff, so use the smallest
        value that holds the contacts. It matters mainly with the elliptic cone, which stiffens friction alone while
        leaving the normal contact response at its own impedance. Defaults to None, resolving to 100 with the elliptic
        cone (1 when 'enable_mujoco_compatibility' is set) and 1 otherwise.
    sparse_solve : bool, optional
        Whether to exploit sparsity (skyline-envelope Cholesky) in the constraint solver.

        Defaults to None, which resolves automatically: enabled on the CPU backend (and not under MuJoCo compatibility)
        when the scene has block structure - at least two DOF-carrying bodies or at least two free joints - so the
        Hessian band stays much tighter than its dimension. Never enabled on GPU, where the dense tiled factorization
        is faster. Set True or False to override the automatic choice; True is ignored with a warning on GPU.
    contact_resolve_time : float, optional
        Please note that this option will be deprecated in a future version. Use 'constraint_timeconst'
        instead.
    constraint_timeconst : float | None
        Time constant of the constraint response, in seconds, used for every geom that does not carry one of its
        own. The smaller it is, the stiffer the constraint, down to a floor of twice the integration interval, below
        which the solve becomes unstable. Set it to None to leave those geoms at that floor: as stiff as the timestep
        allows, and what a model authoring its own values expects, at the cost of contacts that respond more abruptly. This parameter is called
        'timeconst' in Mujoco (https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters). Defaults to
        0.01.
    use_hibernation : bool, optional
        Whether to put bodies that have come to rest to sleep, so the solver skips them until they are disturbed. It
        quietly has no effect on a body that is differentiable, prunable, or under no-slip friction. Defaults to False.
    hibernation_thresh_vel : float, optional
        Velocity tolerance for hibernation, in meters per second: a body sleeps once its maximum DOF speed stays below
        this for a few consecutive steps, and a whole island sleeps once all its bodies are ready. Each rotational DOF
        is weighted by the body's swept radius, so the tolerance is a single linear speed that applies uniformly to
        translation and rotation. If None, it is set to 1e-4 when MuJoCo compatibility is enabled (matching MuJoCo's
        default) and 2e-3 otherwise. Defaults to None.
    max_dynamic_constraints : int, optional
        Maximum number of dynamic constraints (like suction cup) per environment. Adding one past it raises. Defaults
        to 8.
    use_gjk_collision: bool, optional
        Whether to use GJK for collision detection instead of MPR. More stable but much slower. Defaults to
        `sim_options.requires_grad`.
    enable_contact_patch: bool, optional
        Whether to recover the full contact patch from the touching faces inside GJK, in a single detection pass,
        instead of through perturbed re-detections. The contact patch is cheaper and reports the exact contact
        polygon, but it is discouraged: it is less reliable than the perturbation-based detection, which is extremely
        robust at the cost of extra detection passes. Requires GJK collision detection, and raises otherwise. If
        None, it is enabled when MuJoCo compatibility is enabled together with GJK and multi-contact, and disabled
        otherwise. Defaults to None.
    broadphase_traversal : gs.broadphase_traversal, optional
        Broadphase traversal strategy. ``SAP`` (sweep-and-prune) or ``ALL_VS_ALL`` (parallel pair iteration). Defaults
        to ``None`` (auto: ``SAP`` on CPU or with heterogeneous entities, ``ALL_VS_ALL`` on GPU otherwise). See
        ``gs.broadphase_traversal`` for details on each strategy.

    Warning
    -------
    Hibernation hasn't been robustly tested and will be fully supported soon.
    """

    enable_collision: StrictBool = True
    enable_joint_limit: StrictBool = True
    enable_self_collision: StrictBool = True
    enable_neutral_collision: StrictBool = False
    enable_adjacent_collision: StrictBool = False
    disable_constraint: StrictBool = False
    max_collision_pairs: NonNegativeInt = 150
    max_contacts: NonNegativeInt | None = None
    multiplier_collision_broad_phase: PositiveInt = 8
    integrator: gs.integrator = gs.integrator.approximate_implicitfast
    IK_max_targets: PositiveInt = 6

    # batching info
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False

    # constraint solver
    constraint_solver: gs.constraint_solver = gs.constraint_solver.Newton
    iterations: PositiveInt = 25
    tolerance: PositiveFloat | None = None
    ls_iterations: PositiveInt = 50
    ls_tolerance: PositiveFloat = 1e-2
    noslip_iterations: NonNegativeInt = 0
    noslip_tolerance: PositiveFloat = 1e-6
    friction_cone: gs.friction_cone = gs.friction_cone.pyramidal
    contact_resolution: gs.contact_resolution | None = None
    enable_torsional_friction: StrictBool = False
    enable_rolling_friction: StrictBool = False
    impratio: PositiveFloat | None = None
    contact_pruning_tolerance: PositiveFloat | None = 0.02
    sparse_solve: StrictBool | None = None
    constraint_timeconst: PositiveFloat | None = 0.01
    box_box_detection: StrictBool = False

    # hibernation threshold
    use_hibernation: StrictBool = False
    hibernation_thresh_vel: PositiveFloat | None = None

    # for dynamic properties
    max_dynamic_constraints: NonNegativeInt = 8

    # Experimental options mainly intended for debug purpose and unit tests
    enable_multi_contact: StrictBool = True
    enable_mujoco_compatibility: StrictBool = False

    # GJK collision detection
    use_gjk_collision: StrictBool | None = None
    enable_contact_patch: StrictBool | None = None

    # broadphase configuration
    broadphase_traversal: gs.broadphase_traversal | None = None

    def __init__(self, *, contact_resolve_time: float | None = None, use_contact_island: bool | None = None, **data):
        super().__init__(**data)
        if contact_resolve_time is not None:
            gs.logger.warning("'contact_resolve_time' is deprecated. Use 'constraint_timeconst' instead.")
        if use_contact_island is not None:
            if not use_contact_island:
                gs.raise_exception(
                    "'use_contact_island=False' is not supported: the constraint solver always solves the contact "
                    "islands of the scene."
                )
            gs.logger.warning("'use_contact_island' is deprecated and has no effect.")

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.contact_pruning_tolerance is not None and self.enable_mujoco_compatibility:
            if "contact_pruning_tolerance" in self.model_fields_set:
                gs.raise_exception(
                    "'contact_pruning_tolerance' is not supported when 'enable_mujoco_compatibility' is True"
                )
            # User did not explicitly request pruning, silently disable to guarantee mujoco compatibility
            self.contact_pruning_tolerance = None
        if self.friction_cone == gs.friction_cone.elliptic and self.noslip_iterations > 0:
            gs.raise_exception("The elliptic friction cone is not supported with the noslip solver.")
        if self.enable_rolling_friction and not self.enable_torsional_friction:
            gs.raise_exception("'enable_rolling_friction' requires 'enable_torsional_friction'.")


class MPMOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the MPMSolver.

    Note
    ----
    MPM is a hybrid lagrangian-eulerian method for simulating soft materials. In the eulerian phase, it uses a grid representation. The `upper_bound` and `lower_bound` specify the simulation domain, but a safety padding will be added to the actual grid boundary. Therefore, the actual boundary could be slightly tighter than the specified one. Note that the size of the domain affects the performance of the simulation, hence you should set it as tight as possible.

    Parameters
    ----------
    particle_size : float, optional
        Particle diameter in meters. If not given, we will compute `particle_size` based on `grid_density`, where `particle_size` will be linearly proportional to the grid cell size. A reference value is `particle_size = 0.01` for `grid_density = 64`. Defaults to None.
    grid_density : float, optional
        Number of grid cells per meter. Defaults to 64.
    enable_CPIC : bool, optional
        Whether to enable CPIC (Compatible Particle-in-Cell) to support coupling with thin objects. Defaults to False.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-1.0, -1.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (1.0, 1.0, 1.0).
    use_sparse_grid : bool, optional
        This option is deprecated.
    leaf_block_size : int, optional
        This option is deprecated.
    """

    particle_size: PositiveFloat | None = None  # in meters. Will be computed automatically if it's None.
    grid_density: PositiveFloat = 64
    enable_CPIC: StrictBool = False

    # These will later be converted to discrete grid bound. The actual grid boundary could be slightly tighter.
    lower_bound: Vec3FType = (-1.0, -1.0, 0.0)
    upper_bound: Vec3FType = (1.0, 1.0, 1.0)

    def __init__(self, *, use_sparse_grid: bool = False, leaf_block_size: int = 8, **data):
        super().__init__(**data)
        if use_sparse_grid:
            gs.logger.warning("'use_sparse_grid' is deprecated and has no effect.")
        if leaf_block_size != 8:
            gs.logger.warning("'leaf_block_size' is deprecated and has no effect.")

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        if data.get("particle_size") is None:
            data["particle_size"] = 0.01 * 64.0 / data.get("grid_density", 64)
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")


class SPHOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the SPHSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
    particle_volume_factor : float, optional
        Rest volume of a particle, as a fraction of the cube of `particle_size`. The fluid settles where its particles
        fill their rest volume, so this sets how densely the fluid packs relative to the spacing it is sampled at:
        higher values make freshly sampled fluid start compressed and expand, lower values make it start sparse and
        contract. The mass of a particle scales with it as well. Defaults to 0.75.
    use_legacy_solver : bool, optional
        Whether to use the legacy SPH solver. The legacy solver offers the weakly compressible 'WCSPH' pressure solver
        and two-way coupling with rigid, MPM and FEM entities, but its 'DFSPH' pressure solver lets the fluid compress
        noticeably. The default solver enforces incompressibility accurately, but it only offers 'DFSPH', and its
        coupling with rigid entities is one-way (the fluid does not push rigid entities back) and with MPM and FEM
        entities absent so far. Defaults to False.
    pressure_solver : str, optional
        Pressure solver type, either 'WCSPH' (legacy solver only) or 'DFSPH'. Defaults to 'WCSPH' for the legacy
        solver and 'DFSPH' otherwise.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 2 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    max_divergence_error : float, optional
        Maximum divergence error for DFSPH. Defaults to 0.1%.
    max_density_error : float, optional
        Maximum density error for DFSPH. Defaults to 0.05%.
    max_divergence_solver_iterations : int, optional
        Maximum number of iterations for the divergence solver. Defaults to 10.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the density solver. Defaults to 10.
    min_divergence_solver_iterations : int, optional
        Minimum number of iterations for the DFSPH divergence solver. More iterations cost time but reduce the
        velocity divergence left over when the error tolerance is met early. Defaults to 1.
    min_density_solver_iterations : int, optional
        Minimum number of iterations for the DFSPH density solver. More iterations cost time but reduce the density
        error left over when the error tolerance is met early. Defaults to 1.
    skip_divergence_solver : bool, optional
        Whether to skip the DFSPH divergence solver, leaving incompressibility to the density solver alone. This
        saves time per substep, but the fluid becomes noisier and less stable, and the density solver may need more
        iterations to compensate. Only applies to the default (non-legacy) solver. Defaults to False.
    pressure_chebyshev_radius : float, optional
        Bound on the spectral radius of the Jacobi iteration of the DFSPH density solve, for the Chebyshev
        semi-iterative acceleration (Wang 2015) of the density solves of the first two viscosity-pressure coupling
        passes of each substep, in [0, 1); 0 runs plain Jacobi. An accelerated solve keeps the solution and works to a
        tenth of `max_density_error`, since at the tolerance itself its iterate keeps more particle-scale error than
        the plain one; it still takes several times fewer iterations on a stirred cup of water. 0.98 was the best bound
        of 0.95 to 0.998 at every time step tried there; a bound too close to 1 slows the solve down again. Only
        applies to the default (non-legacy) solver. Defaults to 0.98.
    warm_start : bool, optional
        Whether DFSPH starts each pressure solve from half the pressure of the previous substep. This reduces the
        number of solver iterations for fluid at rest or in slow motion, but it can carry pressure spikes from
        violent impacts over to the next substep. Defaults to True.
    max_viscosity_error : float, optional
        Maximum relative residual of the implicit viscosity solve, and of the agreement between the viscosity and the
        pressure of a substep. Smaller values give a more accurate viscous damping and wall drag at the cost of more
        solver iterations. Only used by the default solver. Defaults to 0.1%.
    max_viscosity_solver_iterations : int, optional
        Maximum number of iterations of the implicit viscosity solve. A substep that reaches it keeps the velocity of
        the last iteration and is reported as unconverged. Only used by the default solver. Defaults to 100.
    viscosity_correction : str, optional
        Benchmark switch of the directional correction of the interior viscosity (SPH_SOLVER_GOAL.md N6, A1):
        "directional" (15 coefficients per particle), "isotropic" (a scalar rescaling to the trace of the continuum
        moment) or "none" (the uncorrected pair coefficient). Only used by the default solver. Defaults to
        "directional".
    wall_viscosity_model : str, optional
        Benchmark switch of the wall viscosity (A2): "ghost_columns" (no-slip ghost fluid columns behind each boundary
        particle) or "boundary_pairs" (the boundary particle itself as a viscous neighbor moving with the wall). Only
        used by the default solver. Defaults to "ghost_columns".
    wall_ghost_points : int, optional
        Quadrature points of a ghost column of the wall viscosity (A2): 1, 2, 4 or 8. Only used by the default solver.
        Defaults to 8.
    viscosity_residual_projection : bool, optional
        Benchmark switch (A4) of the rigid velocity correction that removes the net force and torque of the residual of
        the viscosity solve. Only used by the default solver. Defaults to True.
    wall_clearance_velocity, wall_clearance_position, wall_clearance_rotation : bool, optional
        Benchmark switches (A5) of the parts of the wall clearance guard: the unilateral normal velocity constraint,
        the soft position recovery and the rigid rotation of the fluid that compensates the angular momentum of the
        recovery. Only used by the default solver. Default to True.
    correction_diagnostics : bool, optional
        Count how often and how strongly the corrections act, into the solve stats (SPHSolveStats). Costs a kernel and
        a host read per substep. Only used by the default solver. Defaults to False.
    max_coupling_iterations : int, optional
        Maximum number of passes alternating the viscosity solve and the density solve of a substep. A single pass
        suffices for fluids about as viscous as water, while very viscous fluids under pressure need more, which cost
        time. A substep that reaches the limit is reported as unconverged. Only used by the default solver.
        Defaults to 30.
    max_constraint_iterations : int, optional
        Maximum number of times the fluid stage of a substep is repeated after the constraint solve of the rigid solver
        (contacts, joint limits, equalities) changed the velocity of a coupled link, holding that change as an external
        impulse on the link, until the change agrees with the one the fluid saw. Only used by the default solver.
        Defaults to 4.
    max_constraint_error : float, optional
        Tolerance of that agreement, as the change of the coupled link velocities between two repetitions relative to
        their magnitude (the norm of the linear velocity plus the size of the link times the norm of the angular
        velocity). Only used by the default solver. Defaults to 1%.
    isotropic_pressure : bool, optional
        Whether to correct the pressure pair weights of each particle so that its pressure pairs carry an isotropic
        stress. Without it, a sheared particle arrangement under pressure carries a shear stress proportional to the
        pressure, a spurious friction that raises the wall shear of viscous flows under a hydrostatic head. The
        corrected force is not the gradient of the density estimate and heats a fluid at rest: at 1e-3 m^2/s the
        viscosity holds the velocity noise at a few mm/s, at the viscosity of water it grows. Only used by the default
        solver. Defaults to False.
    isotropic_pressure_relaxation : float, optional
        Fraction by which the isotropic pressure correction of a particle moves toward its value for the current
        arrangement at every substep, 1 to take that value. Only used by the default solver. Defaults to 1.
    isotropic_density : bool, optional
        With the isotropic pressure, whether the density that the pressure solves constrain takes the corrected pair
        weights as well, consistently with the velocity constraint and the force of the pressure. Only used by the
        default solver. Defaults to False.
    density_correction : str, optional
        How the density error of the particles is corrected. 'velocity' (DFSPH) lets the density solve change the
        velocities so that the predicted density reaches the rest density, which also turns the error into particle
        velocity, the particle noise of a fluid at rest. 'position' lets the density solve of the coupled passes remove
        only the compression the velocity would cause, and moves the particles by a density solve of the remaining
        compression after the coupled passes, without changing their velocities or momentum. Only used by the default
        solver. Defaults to 'velocity'.
    max_drift_solver_iterations : int, optional
        Maximum number of iterations of the drift correction of the positions (density_correction='position'). A
        substep that reaches it is reported as unconverged. Only used by the default solver. Defaults to 50.
    wall_shear_consistency : bool, optional
        Whether to scale the wall viscosity of each fluid particle near a rigid surface so that the viscosity operator
        is exact for a linear shear along the surface, which keeps the no-slip wall from slipping whatever the distance
        at which the fluid rests from it. Only used by the default solver. Defaults to False.
    energy_audit : bool, optional
        Whether to record the change of kinetic energy of the fluid caused by each stage of a substep (divergence solve,
        explicit forces, coupled viscosity and density solves, wall clearance) in `SPHSolver.solve_stats`, to check the
        energy balance of a scene. It costs one reduction over the particles per stage. Only used by the default solver.
        Defaults to False.
    micropolar : bool, optional
        Whether to model the micro-rotation of the fluid particles, which restores the small-scale vortices that
        particle-based fluids damp out, yielding livelier splashes and swirls. Its strength is set per material by
        `vorticity`, `angular_viscosity` and `microinertia`. It costs one more neighbor pass per substep and
        adds turbulent motion to slow, laminar flows. Only used by the default solver. Defaults to False.
    boundary_offset : float, optional
        Distance in meters by which the boundary particles sampled on rigid geoms are moved inside the geom, along its
        surface normal. Fluid particles rest about one particle spacing away from the boundary particles. With half
        the particle size, fluid touches the geom surface and fluid sampled flush against a geom starts at rest. A
        smaller offset leaves a visible gap between fluid and geom, and fluid sampled flush against the geom starts
        compressed and gets pushed away. Keep the offset below half the thickness of the thinnest coupled geom, or its
        two faces cross each other. Only used by the default solver. Defaults to half the particle size.
    boundary_model : str, optional
        How the rigid geoms enter the density of the fluid. 'particles' sums the boundary particles sampled on their
        surface (Akinci et al. 2012); the tangential part of their pressure force acts as a roughness of the wall, which
        slows fluid sliding along it. 'volume_map' integrates the smoothing kernel over the solid of each link, sampled
        once on a grid in link frame at build time (Koschier and Bender 2017): the wall is smooth, and its pressure
        force is the gradient of the density it adds, so it does no work on fluid at rest. The geoms of a link must
        enclose a volume (closed meshes, primitives or planes). The boundary particles are then sampled on the surface
        itself (`boundary_offset` is not used) and still carry the wall viscosity and the wall clearance. Only used by
        the default solver. Defaults to 'particles'.
    volume_map_cell_size : float, optional
        Spacing in meters of the grid of the volume maps. Defaults to a quarter of the particle size.
    volume_map_offset : float, optional
        Distance in meters by which the solid of the volume maps is shrunk. Defaults to the distance that puts at rest
        density a cubic lattice of fluid of particle volume `particle_volume_factor * particle_size**3` whose first
        layer lies half a spacing from a planar surface, so that fluid sampled flush against a geom starts at rest.
    """

    particle_size: PositiveFloat = 0.02
    particle_volume_factor: PositiveFloat = 0.75
    use_legacy_solver: StrictBool = False
    pressure_solver: Literal["WCSPH", "DFSPH"] | None = None

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    # DFSPH parameters
    max_divergence_error: PositiveFloat = 0.1 * 0.01
    max_density_error: PositiveFloat = 0.05 * 0.01
    max_divergence_solver_iterations: PositiveInt = 10
    max_density_solver_iterations: PositiveInt = 10
    min_divergence_solver_iterations: PositiveInt = 1
    min_density_solver_iterations: PositiveInt = 1
    skip_divergence_solver: StrictBool = False
    warm_start: StrictBool = True
    max_viscosity_error: PositiveFloat = 0.1 * 0.01
    max_viscosity_solver_iterations: PositiveInt = 100
    max_coupling_iterations: PositiveInt = 30
    viscosity_correction: Literal["directional", "isotropic", "none"] = "directional"
    wall_viscosity_model: Literal["ghost_columns", "boundary_pairs"] = "ghost_columns"
    wall_ghost_points: Literal[1, 2, 4, 8] = 8
    viscosity_residual_projection: StrictBool = True
    wall_clearance_velocity: StrictBool = True
    wall_clearance_position: StrictBool = True
    wall_clearance_rotation: StrictBool = True
    correction_diagnostics: StrictBool = False
    max_constraint_iterations: NonNegativeInt = 4
    max_constraint_error: PositiveFloat = 0.01
    isotropic_pressure: StrictBool = False
    isotropic_density: StrictBool = False
    wall_shear_consistency: StrictBool = False
    density_correction: Literal["velocity", "position"] = "velocity"
    max_drift_solver_iterations: PositiveInt = 50
    isotropic_pressure_relaxation: PositiveFloat = 1.0
    pressure_chebyshev_radius: NonNegativeFloat = 0.98
    energy_audit: StrictBool = False
    micropolar: StrictBool = False
    boundary_offset: NonNegativeFloat | None = None
    boundary_model: Literal["particles", "volume_map"] = "particles"
    volume_map_cell_size: PositiveFloat | None = None
    volume_map_offset: NonNegativeFloat | None = None

    _support_radius: float = PrivateAttr(default=0.0)
    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 0.02)
        support_radius = 2 * particle_size
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = support_radius
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        if self.pressure_solver is None:
            self.pressure_solver = "WCSPH" if self.use_legacy_solver else "DFSPH"
        elif self.pressure_solver == "WCSPH" and not self.use_legacy_solver:
            gs.raise_exception("Pressure solver 'WCSPH' requires `use_legacy_solver=True`.")

        self._support_radius = 2 * self.particle_size

        if self.pressure_chebyshev_radius >= 1.0:
            gs.raise_exception("`pressure_chebyshev_radius` should be smaller than 1.")

        if self.boundary_offset is None:
            self.boundary_offset = 0.5 * self.particle_size

        if self.volume_map_cell_size is None:
            self.volume_map_cell_size = 0.25 * self.particle_size

        if self.hash_grid_cell_size < self._support_radius:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 2 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            )
            self._hash_grid_res = np.minimum(max_hash_grid_res, 150).astype(int).tolist()
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int).tolist()


class PBDOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the PBDSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be 1.25 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    max_stretch_solver_iterations : int, optional
        Maximum number of iterations for the solving stretch constraints. Defaults to 4.
    max_bending_solver_iterations : int, optional
        Maximum number of iterations for the solving bending constraints. Defaults to 1.
    max_volume_solver_iterations : int, optional
        Maximum number of iterations for the solving volume constraints. Defaults to 1.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the solving density constraints. Defaults to 1.
    max_viscosity_solver_iterations : int, optional
        Maximum number of iterations for the solving viscosity constraints. Defaults to 1.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 1e-2.
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 1.25 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    """

    # constraints solving iterations
    max_stretch_solver_iterations: PositiveInt = 4
    max_bending_solver_iterations: PositiveInt = 1
    max_volume_solver_iterations: PositiveInt = 1
    max_density_solver_iterations: PositiveInt = 1
    max_viscosity_solver_iterations: PositiveInt = 1

    # self collision
    particle_size: PositiveFloat = 1e-2

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 1e-2)
        # NOTE: 1.25 is a safety factor, as inside one single substep, multiple substages can change the position of
        # the particles but we only do spatial hashing once. The grid cell needs to be a bit bigger so that neighbours
        # are not missed.
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = 1.25 * particle_size
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        if self.hash_grid_cell_size < 1.25 * self.particle_size:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 1.25 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            )
            self._hash_grid_res = np.minimum(max_hash_grid_res, 150).astype(int).tolist()
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int).tolist()


class FEMOptions(GravityMixin, TimeBasedMixin):
    """
    Options configuring the FEMSolver.

    Note
    ----
    - Damping coefficients are used to control the damping effect in the simulation.
    They are used in the Rayleigh Damping model, which is a common damping model in FEM simulations.
    Reference: https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_modeling.05.083.html
    - TODO Move it to material parameters in the future instead of solver options.

    Parameters
    ----------
    damping : float, optional
        Damping factor. Defaults to 0.0.
    floor_height : float, optional
        Height of the floor in meters. If none, it will inherit from `SimOptions`. Defaults to None.
    use_implicit_solver : bool, optional
        Whether to use the implicit solver. Defaults to False.
        Implicit solver is a more stable solver for FEM. It can be used with a large time step.
    n_newton_iterations : int, optional
        Maximum number of Newton iterations. Defaults to 1. Only used when `use_implicit_solver` is True.
    n_pcg_iterations : int, optional
        Maximum number of PCG iterations. Defaults to 500. Only used when `use_implicit_solver` is True.
    n_linesearch_iterations : int, optional
        Maximum number of line search iterations. Defaults to 0. Only used when `use_implicit_solver` is True.
    newton_dx_threshold : float, optional
        Threshold for the Newton solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    pcg_threshold : float, optional
        Threshold for the PCG solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    linesearch_c : float, optional
        Line search sufficient decrease parameter. Defaults to 1e-4. Only used when `use_implicit_solver` is True.
    linesearch_tau : float, optional
        Line search step size reduction factor. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_alpha : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_beta : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 5e-4. Only used when `use_implicit_solver` is True.
    enable_vertex_constraints : bool, optional
        Whether to enable vertex constraints. Defaults to False.
    """

    damping: NonNegativeFloat = 0.0
    floor_height: float | None = None
    use_implicit_solver: StrictBool = False
    n_newton_iterations: PositiveInt = 1
    n_pcg_iterations: PositiveInt = 500
    n_linesearch_iterations: NonNegativeInt = 0
    newton_dx_threshold: PositiveFloat = 1e-6
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_c: PositiveFloat = 1e-4
    linesearch_tau: PositiveFloat = 0.5
    damping_alpha: NonNegativeFloat = 0.5
    damping_beta: NonNegativeFloat = 5e-4
    enable_vertex_constraints: StrictBool = False


class SFOptions(TimeBasedMixin):
    """
    Options configuring the SFSolver.
    """

    res: PositiveInt = 128
    solver_iters: PositiveInt = 500
    decay: PositiveFloat = 0.99

    T_low: float = 1.0
    T_high: float = 0.0

    inlet_pos: Vec3FType = (0.6, 0.0, 0.1)
    inlet_vel: Vec3FType = (0.0, 0.0, 1.0)
    inlet_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)
    inlet_s: PositiveFloat = 400.0
