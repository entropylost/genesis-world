# DFSPH solver development status

Companion of [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md) (target) and [SPH_SOLVER_DESIGN.md](SPH_SOLVER_DESIGN.md)
(equations and acceptance). This file records what is implemented, what is measured, and where the evidence is. Campaign
drivers and their JSON receipts live in FluidBench (`adapters/genesis/sph_validation/`, `results/sph_validation/`,
branch `bench/auto-research` of https://github.com/Roushelfy/fluidbench). Every number below names its receipt (file
name prefix in `results/sph_validation/`). Receipts tagged `dev-*` or `*_dirty` come from development snapshots (a
commit plus uncommitted changes, recorded in the receipt as `genesis_dirty`); acceptance numbers come from committed
revisions only.

Hardware of record: NVIDIA RTX PRO 6000 Blackwell Server Edition (cluster partition rtx-mid), fp32.

## Current delivery scope (2026-10-30)

The owner prioritizes thick-wall engineering delivery and permits moderate accuracy targets. Thin-wall penetration and
the V11 start-up gate are deferred. The dated scope update in [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md) defines the
current engineering targets and the subsequent higher-fidelity research stage.

The measurements below retain their original configurations and strict verdicts. The unified thick-wall campaign on
Genesis 745b4613 is recorded in findings 50-53 (velocity baseline adopted, drift correction measured beside it and not
adopted); the delivery of Stage A is recorded in the next section. The primary baseline uses symmetric pressure, wall
shear consistency and `density_correction="velocity"` with the dissipative clearance guard. The extra position-level
density drift solve is disabled in this baseline and evaluated separately in a matched on/off comparison. Conservation,
energy balance, solver convergence and the complete two-way/implicit-viscosity feature set remain required.

## Next-stage owner goals (2026-10-31)

The owner requests the next stage defined in [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md), section "Next stage:
qualification, coupling robustness and measured acceleration". The historical Stage A delivery below remains a
baseline with limitations. This documentation update does not fix a solver or establish new validation results.

| Goal | Initial state | Required next evidence |
| --- | --- | --- |
| N1 clean baseline qualification | pending | unified clean-commit regression and timing campaign; replace dirty-snapshot qualification claims |
| N2 dynamic-link robustness | pending | failure-stage trace, converged dt sweep, loads/trajectories and supported operating range |
| N3 state and selected reset | pending | state audit, repeated restart/reset controls, original V12 and any prospectively defined statistical verdict |
| N4 pressure shear and viscosity accuracy | pending | pressure/viscous stress separation, derived correction, engineering-viscosity accuracy and conservation/energy regressions |
| N5 measured acceleration | pending for this stage; earlier optimizations retained | matched-feature timings after corrections, task/iteration profile, single and independent batched throughput |
| N6 correction ablations (owner update 2026-11-02) | pending | A1-A7 individually and in key combinations: accuracy gains, runtime/memory cost, conservation, rest noise, leakage, state and stability side effects; retain/simplify/remove recommendation |

Latest committed timing baseline (cc03c40b; finding 68): S1 dt x8 RTF 0.408/0.282/0.126 at R1/R2/R3. This is a
performance diagnostic; S1 water accuracy and RTF 0.5 at R2 remain open. Full implicit viscosity and two-way coupling
remain required. Position-level drift stays disabled; thin-wall repair and accurate water dynamics remain deferred.

Synchronization checked before this goal update: `Roushelfy/Genesis:fluidbench/sph-solver` and
`entropylost/genesis-world:main` both point to c00833b277361c63d78453485868effd238eeca1. This goal/status milestone
must be synchronized as well, preserving all solver code and historical receipts.
Synchronization of this documentation milestone was first blocked: the GitHub integration returned HTTP 403 / resource
not accessible by integration for both updating the collaborator main ref and creating a synchronization pull request
(source goal commit 79f8240915654cf1df6eb2f407bc5d57c0c17931). Resolved on 2026-10-31 through the owner-authorized
fast-forward workflow from the cluster checkout: `entropylost/genesis-world:main` went from c00833b2 to de51bc9c, which
holds 79f82409, b0b6f0bc and the status record of finding 71; source and destination heads were equal after the push.

## Correction ablation requirement (2026-11-02)

The owner requests the N6 campaign in SPH_SOLVER_GOAL.md for directional viscosity consistency, wall ghost columns,
wall shear scaling, viscosity-residual momentum projection, wall clearance and its angular compensation, experimental
isotropic pressure/density, and position drift correction. Include coupling-iteration diagnostic controls separately.
This update adds requirements only; no ablation result or default change is claimed. N6 starts pending.

Separate pressure, viscous and clearance load readouts are required for attribution. The FluidBench optional pressure
getter call is not a completed diagnostic until the committed solver exports and validates the data. Run clean
one-component controls, key interactions and final full-horizon confirmations with unchanged references and bars.

Source and collaborator main both hold 914a900f94e48161a89970018bf86b6fbf9db7cf before this documentation update.
This goal/status milestone requires collaborator synchronization. Source goal commit:
`6fa95c8072298454c4effda0c30925b78ffe51b4`. The 2026-11-02 synchronization attempt is blocked: the GitHub integration
rejected both updating collaborator main and creating a pull request (resource not accessible by integration / HTTP
403). No collaborator integration or pull request is claimed; synchronize this milestone when write access permits.

## Stage A delivery (2026-10-31)

Revision `138b2c5f` (code) with this document, and the independent stop of environments on the commit after it
(finding 66; one environment 2 % slower than the speeds quoted below, batches faster); collaborator
`entropylost/genesis-world:main` holds the same commits.
Evidence: FluidBench `report/sph_goal.md` (sections "Stage A verdict by case and configuration", "Stage A: ..."), the
receipts `results/sph_validation/*__sa0_*` (validation of the baseline on 745b4613) and the equivalence receipts of
every later code change (findings 57-64), and the run records `results/index/gs-dfsph__*@sphgoal-{sb,tr,fin}-*`.

Configuration. `SPHOptions(wall_shear_consistency=True, density_correction="velocity")` with the defaults otherwise:
symmetric pressure pairs, implicit interior and wall viscosity, two-way pressure and viscous coupling with the rigid
links, constraint repetitions for joint and contact feedback, the dissipative clearance guard, the fluid load,
dissipation and pressure readouts, state save, restore and selected reset. FluidBench adapter: pressure solves capped at
100 iterations, particle size dx, boundary offset 1.35 mm, the liquid sampled at least dx/2 from the walls with its
bottom layer aligned and settled for 0.5 s (`sph_relax_s`). Containers and tanks thickened outward to 3 dx (1.5
support radii; 15 / 10.5 / 7.5 / 5.25 mm at R0 / R1 / R2 / R3), the inner domain, liquid mass and prescribed motions
unchanged; no dynamic link is thickened, so no link inertia changes. Command files:
`/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_{sa,sb_fb,tr,fin}.txt` and `gen_*.py` beside them.

Staged verdict of the baseline (receipts `sa0`, unchanged by findings 57-64 within the spread of repeated runs):

| Case | Stage A verdict | Value |
| --- | --- | --- |
| V3 pipe, V5 spin-down, V6 momentum | pass | centerline 1.0-1.6 %; spin-down rms 3.2 %; drift 4.3e-6 and 7.1e-6 per s |
| V7 hydrostatics | pass | load 2.8e-5, bulk pressure 3.6 % (the layer on the floor 24 % high, original bar failed) |
| V8 plank and light box | pass at the finest dx | draft 4.8 % and 8.9 % at dx 1.75 mm (14-30 % at 3.5 mm) |
| V10 held plate and stirring | pass | joints within 3 % of the readout; stirring balance -0.5 % |
| V4 mode decay | fail at nu 1e-4 | 14.5 / 18.9 / 26.6 % at nu 1e-2 / 1e-3 / 1e-4; water (nu 1e-6) +294 %, reported |
| V1 Stokes first problem, V2 Couette | fail | floor torque up to 1.9 times the reference; Couette torque up to 37 % (pressure friction, finding 11) |
| V9 settling sphere | fail | 51-66 % slow (pressure friction) |
| Thick-wall cup at rest (V11 deferred) | pass with the documented sampling and settling | no crossing at R0-R2, rms 0.008 m/s at R1, largest speed 0.046-0.074 m/s; the lattice of the spec (control) bursts against thick walls, the unsettled gap sampling leaks at R0 |
| V12 state | fail (envelope bar) | the first replayed step agrees to round-off; the largest single-particle departure varies tenfold between equivalent runs (finding 54) |

FluidBench (finding 50, 56 and 60): S2 pouring is usable at R1 and R2 at every time step from dt x1 to x8; the floating
box T07a, the box T07b and T08a are usable at dt x1, T07a fails at dt x4; the settling sphere T08c converges with a
coupling cap of 60 but settles 98.6 % slow; S1 water fails the spin-down time (-65 %) and the angular momentum (0.23 at
R2) at every time step, a Stage B item.

Speed (finding 65; usability and speed reported separately, as the goal requires while the target is unmet): one
environment of thick-wall S1 at dt x8 runs at RTF 0.389 / 0.257 / 0.120 at R1 / R2 / R3 against the target 0.5 (from
0.123 at R2 on 745b4613, findings 55 and 57-64), S2 R2 at 0.130; dt x8 keeps the S1 and S2 verdicts of dt x1 (finding
60). Batches of 4 and 16 environments: 0.44 and 0.35-0.49 environment-seconds per second with the independent stop of
environments (finding 66).

Limitations carried to Stage B: the shear stress that the symmetric pressure pairs carry in proportion to the pressure
(V1, V2, V9, T08c); the excess damping of water (V4 at nu 1e-6, S1); V12 on its single-particle envelope; dynamic links
above the FluidBench time step; RTF 0.5 at R2; the thin-wall cup (V11) and the position drift correction, which fails
in box tanks and with dynamic links (findings 51-53).

## Collaborator synchronization

- Source: `Roushelfy/Genesis:fluidbench/sph-solver`.
- Collaborator: `entropylost/genesis-world`. Default integration branch: `main`. Record any explicitly designated
  DFSPH working branch before using it as the destination.
- Requirement: synchronize solver changes with their tests, examples and documentation after each coherent milestone,
  using the workflow in [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md).
- 2026-10-30, integrated: with the owner's approval, `entropylost/genesis-world:main` was fast-forwarded from `92c78327`
  (the starting revision of this work) to the source head `3c2dc83e` by a direct push, without a pull request: 49
  commits, the solver, its options and readouts, `tests/coupling/test_sph_rigid.py`, `examples/coupling/sph_paddle.py`
  and the design, goal and status documents. The destination holds exactly the source commits, so the validation
  recorded here applies to it unchanged (FluidBench evidence at `c5da1747`). Later milestones are synchronized the same
  way, each recorded here with its source and destination heads.
- 2026-10-30 to 2026-10-31, fast-forwards of `entropylost/genesis-world:main` to the source heads `9ce08147`,
  `1e8a0c0b`, `993fc331`, `9b621c30`, `4beb80fb` (status records), `793e9685` (chunked reductions) and `18bf3cc0`
  (error read every fourth iteration); after each push the destination head equals the source head.
- 2026-10-31, the same way: `4637eb47`, `a7f11ce9` (status records), `3203a547` (boundary neighbor lists), `676daa52`
  (shared neighbor loops), `48127a19` (error sums), `219e1dfb` (viscosity sums), `138b2c5f` (link response, wall tensor
  shares) and the commit recording the Stage A delivery.

## Milestones

State on the committed revision 76f37038. Evidence: FluidBench `report/sph_goal.md` (generated by
`cluster/sph_goal_report.py`), the receipts in FluidBench `results/sph_validation/`, and the run commands below (run
through `scripts/campaign_env.sh <revision>` of the data directory against a frozen worktree of the revision).

| Milestone | State | Commits | Commands | Configuration | Evidence |
| --- | --- | --- | --- | --- | --- |
| M0 equations, units, cases, acceptance, measurement boundaries | done: SPH_SOLVER_DESIGN.md (V12 acceptance revised for GPU round-off), baseline below | 26c1fee2 (design), 92c78327 (baseline) | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_m0.txt` | starting revision, default options | section "Baseline at the starting revision" |
| M1 implicit interior and wall viscosity with analytic validation and load accounting | implemented and measured: block-Jacobi CG of the implicit viscosity with wall ghosts, the dissipation readout 0.97-1.00 of the energy loss; V3 and V5 pass (wall shear consistency), V2 passes with the isotropic pressure; V1 (floor torque) and V4 at nu 1e-6 fail in every configuration | fcbc20ca, b7fcd76e, eaa36b0a, 712b9755, 22f43b2f, ce44ba1a | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_v6w.txt`, `cmds_sphgoal_wi.txt` | `v6w`, `v6wp`, `wi`, `wip` (section "Remaining limitations") | FluidBench `report/sph_goal.md`, receipts `results/sph_validation/{stokes1,couette,pipe,decay,spindown}__{v6w,v6wp,wi,wip}_*` |
| M2 reaction readout and rigid stage integration | done: readout for fixed, prescribed and dynamic links, fluid stage inside the rigid substep; the readout equals the link impulse; particle pressure readout (virial, finding 42) | 5b171c14, 26832080, 8cd31d27, 18efcc66, 76f37038 | `tests/coupling/test_sph_rigid.py::test_fluid_load_readout_is_link_impulse`; `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_rd1.txt` | default | V7 cup load within 1e-4 of the weight (`hydrostatic__*`), V10 joint torques within 0.2 % of the readout (`articulation__v6w_hold_*`) |
| M3 strong pressure-viscosity-rigid coupling | done: V6 within 4e-6 per s, held joints agree with the readout within 0.2 %, stirring energy balance within 2.1 %, coupling converged in every substep (findings 25-28, 34); with the wall shear consistency on ce44ba1a | 22f43b2f, a071c5e2, daa41b6e, 97aebd75 | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_v5.txt`, `cmds_sphgoal_v6w.txt` | `v5`, `v6w` | `momentum__v6w_*`, `articulation__v6w_*`, `contact__v6w_*` |
| M4 robustness campaigns | partly: V12 passes with `v6wp` on both GPUs; V11 fails (no crossing at R1 with the drift correction, start-up speed 0.15-0.2 m/s; R0 leaks); V8 light box and V9 pass only in some configurations (section "Remaining limitations") | ce44ba1a, 76f37038 | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_v6w.txt`, `cmds_sphgoal_v6wrtx.txt`, `cmds_sphgoal_v6vw.txt` | `v6w`, `v6wp`, `v6vw`, `v6vwp` | `state__*`, `thinwall__*`, `floating__*`, `settling__*` |
| M5 optimization ablations | done for the implemented candidates (1, 5, 8, Chebyshev, fused iteration, compile-time constants); 2, 6 measured and not changed; 3, 4, 7, 9, 10 proposed only. RTF 0.5 not reached: 0.039 / 0.084 / 0.142 at dt x1 / x4 / x8 on S1 R2 (default), every solve converged; S1 R2 is not accuracy-qualified | cbc79cd9, 51e488d4, 0ea1569d, dc03db16, cab47d96 | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_abl.txt`, `cmds_sphgoal_t4.txt`, `cmds_sphgoal_batch.txt` | S1 f2 R2, one environment, iteration cap 100 | FluidBench `results/index/gs-dfsph__S1__f2@sphgoal-abl-*`, `@sphgoal-t4-*`, `results/profiles/sphgoal/`; section "Performance (M5)" |
| Stage A thick-wall engineering solver (owner update 2026-10-30) | delivered with limitations: velocity baseline on thick walls, staged verdicts, FluidBench runs over time steps and resolutions, optimization to RTF 0.257 at S1 R2 dt x8 (target 0.5 not reached); section "Stage A delivery" | 745b4613 (candidate), 793e9685, 18bf3cc0, 3203a547, 676daa52, 48127a19, 219e1dfb, 138b2c5f | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_{sa,sb_fb,tr,fin}.txt` | `sa0`, `sphgoal-sb`, `sphgoal-tr-*`, `sphgoal-fin-*` | FluidBench `report/sph_goal.md`, findings 50-65 |
| M6 FluidBench re-run and documentation | done: S1-S3 at R1-R3 on cab47d96 (two configurations) and at R2 with the later configurations; S2 is usable at R2 with `v6w`; S1 fails the spin-down time in every run (-54 % to -86 %), S3 the mixing index; example `examples/coupling/sph_paddle.py`, option and readout docstrings, limitations below | cab47d96, ce44ba1a | `/mnt/data/zhaofeng/projects/workspace/fluidbench/cmds/cmds_sphgoal_final*.txt`, `cmds_sphgoal_v6w.txt`, `cmds_sphgoal_v6vw.txt` | `sphgoal-v4`, `-v4s`, `-v5p`, `-v6w`, `-v6wp`, `-v6v*` | FluidBench `report/sph_goal.md` (scene tables), `results/index/gs-dfsph__*@sphgoal-*` |

## Baseline at the starting revision (92c78327)

- Existing SPH tests (`tests/particles/test_sph.py`, `tests/coupling/test_sph_rigid.py`, legacy solver): 17 passed, 2
  failed (`test_pressure_carries_column_weight[WCSPH|DFSPH]`: the band `heights >= 0.3` holds no particle, so its mean
  pressure is NaN). Log: `/mnt/data/zhaofeng/projects/workspace/genesis-world/logs/pytest_sph_base.log`.
- FluidBench S1 f2 R2 with DFSPH: uninstrumented timing record `gs-dfsph__S1__f2@sphgoal-m0-base__R2`, instrumented
  profiles `results/profiles/gs-sph-dfsph_S1_f2_R2_{coarse,fine}_*.json` (FluidBench), earlier profile at 49f63720:
  pressure solves 52 %, other SPH 13 %, rigid 28 % of a sub-step.
- Water at rest in a cup (dx 3.5 mm, regular sampler, no surface tension) keeps a particle velocity of 6-8 cm/s rms for
  over a second at the starting revision: the potential energy the initial lattice releases as it compacts under
  gravity stays as kinetic energy at the particle scale, which water viscosity damps over dx^2 / nu ~ 10 s. The same
  holds at every later revision. Validation cases therefore settle with their velocities reset. Logs:
  `logs/noise_water_92c783279.log`, `logs/noise3_*.log`, `logs/noise4_*.log`.

## Findings that change the model

1. Explicit viscosity replaced by an implicit backward Euler solve (conjugate gradient, block-Jacobi preconditioner) of
   the Rayleigh dissipation of SPH_SOLVER_DESIGN.md section 3.3. The material `viscosity` is the kinematic viscosity
   (m^2/s), converted once to the dynamic viscosity.
2. Boundary pressure (Akinci particles, full kernel gradient) acts as a rough wall: fluid in rigid rotation at water
   viscosity inside a fixed cup lost 58 % of its angular momentum in 1 s at the starting revision (spin-down test, dx
   5 mm, boundary spacing 0.75 particle sizes), where the physical spin-down takes tens of seconds
   (`logs/spin_down.log`). Projecting the boundary kernel gradient on the wall normal (a smooth impermeable wall) kept
   100-102 % over 1 s (`logs/spin_down2.log`), but the projected force is not the gradient of the density the boundary
   adds, so the pressure solve feeds the density variation of fluid sliding past the samples back along the normal only:
   a closed column of water at rest heated from 4.4e-4 to 8.9e-4 J of kinetic energy over 0.9 s
   (`logs/fg_cn_proj75.log`, dev/closed_noise.py), where the full gradient lets it decay from 3.3e-4 to 1.3e-4 J
   (`logs/fg_cn_full50.log`, `logs/fe_cn.log`). The projection is therefore not used. The roughness drops with denser
   boundary sampling: with the isotropic pressure of finding 11, the rotating fluid kept 0.74, 0.86 and 0.88 of its
   angular momentum over 1 s at spacings of 0.75, 0.5 and 0.35 particle sizes (`logs/fg_spin_full{75,50,35}.log`). The
   spacing is 0.5 particle sizes; at it the fluid keeps 0.72 without the isotropic pressure
   (`logs/fe_spin_inviscid.log`).
3. Wall viscosity: the pair form between fluid and a single layer of boundary particles is mostly normal to the wall,
   so its tangential friction is several times too weak and the fluid slips. It is replaced by ghost-fluid columns
   behind each boundary sample with linear extrapolation of the tangential velocity (second-order no-slip), projected on
   the tangent plane, with the reaction at the fluid particle.
4. Anisotropy of the viscous operator. The pair form reproduces the Laplacian only if the fourth moment tensor of its
   weights is isotropic, which two particle spacings of support do not give: on a cubic lattice it reproduces 0.70 of
   the Laplacian of a shear flow and 1.19 of that of an extension (offline lattice computation). A per-particle
   correction of the pair weights by a function of the direction in the 15 even spherical harmonics up to degree 4,
   solved from the moment equations with the wall ghosts included, brings both to 0.99 on the lattice and 0.97 +- 0.06
   on a shaken one. The pair weight is the mean of the corrections of both particles, clamped at zero, so the operator
   stays symmetric, positive semi-definite and central.
5. Linear-shear consistency at walls (tried, removed). The ghost columns are a continuum and the fluid neighbors of a
   near-wall particle sit on discrete layers, so a linear shear along a wall leaves a spurious force on the first layer
   (a fifth of the viscous stress gradient of a unit shear on a cubic lattice, offline lattice computation). Scaling
   each slot tensor to make the operator exact for a linear shear lowered the Couette torque, but that torque was
   inflated by the pressure friction of finding 11; with the isotropic pressure the scaling leaves the torque 13 % low
   and the spin-down 27 % slow, against 3-7 % and 17 % without it (receipts `couette__mxA_couette`,
   `couette__mxB_couette`, `spindown__mx{A,B}_spin`), so it is not used.
6. The viscosity-pressure agreement of a substep is solved by a block iteration whose density solves use a forcing term
   of 0.1 and whose pressure is extrapolated by Chebyshev semi-iteration on the spectrum bound [1 / L, 1]. At nu = 1e-2
   the passes contract by about 0.87 each (receipt `spindown__devhist_nu1e-2`, `worst_coupling_history`), slower than
   the bound predicts, since a pass runs one density iteration on the pressure of the previous one.
7. Rigid coupling. The fluid stage runs inside the rigid substep, before the constraint solve, with the mobility of the
   coupled links in the pressure and viscosity solves. The viscous impulse a link takes is the reaction of the impulse
   the fluid took at the link velocities it saw, so the exchange balances momentum at every pass, converged or not, and
   the implicit link velocities enter the next pass as a correction (stable for light links).
8. Rigid integrator defect (fixed in 546010aeb): the single precision midpoint integration of free bodies stopped its
   Newton iteration at the initial guess when the residual was below 1e-6 (1 + 2 |I w| / h) N m, which drops every
   torque below about 1e-6 N m on a small body. The fluid torques on a 3 cm box are of that order, so the box kept its
   spin while the fluid lost angular momentum. With the tolerance floor scaled by the torque, the closed fluid-box
   system keeps linear and angular momentum to 1.4e-5 and 7.6e-6 per second at nu = 1e-3 (receipt
   `momentum__mid2_nu1e-3`, V6 limit 1e-4), and 7e-7 and 7.9e-6 inviscid (`momentum__mid2_nu0`). Regression test
   `tests/rigid/test_dynamics.py::test_small_free_body_torque`.
9. Leakage through a single boundary layer. A free-surface particle short of fluid neighbors reaches the density target
   only very close to the boundary particles and can cross their layer, after which the boundary pressure keeps it
   inside the geom. A wall clearance guard after the solves keeps every particle a quarter spacing in front of the
   surface it faces, as a unilateral velocity constraint with a slow position recovery and the reaction on the link.
   The faced surface comes from the boundary particles whose tangent region contains the particle, the nearest along
   the normal, so that the far face of a thin wall does not count. With it, a floating box keeps no fluid inside
   (`logs/mid2_gap.log`, box frame count).
10. A cube at half the density of water is unstable face down (a cube floats face down below 0.21 and tilts onto an edge
    between 0.21 and 0.79): once torques reach the box (finding 8), it tilts by about 45 degrees. The V8 case at density
    0.5 is a plank of aspect 2.
11. Pressure friction. The Couette torque grew with the hydrostatic pressure: at dx 3.5 mm the torque on the deep layer
    was +4 %, +21 % and +48 % above the reference at gravity 2, 9.81 and 20 m/s^2 (receipts
    `couette__mid3_g2_dx0.0035`, `couette__g3_pbs_dx0.0035`, `couette__mid3_g20_dx0.0035`), and its excess was the same
    2-3e-4 N m at nu = 3e-4, 1e-3 and 3e-3 (`couette__c1_nu*`): an additive, pressure-proportional shear stress. The
    stress budget at mid gap (`logs/stress_g9.81.log`, dev/couette_stress.py) shows the cause: the symmetric pressure
    pairs carried a shear stress of -2.3 Pa at a mean pressure of 330 Pa, twice the viscous stress, because a sheared
    neighborhood is anisotropic and the pair form turns the pressure into a stress along the anisotropy. The wall
    geometry was not involved (smooth-normal projection `couette__face2_g*`, 128-facet rotor
    `couette__mid2_rot128_dx0.0035`, no wall guard `couette__tog4_noguard_*`: unchanged). Particle shifting reduced it
    (`couette__shift*`) but broke angular momentum by 4e-3 to 1.4e-2 per second (`momentum__shift*_mom`) and lowered the
    hydrostatic pressure, so it is not used. A symmetric directional correction of the pressure pair weights in the
    degree 2 harmonics makes the pressure virial of every particle isotropic (central, momentum conserving): with it
    the torque no longer depends on gravity (`couette__iso1_g20_dx0.0035`: within 4.5 % deep at g = 20), the momentum
    balance is unchanged (`momentum__iso1_nu1e-3`), and with the full boundary gradient of finding 2 the Couette torque
    is within 2-6 % at both depths (`couette__fgE_couette`) against 10-27 % and 26-48 % without it
    (`couette__fe_couette_dx0.0035`). But the corrected force is not the gradient of the density estimate, and water at
    rest heats: the closed column of finding 2 went from 2.3e-3 to 3.7e-3 J of kinetic energy over 0.9 s
    (`logs/fg_cn_full50_iso.log`). The correction is therefore the option `SPHOptions.isotropic_pressure`, off by
    default. In FluidBench S1 (water) the spin-down time is 2.0 s without it and 2.8 s with it (OpenFOAM 7.9 s; records
    `gs-dfsph__S1__f2@sphgoal-fe`, `@sphgoal-fgE`).
12. Particle noise near the edges of a box immersed in fluid at rest (fixed or articulated paddle): with the projected
    boundary gradient, the viscous dissipation readout of a fluid at rest was 20 times that of the same cup without the
    box, concentrated at its edges and corners, and the isotropic pressure tripled it (0.022 J and 0.059 J over 1.5 s,
    receipts `articulation__mx{D,C}_stir0`). With the full boundary gradient of finding 2 it is 2.0e-4 J
    (`articulation__fe_stir0`).
13. Velocity noise under gravity. With the projected boundary gradient, a closed column at rest at nu = 1e-3
    dissipated 1.6e-4 W per 0.3 kg whatever the warm start of the density solve or the substep (receipts
    `decay__ws*_rest_g9.81`): the projection of finding 2 kept feeding the noise. With the full gradient the noise of
    the column decays (rms speed 2.9 to 0.6 mm/s over 0.9 s, `logs/fi_cn_nu1e-3.log`). In FluidBench S1 the initial
    lattice releases 2.5e-3 J as it compacts in the first control step (`timeseries.csv` of the S1 runs).

14. Fluid trapped inside a thin wall. The cup of FluidBench S1-S3 has a 4 mm wall, and the FluidBench adapter sets
    the boundary offset to 0.7 particle sizes, 1.885 mm at R2 and 2.64 mm at R1: the boundary particles of the inner
    and outer faces of the wall lie 0.23 mm apart at R2 and cross at R1. A fluid particle that gets between them stays
    trapped at up to 14 % above the rest density while its pressure grows, until the density solve expels it within
    one substep at a speed that grows as the substep shrinks (substep trace `logs/eject_substep.log`,
    dev/eject_substep.py: particle at r = 41.5 mm inside the wall, density 1.11 to 1.14, pressure rising tenfold,
    expelled at 8 m/s). At the FluidBench time step (dt 0.31 ms) this flags S1, S2 and S3 at R2 as unstable (particle
    speeds above ten times the characteristic speed, 0.1 % of the liquid out of the domain), and R1 diverges in all
    three scenes, at the starting revision of the goal as well (`@sphgoal-final`, `@sphgoal-r1-26832080b`); at dt x4
    and x8 the expulsions stay below 2 m/s (`timeseries.csv` of `@sphgoal-nl1-it100-dt*`). With the offset at 1.35 mm
    (a third of the wall, 0.5 particle sizes at R2, the Genesis default) S1 at R2 runs without a flag at the FluidBench
    time step (`@sphgoal-off0.00135-R2`), while 1.0 mm let 0.11 % of the liquid out and R1 at 1.35 mm 0.30 %
    (`@sphgoal-off0.001-R2`, `@sphgoal-off0.00135-R1`). The goal runs of FluidBench use 1.35 mm, recorded in their
    parameters. Guarding a second faced surface at concave edges (wall clearance guard) made it worse (speeds up to
    59 m/s, `@sphgoal-g2p-dt*`) and is not used.
    The boundary particles of each face were moved by the offset along the face normal while the face kept its extent,
    so at a reentrant edge of a geom (the inner corner between the floor and the wall of a cup) the two moved faces
    left an unsampled slot as wide as the offset. At R3 fluid entered the solid of the cup through it (substep trace
    `logs/eject_substep_R3.log`: a particle at r = 42 mm and z = -1.4 mm, beyond both boundary layers, by 0.03 s) and
    the run diverged at 0.24 s whatever the iteration cap and with or without the accelerated solve
    (`@sphgoal-final` R3, `@sphgoal-r3-it1000`, `@sphgoal-r3-it1000-nocheb`). The boundary sampling now extends both
    moved faces of every reentrant edge to the line where their planes meet (`sample_reentrant_edges`, test
    `test_boundary_offset_closes_reentrant_edges`): S1 at R3 runs (2 s, `@sphgoal-re1-R3`), and S1 at R2 runs without
    a flag at the FluidBench time step with the default offset of the adapter as well (`@sphgoal-re1-R2-off0.7ps`);
    R1 at that time step still lets 0.27 % of the liquid out (`@sphgoal-re1-R1-off1.35`), at dt x4 it runs.
15. Chebyshev acceleration of the density solve and the particle noise. Stopped at the FluidBench tolerance (average
    density error 0.05 %), the accelerated solve leaves a closed column of water at rest with 2.9e-4 J of kinetic
    energy after 1 s, where the plain solve leaves 1.3e-4 J; with the tolerance ten times tighter both leave 1.4e-4 J
    (`logs/ch1_cn.log`, `logs/fe_cn.log`, `logs/kp_cheb_tight_cn.log`, `logs/kp_plain_tight_cn.log`): the plain
    Jacobi iterate is already converged in this respect at the tolerance, the accelerated one is not. Accelerating the
    density solves of the later coupling passes as well broke the viscosity-pressure coupling of a viscous scene (33
    passes per substep and 3029 unconverged substeps against 9.5 and none, `articulation__ch1_stir0`); with plain
    later passes it converges as before (`articulation__kp_stir0`). The acceleration therefore applies to the first
    density solve of a substep, which stops at a tenth of the tolerance.

16. Wall slip of fluid that is not pressed against the wall. The ghost columns of the wall viscosity start at the
    surface, and the fluid nearest to it stands for the fluid down to half a particle spacing below it; a wider gap
    between that edge and the surface is neither fluid nor ghost, and the wall transmits only part of the shear across
    it. Offline, for a linear shear over a flat wall (dev/wall_flux.py, FluidBench
    `adapters/genesis/sph_validation/dev`), the wall stress is 0.93, 0.51 and 0.21 of mu gamma with the first layer at
    0.5, 0.75 and 1.0 particle sizes from the surface. Under hydrostatic pressure the first layer rests at 0.42 particle
    sizes (`logs/wall_gap.log`), in the falling column of the pipe case at 0.80 (`logs/pipe_gap.log`), where the
    centerline ran 58-75 % too fast. Each boundary particle now records the distance of the nearest fluid particle in
    front of it, and its column fills the gap beyond half a particle spacing with the same extrapolated velocity (0.93
    of mu gamma at any gap offline): the pipe centerline error drops from 0.58 to 0.17 at dx 2.5 mm and from 0.75 to
    0.18 at 1.875 mm, the mode decay from -23 % to -17 %, the spin-down from -11.9 % to -9.6 % (receipts
    `pipe__band_pipe_dx*`, `decay__band_decay_*`, `spindown__band_spin_*`); water at rest, momentum, contact, state,
    plank and paddle unchanged (`*__band_*`).

17. Pressure friction acts as a static friction on a settling sphere (V9). At dx 3.5 mm and nu 1e-2 the sphere settles
    at 0.0037 m/s against 0.0144 (-74 %), at dx 2.5 mm at 0.0013 (-91 %), and it decelerates instead of reaching a
    terminal velocity; a coupling cap of 300 passes, which converges every substep, leaves it at -68 %
    (`settling__v9c_*`). The shear stress that the pressure pairs carry in a sheared arrangement (finding 11) grows with
    the pressure and opposes the sliding of the fluid along the sphere, like a friction. With the isotropic pressure
    the sphere reaches a steady terminal velocity, -32 % at dx 3.5 mm and -29 % at dx 2.5 mm (`settling__v9d_*`,
    `settling__ib_settle_*`), and the Couette torque errors drop from 13-48 % to 2-16 % (`couette__band2_*` against
    `couette__ib2_*`, both with the free-surface height taken as a high quantile of the particle heights, which a
    stranded particle no longer moves). The isotropic pressure still heats water at rest at low viscosity (finding 11).

18. Volume maps (smooth wall, `SPHOptions(boundary_model="volume_map")`). The boundary density of a link is the
    integral of the kernel over its solid (Koschier and Bender 2017), computed once at build time on a grid in link
    frame by FFT convolution of the occupancy given by the signed distance of its geoms (`compute_volume_map`), and
    its pressure force is the exact gradient of that density, so the wall is smooth and does no work on fluid at rest.
    Water at rest in a closed column cools as with the boundary particles (kinetic energy 4.1e-4 to 1.3e-4 J over 1 s,
    `logs/vm5_cn.log`). Inviscid fluid in rigid rotation in a cup (dx 5 mm, `dev/spin_down.py`) keeps after 1 s a
    fraction of its angular momentum of 0.72 with the boundary particles (`logs/fe_spin_inviscid.log`), and with volume
    maps: 0.26 with the volume interpolated trilinearly on cells of 1/4 particle size (`logs/vm3_spin0.log`). A
    trilinear interpolation ripples from cell to cell along a surface not aligned with the grid, a roughness of second
    order in the cell size. The map therefore stores the distance at which a half-space has the volume
    (`volume_map_to_distance`), linear for a planar surface: interpolated trilinearly, the rotating fluid keeps 0.56,
    0.87 and 0.973 on cells of 1/2, 1/4 and 1/8 particle size (`logs/vm4_spin0*.log`; offline, along a tilted plane,
    the tangential gradient is 0.01-0.3 % of the normal one against 4-14 % for the trilinear volume,
    `logs/vm4_ripple.log`), so the remaining loss presumably comes from the edge of the floor, where the distance
    curves within the kernel support (not isolated).
    Interpolated by a cubic B-spline, which reproduces linear fields exactly and has a smooth error elsewhere, it keeps
    0.988 and 0.997 on cells of 1/2 and 1/4 particle size (`logs/vm5_spin0_c*.log`), the default being 1/4. The
    boundary particles remain for the wall viscosity and the clearance guard, sampled on the surface itself; a thin
    wall needs no offset, the fluid resting half a particle spacing from the surface.

    Validation with the volume maps (receipts `*__vm_*`, `*__vm6_*`, FluidBench): pipe centerline error 0.020 and
    0.041 at dx 2.5 and 1.875 mm (V3 passes; 0.17 and 0.18 with the boundary particles), spin-down -3.6 % (-9.6 %),
    sphere on the floor buoyancy -16 % (-22 %), momentum, state and plank load unchanged; but the Couette torque is
    26-65 % high (13-48 %), the sliding-floor shear +59 % at nu 1e-2 and +129 % at 1e-3 (+32 % and +88 %) and the plank
    draft +14 to +23 % (-14 %). Against the smooth wall the first fluid layer rests on the clearance guard, a quarter
    particle size from the surface, 13 % sparser than against the boundary particles (0.42 particle size), with the
    density at 1 in both (`logs/vm_wall_gap2.log`, `logs/pa_wall_gap2.log`): the wall pressure enters the force of a
    particle by its own pressure alone (BWJ23), while the fluid above pushes with the pressures of both particles of a
    pair, and the smooth wall, whose gradient at the surface is 0.69 per particle size against 0.86-1.0 for the
    particles, supports the column only once the layer is that close. The volume maps remain an option.

19. Wall column against fluid pressed closer than half a spacing. The ghost columns of the wall viscosity started at
    the surface, while the fluid particle nearest to it stands for the fluid down to half a spacing below it: with the
    first layer at a quarter spacing both stand for the fluid near the surface, and a linear shear transmits 1.34 of
    mu gamma (offline, `dev/wall_flux2.py`; 1.11 at 0.4 spacing, the rest distance against the boundary particles). The
    column now starts at that edge, below the surface when the fluid is closer (0.93 at any gap); with the boundary
    particles the Couette torque errors drop from 13-48 % to 10-43 %, the other cases unchanged (receipts `*__sb_*`).

20. Particle volume of the FluidBench liquid. The adapter samples the liquid on a lattice of spacing dx and sets the
    particle size to dx / 0.8^(1/3), sized for a particle volume of 0.8 particle size^3, with a volume factor of 0.75:
    each particle carries 0.9375 dx^3 of water, the lattice has a density of 0.941 (`logs/lattice_sum.log`), and the
    liquid compacts by 6 % as it settles, the volume error of S1 (`|e_V|` 0.055-0.06 "from the first frame", the center
    of mass of the liquid falling from 29.9 to 28.2 mm within 0.5 s). With the factor 0.8 the lattice has a density of
    1.004 and the column erupts at the start (2.6 m/s, liquid over the rim, `@sphgoal-f08dev`, `@sphgoal-vmdev2`); a
    particle size of dx with a factor of 1 puts the lattice at a density of 1.000 and conserves the declared mass of
    water. A Cartesian lattice cut by the curved wall of the cup still starts some particles nearly touching it, and
    the liquid at the wall below the free surface is thrown up at 0.6-0.8 m/s (`logs/fast_src.log`); FluidBench S1
    leaves the liquid at rest for 0.5 s before stirring, so a quasi-static settling of 0.5 s before t = 0 (velocities
    zeroed after every step, the adapter option `sph_relax_s`, recorded as a deviation of the run) removes the leakage.
    S1 f2 R2 then has a volume error of 0.51 % (`@sphgoal-rx-pi`, dynamic settling) or 0.9 % (`@sphgoal-rx2-p`).

21. Spin-down of S1. The adapter now reads the fluid load on the rod with `get_links_fluid_wrench`. While the fluid
    spins down (6-8 s) the rod at rest takes 36 % of the loss of angular momentum with the boundary particles and the
    isotropic pressure, 47-52 % with the volume maps; the rest goes to the cup wall, internal forces conserving the
    angular momentum (`logs/s1_budget_rx*.log`). Per unit angular momentum each of the two takes 0.14-0.25 per second,
    about the whole decay rate of the reference (0.17 per second, tau_sd = 5.8 s); a rough estimate of the drag of the
    rod (flow speed from the mean angular velocity at the radius of the rod) gives a drag coefficient near 8 at a
    Reynolds number of about 400, where about 1 is expected. The spin-down time stays 51-71 % short at R2 in every
    configuration (`@sphgoal-rx2-*`); no particle method of FluidBench passes it (gs-dfsph, gs-mpm, gs-pbd, ch-sph and
    fl-mpm all fail S1, only of-vof passes, `logs/s1_rows.log`).

22. Isotropic density. With the isotropic pressure, the density that the pressure solves constrain can take the
    corrected pair weights as well (`isotropic_density`), consistent with their velocity constraint and force. The
    Couette torque error is then 0.1-9.3 % and the profile 1.5-2.7 % (V2 passes, `couette__id_*`), the settling sphere
    -23 % at dx 3.5 mm (`settling__id_*`), and the new V1 case (finding 23) 3-10.3 % on the profile at dx 2.5 mm (its
    floor torque was then read against a reference withdrawn in finding 23). It does not stop the heating of water at
    rest by the isotropic pressure (closed column at nu 1e-6: kinetic energy 2.9e-3 J after 1 s, 3.6e-3 J without it,
    1.3e-4 J and decaying without the isotropic pressure, `logs/id_cn_*.log`). Relaxing the correction in time
    (`isotropic_pressure_relaxation` 0.2 and 0.05 per substep) lowers the heating to 1.1e-3 and 5.0e-4 J but raises
    the Couette errors to 2-14 % and 5-20 % (`logs/ir*_cn_*.log`, `couette__ir*`): the correction, recomputed from
    the arrangement, does work on the fluid.

23. V1 case. The sliding-floor trough (`trough.py`) compares with a plane zero-flux shear flow that assumes the return
    flow at once and no side walls; at nu 1e-2 the side wall layers span the trough, and the return flow waits for the
    free surface to tilt (L / sqrt(g H) = 0.36 s), so its profile errors stay at 0.45-0.94 in every configuration.
    `stokes1.py` realizes the first problem of Stokes of the design: a disk floor spun at once under a deep layer in a
    fixed tube, without end walls, with the profile u_theta / (Omega r) between 0.25 R and 0.6 R compared with erfc.
    Its first version compared the torque on the whole floor with an axisymmetric finite-difference solution that
    includes the corner with the fixed tube. The torque of that corner grows without bound as the grid refines (a
    logarithmic singularity of the velocity jump; reference at 40 to 240 cells, `logs/stokes_ref_conv.log`). That
    reference and the torque numbers against it (`stokes1__dev2_*`, `stokes1__id2_*`) are withdrawn. The floor is now
    an inner disk of radius 0.6 R and an outer ring under the tube wall, both spun, and the torque on the inner disk
    alone is compared with 2 pi mu Omega R_i^4 / (4 sqrt(pi nu t)) while the layer is resolved (2 sqrt(nu t) > 3 dx)
    and thin (< 0.5 (R - R_i)). At cab47d96, nu 1e-4 (`stokes1__v4*_`):
    - default: profile 8-34 % off, torque 1.36-2.64 times the reference;
    - isotropic pressure and density: profile 1.3-8.9 % at dx 2.5 mm and 3.7-6.9 % at dx 1.75 mm (7-18 % at
      dx 3.5 mm), torque 0.51-0.97;
    - volume maps: profile 6-16 %, torque 0.29-0.76.

    No configuration meets the torque bar of 15 %.

24. Constraint consistency loop with held joints (V10 hold). After the rigid constraint solve the fluid stage is
    repeated with the new link velocities until they agree (`max_constraint_iterations`, `max_constraint_error`).
    The joints of V10 hold are held by stiff position controllers, whose damping the rigid solver makes implicit
    (it factors M + h D). Against such joints the repetitions alternate, and the loop reaches its cap of 4 in 2440 of
    3000 substeps. The fluid load of the last repetition is then 19 % and 31 % away from the controller force of the
    two joints (`articulation__v4_hold`). Disabling the loop (`max_constraint_iterations=0`) makes them agree within
    0.2 % (`articulation__dbg_hold_noconstr`). The variants tried, on the receipts of FluidBench
    `report/sph_goal.md`:
    - a uniform under-relaxation of the repetitions of 0.5: 3.1 % / 11.6 %, every substep at the cap;
    - 0.25: 0.04 % / 0.09 %, 781 substeps at the cap. The resting sphere of V7, which converges within the cap
      without it (no substep at the cap), then reaches the cap in 3443 of 4000 substeps, with its weight balance at
      -0.4 %;
    - the first repetition full and the rest at 0.25 with a cap of 4 or 8: 4.4 % / 11.5 % and 0.7 % / 1.7 %, at the
      cap in 2999 and 2936 substeps;
    - the relaxation shrunk only when the residual grows: 7.4 % / 18.9 %, and the V7 balance moves to 2.1 %.

    A substep at the cap counts against a run, so the uniform relaxation is an option (`constraint_relaxation`,
    96c3619c), default 1, documented with this trade-off. In every variant the fluid load on the held plate stays
    17-52 % from the Archimedes force with boundary particles. With the volume maps it is 2.4-6.9 %, the same deficit
    as the buoyancy of the sphere resting on the floor (24.5 % and 7.1 %). The stirring paddle, driven by velocity,
    reads its torque within 0.4 % in all variants.

    Superseded by finding 26: the repetitions read a stale acceleration, and with it fixed the held joints agree
    without any relaxation; the option was removed (a071c5e2).

25. Momentum drift at the tolerance of the viscosity solve (V6). With the isotropic options the closed fluid-box
    system drifted 1.2e-4 / 1.5e-4 per second (`momentum__v4i_mom`), the fluid ball alone with rotation 1.2e-4 per
    second in angular momentum even by default (`momentum__m6_def_nobox_mom`). The isotropic pressure alone and the
    isotropic options without viscosity keep it (`momentum__m5_isop_mom`, `momentum__m5_iso_nu0_mom`), and a tighter
    viscosity tolerance (1e-5) cuts it to 4.7e-5 (`momentum__m6_iso_tightvisc_mom`): the viscous pair forces are
    central, but a conjugate gradient solution stopped at its tolerance carries a net residual force and torque. A
    rigid velocity correction after every viscosity solve, solving the 6x6 system of the mass and wall tensors for
    that net residual (rigid motions are null modes of the pair operator), cancels it (22f43b2f): 3.5e-6 / 3.1e-6
    default, 1.9e-6 / 3.3e-6 isotropic, 3.9e-6 / 4.3e-6 volume maps (`momentum__mproj_*`); the unit test
    `test_implicit_viscosity_conserves_momentum` now checks 2e-5 at a viscosity tolerance of 5 %.

26. Stale acceleration in the constraint repetitions (V10 hold). The repetition of the fluid stage after the rigid
    constraint solve compared the change the stage held with h S (acc - acc_smooth), but the rigid solver copies the
    solved acceleration from its constraint state to the dofs only after the repetitions: acc was the final
    acceleration of the previous substep (trace `logs/hold_trace.log`, dev/hold_loop_trace.py: acc unchanged over
    the repetitions of a substep while acc_smooth moves). Steady contacts barely change it, so the resting sphere
    converged, while the held joints never could. Reading the constraint state (a071c5e2) makes the held plate agree
    with its controllers within 0.01 % / 0.08 % without any repetition (`articulation__qacc_hold`), the resting
    sphere balance within 0.3 % with a third of the repetitions (`contact__qacc_contact`).

27. Convergence of the viscosity-pressure coupling. The coupling passes stopped at their cap of 30 in 377 of 420
    substeps of V4 (zero gravity), half the substeps of V5 and many of V9 (`decay__v4_*`, `spindown__v4_*`,
    `settling__v4_*`), so these receipts were not valid samples. Without gravity the disagreement oscillated between
    9 and 17 times its tolerance while every density solve was converged: the pressure clamp switches particles
    within the tolerance of the density solves, which resolve the velocity only to eta dx / h (eta their tolerance),
    far above 0.1 % of the weak viscous impulse of a slow mode. Under a strong viscosity the Chebyshev acceleration
    of the passes diverged once the response of the sphere broke its spectrum bound. The coupling now also counts as
    converged once the pressure velocity changes by less than eta dx / h, and the acceleration restarts when a pass
    raises the disagreement (daa41b6e): no capped substep remains in V4 at nu 1e-2, 1e-3, 1e-4, V5 and V9
    (`*__coup2_*`), with the same accuracy (V4 16.5 / 17.4 / 25.3 %, V5 2.3 %, V9 -68 %).

28. Stirring energy balance (V10). The driver took the kinetic energy of the paddle from the rigid mass matrix, which
    holds h kv of the velocity controller after the implicit damping factorization (I + h kv for I), and read the
    controller torque once per scene step, at the last of its substeps, while the torque of the spin-up changes
    within the step. With the link inertia and one substep per step (same substep length) the balance closes within
    1.1-1.4 % of the work (`articulation__ke2_s1sub_stir`, `articulation__qacc_stir`); the fluid balance alone, from
    the torque readout, was within 1.4 % throughout.

29. Hydrostatics (V7, `hydrostatic.py`). The fluid load on a fixed cup equals the weight of the fluid within 1e-4
    in every configuration. The pressure of the particles (`get_particles_pressure`, 8cd31d27: rho_0 C_i kappa_i,
    C_i the gradient scale of the arrangement, 1 in the bulk) follows rho g (H - z) in the bulk with a gradient of
    1.009-1.02 rho g, within 4.2 % at dx 2.5 mm by default and 3.2-4.6 % with the volume maps (5.45 % at dx 3.5 mm by
    default) (`hydrostatic__h4*`). Within a support radius of a wall the multiplier reaches 1.6 times the pressure:
    the boundary force takes the particle's own multiplier alone (BWJ23), so these layers carry the wall load with a
    larger multiplier, which also steepened the profile averaged over whole layers (1.05-1.16 rho g).

30. Boundary sampling of large geoms. The greedy furthest point sampling of the boundary particles ran on the CPU over
    16 candidates per sample, quadratic in the samples: the cup of the plank case at dx 1.75 mm (about 2e5 samples)
    did not leave the build in 4 h. It runs on the GPU above 1e10 operations (14dc5bb6), with the same picks
    (`dev/fps_check.py`: 4000 of 4000, 10 times faster), and the plank at dx 1.75 mm builds and runs in 4 min: draft
    error 14.3, 8.8 and 4.9 % at dx 3.5, 2.5 and 1.75 mm (`floating__q2_plank500_dx0.00175`).

31. Pressure friction, further attempts (finding 11). The friction does not depend on the solver tolerances: the
    Couette torque errors stay at 12-45 % with a density tolerance from 5e-3 to 1e-4, a divergence tolerance of 1e-2
    and no warm start (`couette__tol_*`). A density offset that holds the jumps of the isotropic density when its
    weights are recomputed does not stop the heating of the isotropic pressure (closed column 4e-3 J after 1 s at a
    relaxation time of 0.05 s) and lets the column collapse at 0.2 s and 1 s (`logs/ido1_cn_*.log`): the heating
    comes from the linearization of the corrected density, whose weights depend on the pair directions, which the
    central Jacobian of the solves omits. A Fickian particle shift with a rigid rotation restoring the angular
    momentum heats water at rest (3.4e-3 and 4.7e-3 J at coefficients 0.1 and 0.2) and leaves the deep Couette layer
    11-50 % off (`couette__sh3_*`); scaled by the rate of relative motion of the neighbors it brings the Couette
    torques within 0.5-13.9 % (coefficient 300, `couette__sh4_300_*`), the V1 floor torque within 15 % and V4 to
    13.6 %, but the particle noise of water at rest (3-8 cm/s rms) drives the same shift and heats it (9e-4 J,
    `logs/sh5_*_cn.log`), and angular momentum drifts 1.2e-4 per second (`momentum__sh5_*`): not adopted. A kernel
    support of 3 particle sizes (an experimental option) gave NaN constraint forces at the first step: the boundary
    volumes, wall columns and slot sizes assume a support of 2 particle sizes.

32. Density drift corrected by the positions (`SPHOptions(density_correction="position")`, 3f654034). In DFSPH the
    density solve answers the density error with a velocity, and the error it leaves, the second order of the
    advection and the jumps of the arrangement turn into particle velocity: water at rest keeps 3-5 cm/s rms for
    seconds (finding 13; the same at density and divergence tolerances of 1e-5, `logs/n5_*.log`, so the solver
    tolerance does not set it). With the option the density solve of the coupled passes only removes the compression
    the velocity would cause (from the rest density, or from the density of a free-surface particle below it), and
    carries the pressure, and a density solve of the same form after the coupled passes moves the particles by h^2
    times its acceleration, the rigid surfaces held in place, with a rigid rotation restoring the angular momentum
    the displacements change. Water at rest: kinetic energy 1.2e-5 J after 1 s and decaying, against 1.4e-4 J
    (`logs/drift1_cn.log`, `logs/n5_def.log`); V4 at nu 1e-3: 10.1 % against 17.4 %; V11 R1 from rest: 2 particles
    through the wall and 0.7 m/s against 4607 and 6.0 m/s; V6 8e-6 per s; V7 bulk 6.9 % against 4.2 % at dx 2.5 mm;
    the Couette torque keeps its pressure friction (11-39 %, `*__drift1_*`). Correcting both signs of the error
    inside the fluid, which a displacement may since it exerts no force, heats water at rest instead (6.8e-4 J,
    `logs/drift2p_cn.log`): only compression is corrected.

33. The isotropic pressure with a velocity-level projection. With the drift taken by the positions, the density solve
    of the coupled passes is a projection of the velocity, which with the weights of the isotropic pressure in both
    its constraint and its force is orthogonal in the norm of the masses and cannot add kinetic energy; it removes
    the pressure friction (Couette 0.3-8 % at both depths, profile 3.9 %, `couette__proj2i_*`, `couette__drift3_*`).
    But a velocity without weighted divergence still changes the plain density, so the drift correction keeps
    working against it: correcting compression alone, the column expands by 7 % and gravity turns the lifted
    potential energy into motion (1.3e-3 J after 1 s, `logs/proj2i_cn.log`); with the weighted density and weighted
    drift (`logs/drift3_cn.log`, 1.5e-3 J, 1932 of 2000 drift solves at their cap) or both signs corrected
    (`logs/drift2pi_cn.log`, 5.9e-3 J) it heats as well. A particle shift on top of the drift correction lifts the
    particles below the free surface, whose concentration gradient points down, and heats too (2.6e-3 J,
    `logs/dsh1_*_cn.log`). No variant of the isotropic correction or of the shift is adopted.

34. Coupling floor and the drift correction in practice (97aebd75). The floor of finding 27 at the full eta dx / h let
    the stirred cup of V10 stop its coupling passes early enough to misread its viscous dissipation: balance -7.9 %
    (`articulation__v5_stir_s1`) against -1.4 % before; at a tenth of it the balance is -1.0 to -1.8 % and V4 stays
    converged with the same rate (`*__fl0.1_*`, `*__fl0.3_*`, `*__f2_*`). The drift correction took the density solver
    cap (10 in the validation drivers) and stopped there in every substep of the floating cases; with its own cap of
    50 it converges (plank draft 4.0 %, resting sphere balance 0.05 %, `*__f2p_*`; a warm start with Chebyshev
    acceleration instead raised the noise of water at rest to 1e-3 J, `logs/d4_cn.log`). In FluidBench
    (`sphgoal-v5p`, 47499e5c with the drift correction) it removes the leakage of S1 R2 and makes S2 R2 usable, with or
    without the settling before t = 0, but the spin-down of S1 gets shorter (-77 % at R2, -86 % at R3, against -71 % and
    -75 %), the mixing index of S3 moves away and the solve costs a third of the speed: it stays an option.

35. Wall slip and the boundary offset (V3, V4). The pipe profile of V3 with boundary particles is the analytic one
    shifted by a uniform slip of 11 mm/s at dx 2.5 mm, a slip length of 0.6 particle sizes (3 mm/s, 0.16 with the
    volume maps; `pipe__v5_*`, `pipe__v5v_*`). The boundary offset sets it: the centerline error is 0.7 %, 17 % and
    33 % at offsets of 0.25, 0.5 and 0.75 particle sizes, and the fluid leaks at 0 (`pipe__p1_off*`). A boundary
    particle that finds no fluid particle within half a particle size of its normal starts its ghost column at the
    surface and leaves the gap unfilled (about a fifth of them on a lattice); searching 0.75 particle sizes brings
    the pipe to 13 % (`pipe__gap1_*`). The default offset of 0.25 particle sizes with that search (`*__v6o_*`) passes
    V3 (2.7 / 3.4 %), brings V2 to 1-23 % and V4 to 11-17 %, but the standoff of the fluid that sets the wall slip also
    sets the buoyancy: the plank draft goes to 29-48 %, the light box to 62-98 %, the held plate to 25-37 % from
    Archimedes, the stirring balance to +40 % and the thin wall leaks more; not adopted. V4 converges with resolution
    at the default offset (38.5, 17.4, 13.0 % at dx 3.5, 2.5, 1.75 mm, `decay__w1_*`, `decay__w2_*`). The wall
    viscosity model must not depend on the standoff for the two to pass together: the open item.

36. Wall shear consistency (`SPHOptions(wall_shear_consistency=True)`, ce44ba1a). The wall tensor of each fluid particle
    near a flat wall is scaled so that the viscosity operator is exact for a linear shear along the wall, whatever the
    standoff of the particle (corners keep the plain tensor, the scale is bounded to [0.25, 4]). With boundary particles
    at the default offset (`*__v6w_*`) it passes V3 (centerline 1.0 / 1.6 % at dx 2.5 / 1.875 mm, 17-18 % before), keeps
    V5-V7 and V10 (stirring -2.1 %, held plate 4.3-4.9 % from Archimedes) and V8 at dx 1.75 mm (plank 5.0 %, light box
    9.3 %; 8.8 / 18.5 % at dx 2.5 mm), but leaves V4 at 15-27 % and V1, V2 and V9 failing (pressure friction, finding
    11); V12 on the RTX PRO 6000 failed its envelope bar (first replayed step 2.5e-4, two replays 2.4e-4,
    `state__v6w_state_rtx`). With the drift correction of the positions as well (`*__v6wp_*`) V4 passes at nu 1e-2, 1e-3
    and 1e-4 (modal rate 8.2, 5.9, 9.6 %), V11 R1 keeps every particle in the cup (largest speed 0.15-0.2 m/s), V8
    passes at dx 1.75 and 2.5 mm (plank 0.3 / 4.7 %, light box 1.5 / 6.1 %), but V7 (bulk pressure 6.6 %) and the held
    plate (5.4-6.0 % from Archimedes) fail. An option; the default is unchanged.

37. Volume maps with the wall shear consistency (`*__v6vw_*`, `*__v6vwp_*`, ce44ba1a). The wall shear scaling on the
    volume maps passes V3 (3.0 / 4.3 %), V5, V6, V7 bulk and V12 on both GPUs, but the floating bodies sit too high
    (plank 13-22 %, light box 29-52 %), V9 settles 60-70 % too slowly, the torque of V2 is off by 20-60 % and V4 by
    14-28 % at nu 1e-2 to 1e-4. Not adopted.

38. Thin-wall start (V11). The FluidBench lattice of the liquid fills the cup to its inner radius: the outer column of
    S1 at R1 sits 0.09 mm from the inner wall, 1.4 mm from the boundary particles behind it (`logs/twprobe_R1.log`), so
    that the first control step from the lattice throws the liquid up at 6 m/s (rms 2 m/s) and over the 4 mm wall. With
    the drift correction of the positions the overlap is taken by the positions: no particle crosses at R1
    (`*__v6wp_thin_R1_*`), and the largest speed is 0.15-0.2 m/s; a probe with ce44ba1a (`logs/twprobe_R1_p.log`, 0.63
    m/s at 64 ms) finds the fastest particles after 0.1 s in the bottom layer of the liquid, which settles 0.8 mm toward
    the floor, and an rms speed of 0.03 m/s. At R0 (dx 5 mm wider than the wall) 19 particles cross. The bar of 0.05 m/s
    from rest is below the speed of the particle noise of water at rest at these resolutions (largest speed 0.03-0.08
    m/s in the closed column, whatever the solver tolerances, `logs/noise_nu1e-3*.log`).

39. Mode decay at nu 1e-6 (V4). The numerical dissipation of the particle noise exceeds the physical one: the modal
    amplitude decays about three times faster than the analytic rate (`decay__v6wm_decay_nu1e-6_*` 2.08,
    `decay__v6wpm_decay_nu1e-6_*` 2.00 relative error) at dx 2.5 mm, against 6-10 % at nu 1e-4 to 1e-2 with the wall
    shear consistency and the drift correction. The modal projection removes the noise from the amplitude, so the excess
    is a damping of the mode by the solver, below the resolution of these runs.

40. A pressure predictor against the pressure friction (finding 11; development snapshots, not committed, diffs in the
    data directory `worktrees/dev-pred*/final.diff`). The solves would find the increment of the pressure over a
    substep, while the total pressure of the last substep acts through a predictor with the gradient corrected by L_i,
    the inverse of the moment sum_j V_j grad W_ij (x_j - x_i)^T (Bonet and Lok 1999), whose pair force carries no shear
    stress whatever the arrangement. Water at rest in the closed column (nu 1e-6) keeps 1.3e-4 J of kinetic energy after
    1 s without it, and decaying (`logs/wd0_cn.log` with the wall shear consistency: 1.4e-4 J); every variant reaches
    0.8-4.6e-3 J and grows. (1) Predictor before the divergence solve, plain solves (`*__pred1_*`, `*__pred2_*`): V9
    settles within 9 %, V2 passes (torque 2-9 %, profile 5.6 %), V1 holds its profile within 2-7 % (torque 0.97 / 0.84
    of the reference), V6 1.2e-5 per s with the torque of the corrected pairs cancelled by a rigid rotation, but the
    column reaches 4.0e-3 J (the stages the audit covers lose 2.7e-2 J over the run, so the predictor gives it), the
    stirring balance is -62 %, the liquid of the floating plank leaves the cup and the held plate ends in NaN; with the
    full increment carried over and a bilateral error, 31289 density solves stop at their cap. (2) The solves with the
    same corrected gradient and its adjoint L_i grad W_ij in their velocity constraint, and the Jacobi diagonal from the
    corrected gradients (`*__pred4_*`, `*__pred4p_*` with the drift correction): the projection is then orthogonal and
    the predictor only a warm start, V9 within 2.6-5.2 %, V2 passes (torque up to 12 %, profile 3.5 %), but the column
    still reaches 2.4-4.6e-3 J: the density solve constrains the summation density, whose rate is the plain divergence,
    and the corrected constraint leaves an error it keeps answering (+2.9e-2 J in the density stage); the stirring
    balance is -62 % with the drift correction and the scenes with kinematic or held links end in NaN in their first
    step. (3) The predictor after the divergence solve, with gravity, and plain solves (`*__pred5_*`, `*__pred6_*` with
    the update relaxed by half): the total pressure, the sum of the increments, drifts without bound along modes the
    corrected force hardly feels (the cup load of V7 at 1850 and 930 times the weight of the fluid). Not adopted.

41. Particle shift with a free-surface projection (development snapshot, diff in `worktrees/dev-sh2/final.diff`). The
    rate-scaled Fickian shift of finding 33 with, in addition, only the part along the free surface kept for a particle
    with a free-surface neighbor (normal from the kernel sum of its fluid neighbors, Sun et al. 2017), on top of the
    wall shear consistency and the drift correction of the positions (`*__sh2a100_*`, `*__sh2a300_*`, A = 100 and 300).
    It reduces the pressure friction: V2 torque 4-12 % and profile 2.1 % at A = 100 (5-32 % and 15 % without it,
    `couette__v6wp_*`), the plank draft 0.8 %. But water at rest takes 1.5e-3 J within 0.2 s and keeps it
    (`logs/sh2a*_cn.log`, rms speed 0.11 m/s, against 1.4e-4 J and decaying without the shift), the stirring balance is
    -15 times the work, the settling sphere falls 16 times too fast, V1 does not improve (profile 5-17 %, torque
    1.07-1.39 of the reference), the held plate ends in NaN and 16-17 particles cross the thin wall of V11 R1. The shift
    moves every particle by up to a tenth of its size per substep at the rate of the particle noise, which then feeds
    itself. Not adopted.

42. Pressure readout (18efcc66). `get_particles_pressure` returned rho_0 C_i kappa_i, whose boundary samples count as
    pairs of two equal multipliers, while the boundary force takes the multiplier of the fluid particle alone: 1.65
    times the hydrostatic pressure in the layer on the floor of V7 and 5-9 % above it within a support radius of the
    side wall (`hydrostatic__v6w_*`). The virial of the pair forces, half of each pair to each particle and the boundary
    samples at half weight, reads 1.24 times in the floor layer and 2-5 % above it elsewhere; the bulk metric of V7
    (above the floor layers, away from the side wall) goes from 4.2 to 3.7 % at dx 2.5 mm with the default solver and
    with the wall shear consistency, from 6.6 to 5.6 % with the drift correction as well (`hydrostatic__rd1*`). The
    layer below the top one reads 8 % low; the bar on every layer below the top one is still not met.

43. Wendland C2 kernel (development snapshot, diff in `worktrees/dev-wd1/final.diff`). The kernel shape changes how the
    pair forces answer a distorted arrangement, so the Wendland C2 kernel with the same support radius was tried with
    the wall shear consistency (`*__wd1_*`; its sum on the particle lattice is 1.034, so the liquid starts compressed by
    3.4 %). The pressure friction does not go down: V2 torque 8-58 % against 9-38 % with the cubic spline
    (`couette__v6w_*`), V9 66 % against 66 %, V1 profile 11-24 %; the momentum drifts by 1.1e-3 per s (V6), the stirring
    balance is +62 %, the held plate ends in NaN and the liquid of the plank leaves the cup. Not adopted.

44. Wall shear consistency with the isotropic pressure and density (`*__wi_*`, and `*__wip_*` with the drift correction
    of the positions; Genesis 59ed5a81, isotropic_pressure_relaxation 0.2). It removes the pressure friction where it
    matters most: V9 settles within 0.3 % at dx 2.5 mm and 9.9 % at 3.5 mm (the run at dx 1.75 mm ends in NaN in the
    rigid solver), V2 passes (torque within 10 %, profile 2.7 %), the V1 profile is within 2-8 % at every resolution,
    and V8 passes at dx 2.5 mm (plank 6.0 %, light box 5.5 %). But the floor torque of V1 is 0.56-0.86 of the reference
    (finding 23), the pressure does work: water at rest keeps 1.2e-3 J (1.6e-3 J and growing with the drift correction)
    against 1.4e-4 J and decaying without it, and the stirring balance of V10 is -4.9 times the work (-2.5 with the
    drift correction); the held plate is 16 % (11 %) from the Archimedes force, V5 decays 23 % (22 %) slow against 17 %
    without the isotropic pressure, V4 is 13.5-26 % off (8.6-11.9 % with the drift correction), a few particles climb
    the cup wall, which puts the top layer of V7 1 cm above the liquid, and 5 (14) particles cross the thin wall of V11
    R1. Not adopted as the default; the trade between the two families is in the section below.

45. Continuity of the corrected density across weight updates (development snapshot, diff in
    `worktrees/dev-dc1/final.diff`). With the isotropic density, an update of the pair weights changes the constrained
    density with no motion; an offset per particle took that change and decayed over tau (`*__dc1_*` relaxation 1 and
    tau 0.05 s, `*__dc2_*` relaxation 1 and 0.3 s, `*__dc3_*` relaxation 0.2 and 0.3 s, with the wall shear
    consistency). The offset lets the constrained density leave the arrangement: water at rest reaches 4e-3 J at tau
    0.05 s and ends in NaN at 0.3 s, the Couette runs with relaxation 1 end in NaN, the plank draft is 24-74 % off, the
    held plate 10-15 % from the Archimedes force, and the stirring balance stays at -1.4 to -2.0 times the work. Not
    adopted.

46. Shear sensitivity of the constrained density (lattice study, FluidBench
    `adapters/genesis/sph_validation/dev/shear_density_check.py` and `affine_density_check.py`). The pressure friction
    is the response of the constrained density to a shear of the neighborhood: on a lattice sheared by gamma, the
    derivative of the kernel sum under a further pure shear is 1.1e-3, 2.2e-3 and 3.5e-3 at gamma 0.05, 0.1 and 0.2
    (zero only for the unsheared lattice), so the pressure carries a shear stress rho_0 kappa times that. The isotropic
    correction does not remove it from the density: with the weights frozen the derivative is 1.5e-3, 2.8e-3 and 4.2e-3,
    and with the weights recomputed, as an exact (conservative) gradient would need, 1.9e-3, 3.4e-3 and 4.5e-3. The
    option escapes the friction only because its force drops the derivative of the weights, which is what makes it do
    work (findings 22, 44, 45); a conservative isotropic pressure would keep the friction. A density that no isochoric
    affine map of the neighborhood changes would have neither: the kernel sum in a unimodular metric, the inverse square
    root of the second moment of the neighbors in that same metric. On the lattice its shear derivative is 6.0e-4 at
    gamma 0.1 and 5.9e-5 at 0.2, with the per-particle fixed point still unconverged after 40 iterations (contraction
    about 0.83 per iteration); it reads compression and non-affine displacements as the kernel sum does, but at a free
    surface the metric collapses (0.32 for a half neighborhood against 0.85). Its exact gradient goes through the fixed
    point in every pressure iteration, about twice the cost of an iteration, against an RTF already 3.5 times short of
    the target. Not implemented in the solver.

47. Thin-wall start with the liquid sampled clear of the walls (V11; FluidBench `thinwall.py --gap`,
    `dev/thinwall_probe.py`). The leakage of V11 comes from the lattice of the spec, whose outer column starts 0.09 mm
    (R1) and 0.47 mm (R0) from the inner wall, inside the particle size from the boundary particles. Sampled as for a
    ghost-volume boundary (the same number of particles none closer than dx/2 to the cup, the free surface raised to
    keep the mass, `fluidbench.scene.sample_with_gap`), with the wall shear consistency and the drift correction of the
    positions on 8be1077b, no particle crosses the 4 mm wall or leaves the domain at R0 or R1, from the lattice or
    settled 0.5-5 s (`thinwall__twgp_*`; 19 crossed at R0 before, `thinwall__v6wp_*`). Without the drift correction one
    particle crosses at R1 and the start-up speeds reach 1-1.8 m/s (`thinwall__twg_*`); volume maps leak through the
    thin wall in every run (8-1329 particles, `thinwall__twvg*`). The start-up speed bar still fails: after release the
    rms speed of the liquid grows from 0.002-0.004 m/s to 0.008-0.011 m/s within a second whatever the settling time,
    and particles of the layer against the wall and the floor oscillate along it with a period of about 25 ms at up to
    0.14-0.17 m/s (traced substep by substep in `logs/twprobe4_R1.log`), above the 0.05 m/s bar. Water (nu 1e-6) damps a
    parcel of the particle size against the wall at about nu / dx^2 = 0.08 per s, so the oscillation, excited again and
    again, persists.

48. Energy of the wall clearance guard (fe8d9100). The guard of `_kernel_enforce_wall_clearance` gave a particle closer
    than a quarter of a particle size to a rigid surface an outward recovery velocity. Particles of the first layer rest
    that close in a cup, so the guard kicked them at every substep: water settled in the thin-wall cup of V11 (gap
    sampling, wall shear consistency, drift correction) took 5.2e-4 J per s from the guard, 4049 corrections per s, the
    only stage that added energy (`thinwall__twgpa_*`), and the wall layer oscillated at up to 0.16 m/s. The recovery
    now moves the position by the same hundredth of the deficit per substep, the velocity constraint only removes the
    approaching velocity, and a rigid rotation of the fluid cancels the angular momentum the recovery would change (V6
    drift 3.1e-6 / 7.3e-6 per s, against 8.5e-5 without the rotation, `momentum__cl3w*`, `momentum__cl1w*`). The guard
    then removes 4e-5 J per s; the stirring balance is -0.4 to -0.6 %, the held plate 4.5-5.1 % from the Archimedes
    force, the plank 4.8 % and the light box 9.6 % with the drift correction (`*__cl1wp_*`, `*__cl3wp_*`). In V11 no
    particle crosses at R0 or R1, and the largest start-up speed goes from 0.14 / 0.16 m/s to 0.083 / 0.10 m/s (R0 / R1,
    settled 1.5 s); with the density tolerance at 0.02 % instead of 0.05 % the net gain of the gravity and density
    stages drops from 4.3e-4 to 0.7-1.1e-4 J per s and the largest speed to 0.074 / 0.075 m/s (`thinwall__cl2t*`), still
    above the bar of 0.05 m/s.

49. Start of V11 from the lattice and from rest (FluidBench `thinwall__tw5*`, `thinwall__tw6*`, on baa067d1 with the
    wall shear consistency, the drift correction and the density tolerance at 0.02 %). The gap sampling left the layer
    next to the floor up to a spacing above dx/2 (4.75 mm at R1), so that the liquid fell onto the floor at t = 0;
    `sample_with_gap(..., align_bottom=True)` now puts it at dx/2 at every resolution. Even so, a cubic lattice in a
    round cup leaves a stepped gap at the wall, from dx/2 to 5 mm at R1 where a column is dropped, and the hydrostatic
    pressure drives the outer column into it at 0.35-0.6 m/s at t = 0 (`logs/twprobe8_R1.log`): a lattice is not at rest
    in a round cup, so the bar on the start-up speed from rest is met only by a settled liquid. In all twelve runs from
    the lattice and settled 1.5 s, R0 and R1, turned 0 and 10 degrees, with the divergence tolerance at 0.1 and 0.02 %,
    no particle crosses the wall or leaves the domain. The largest speed of the settled runs over 1 s is 0.062 m/s at R0
    and 0.106 m/s at R1 (0.039 / 0.082 m/s in the first 0.1 s), from single particles of the first layer at the floor
    and the wall; the bar of 0.05 m/s is not met. Boundary particles sampled at 0.35 and 0.25 particle sizes instead of
    0.5 (development snapshot, `worktrees/dev-bs1/final.diff`) do not lower it (0.077-0.128 m/s, `thinwall__bs*`), so
    the roughness of the sampled wall is not what sets it.

50. Stage A replay on the frozen thick-wall candidate (Genesis 745b4613, the dissipative clearance guard of fe8d9100;
    FluidBench receipts `*__sa0_*` for the velocity baseline, `wall_shear_consistency=True,
    density_correction="velocity"`, and `*__sa_*` for the same with `density_correction="position"`, all validation
    containers with walls of 10 mm or 3 dx; report `report/sph_goal.md`, sections "Acceptance" and "Stage A verdict").
    Against the staged targets the velocity baseline passes V3 (centerline 1.0 / 1.6 %), V5, V6 (4.3e-6 / 7.1e-6 per s),
    V7 bulk (3.6 % at dx 2.5 mm, load 3e-5), V8 at the finest resolution (plank 4.8 %, light box 8.9 %), V10 (held plate
    4.3-8.6 % from Archimedes, readout within 3.0 %, stirring balance -0.49 %), and the thick cup at rest once settled
    (rms 0.0065-0.013 m/s, no crossing). It fails V1 (profile up to 32 %, floor torque 0.88-1.92 of the reference at dx
    1.75 mm), V2 (torque up to 37 %), V9 (51-66 % slow, 370 capped solves), V4 at nu 1e-4 (26.6 %; 14.5 / 18.9 % at 1e-2
    / 1e-3 pass the staged 20 %), and V12 on both GPUs, whose envelope, the largest departure of a single particle over
    the run, reaches 0.36 after a selected reset against 0.002 between two replays. With the drift correction the same
    replay passes V4 within the staged 20 % (10.6, 8.4, 11.7 % at nu 1e-2, 1e-3, 1e-4), V12 on both GPUs and the thick
    cup at rest from every wall-clear start, and fails V1, V2 and V9 alike. The original lattice of the spec, kept as a
    control, starts inside the wall standoff and pushes particles into the thick wall (70-4177 past its middle).

51. Drift correction outside the cup (FluidBench `sphgoal-dk-*`, `sphgoal-dev-*`, `sphgoal-dj-*`; diffs in
    `worktrees/dev-dj/final.diff`). With `density_correction="position"` the solver ends in NaN in every FluidBench
    scene with a box tank, still water included (T01 at R0 and R1, the dam break T02, the floating cube T08a, the dam
    break against a free cube T07a, the settling sphere T08c), and in S1 at dt x8, with or without thick walls; the
    velocity baseline runs all of them. A trace of the drift displacements in T01 (`logs/nanprobe_T01.log`) shows no
    displacement above a tenth of a particle size for 50 substeps, then a growing number of particles at rest density
    with displacements at that size (456 at substep 100, 6682 at 350): the drift multipliers grow without a density
    error to answer. Building the tank as one watertight mesh instead of five boxes (`sph_box_mesh`), clamping the
    displacement to a tenth of a particle size, and a plain Jacobi iteration instead of the Chebyshev acceleration do
    not stop it. The drift correction is therefore not adopted; the velocity baseline is the Stage A candidate (owner
    update of 2026-10-30), and the drift correction stays an option measured beside it.

52. Thick-wall FluidBench runs of the velocity baseline (`sphgoal-sb`, Genesis 745b4613 code; walls and bottoms of 3 dx
    grown outward, `wall_overrides`, with the inner volume, the liquid and the motions unchanged; wall-clear lattice
    with the bottom layer at dx/2, `sph_gap_align`; single-mesh tanks; 0.5 s settling before t = 0; matched
    drift-correction runs `sphgoal-sa`). Usable by the FluidBench bar: S2 at R1 and R2 (onset 0.37 / 0.053 degrees, no
    leakage), the dam break with a free cube T07a, the spring plate T07b and the floating cube T08a at R1 and R2. The
    settling sphere T08c runs but 339 of its viscosity-pressure couplings stop at the cap of 30 passes. S1 (water, Stage
    B) misses L_z by 23 % and the spin-down time by 65 % at R2, and its volume drifts 1.3 % (bar 1 %); the original
    geometry and lattice (`sphgoal-sb-orig`) leak. The drift correction gives S1 a volume error of 0.6 % but ends T07a,
    T08a and T08c in NaN.

53. Speed of the velocity baseline on thick-wall S1 R2 (RTX PRO 6000, profiles `sb_prof_dt*` in the data directory,
    every solve converged): RTF 0.035 at the FluidBench time step, 0.079 at dt x4, 0.123 at dt x8 (57 density, 12
    divergence iterations per substep), NaN at dt x12. At dt x8 a substep takes 22 ms: density solve 9.0 ms (0.16 ms per
    Jacobi iteration), viscosity solve 4.8 ms, divergence solve 1.9 ms, density and factor 1.6 ms, viscosity geometry
    1.5 ms, clearance guard 0.6 ms, rigid collision 0.4 ms. With the drift correction the same runs take 11-28 % longer
    (RTF 0.027 and 0.062-0.071 at dt x1 and x4) and are unstable at dt x8. RTF 0.5 needs 4 times the speed of dt x8.

54. State restore of the velocity baseline (V12; FluidBench `state__sa0*`, `state__sa0d_*`, `state__sa0e_*` with rms
    departures and two more restarts in the receipt). Without the drift correction the verdict of V12 changes from run
    to run: the envelope bar failed in three of four runs and the first replayed step departed by 9.2e-4 once (a third
    of the departure without warm starts), while every run with the drift correction passes (`state__sa*`,
    `state__sad_*`, `state__sae_*`). No state is missing: a full reset after the perturbation of one environment and
    selected resets without it depart from the reference by rms 1.3e-4 and 3.9e-4 over the run, within the band of two
    plain restarts (2.2e-4 to 2.3e-3). Without the drift correction the particle velocities amplify the round-off of the
    GPU reductions faster, and the largest departure of a single particle, which the envelope bar takes, varies by an
    order of magnitude between equivalent pairs of runs. An extra iteration of a solve that sits at its tolerance, which
    round-off can trigger, would explain the departure of 9.2e-4 at the first step; this is not verified. Open: a V12
    bar on the distribution of departures rather than the single largest one, or repeated runs.

55. Speed of the velocity baseline in the timing runs (FluidBench `sphgoal-sbt-*`, thick-wall S1 f2 R2 at dt x8 on the
    RTX PRO 6000, every solve converged, three repeats). RTF 0.123 for one environment (8.2 s per simulated second, 89
    density iterations per substep), 0.104 at dt x6. Batches of independent environments (`env_time_shift` 0.25 s) run
    0.33 and 0.35 environment-seconds per second with 4 and 16 environments (RTF per environment 0.082 and 0.022): the
    iterations follow the worst environment (112 per substep with 16). Loosening the density tolerance from 0.05 % to
    0.1 and 0.2 % cuts the density iterations by 6 and 17 % but leaves the RTF at 0.122-0.124, and storing the viscosity
    pair coefficient of every listed pair once per substep for the viscosity solves, their iterations and the coupling
    residual (development snapshot, `worktrees/dev-vc/final.diff`; identical momentum, decay and Couette results,
    `*__vc_*`) leaves it at 0.123 (viscosity solve 4.74 against 4.80 ms per substep). The arithmetic of the pairs does
    not set the speed at 19172 particles: the GPU time goes to the launches, the host synchronization of every iteration
    (16 us) and the serial tasks of the statements outside the parallel loops of each kernel (about 6 us each, a third
    of a Jacobi iteration, finding in section "Performance (M5)"). Not adopted.

56. Volume of S1 and the settling sphere T08c on the velocity baseline (FluidBench `sphgoal-sb-tol0.02`,
    `sphgoal-sbt-tol0.02`, `sphgoal-sb-c60`, `sphgoal-sb-c100`). The volume of thick-wall S1 R2 drifts 1.27 % over the
    run (bar 1 %) and 1.18 % with the density tolerance at 0.02 % instead of 0.05 % (RTF at dt x8 0.115 against 0.123):
    the solve tolerance does not set it; the drift correction gives 0.61 % (finding 52). In T08c (ten Cate et al. E1, a
    sphere settling in oil, R3) the viscosity-pressure coupling stopped at its cap of 30 passes in 339 substeps; with a
    cap of 60 or 100 every coupling converges (8.5 and 11.3 passes per substep, RTF 0.0006 and 0.0004), and the run is
    usable by the FluidBench bar, but the sphere travels 1.6 mm in 3 s: its speed is 98.6 % below the measured terminal
    velocity, the static pressure friction of finding 17 (V9 settles 51-66 % slow, finding 50).

57. Two changes of the reductions speed the velocity baseline up to RTF 0.174 at dt x8 (FluidBench timing runs
    `sphgoal-sbt-n1`, `sphgoal-rct-n1`, `sphgoal-ect-n1`: thick-wall S1 f2 R2, one environment, three repeats, every
    solve converged; profiles `sb_*_dt8`, `rc_*_dt8`, `ec_*_dt8` in the FluidBench data directory
    `profiles/`). Each thread of the momentum projection of the viscosity solve and of the clearance guard now sums 64
    particles before one atomic addition per environment (793e9685), instead of one atomic addition per particle on the
    same few environment sums: the projection went from 3.64 to 0.31 ms of GPU time per substep and the guard from 0.61
    to 0.11 ms, RTF 0.123 to 0.153 (walls 8.27, 8.19, 8.17 s to 6.66, 6.61, 6.58 s). The pressure solves then read their
    error after the first iteration and every fourth one instead of after every iteration (18bf3cc0), which saves a
    reduction over environments and a host synchronization per iteration at the cost of up to three extra iterations:
    RTF 0.174 (walls 5.84, 5.80, 5.80 s), GPU time per substep 19.31, 15.98 and 14.41 ms for the three revisions
    (`--kernel-profiler`). Equivalence (receipts `*__rc_*`, `*__ec_*` against `*__sa0_*`): momentum drift 4.0-4.5e-6
    and 7.0-7.4e-6 per s against 4.3e-6 and 7.1e-6, mode decay error 0.189 in all three, Couette torques and profile
    within the spread of repeated runs (profile 0.137-0.147), the stirring balance -0.68 and -0.97 % against -0.49 %,
    held joints within 0.4 % of the readout; `tests/coupling/test_sph_rigid.py` and `tests/particles/test_sph.py` pass
    except `test_pressure_carries_column_weight` (WCSPH and DFSPH of the legacy solver), which fails identically on the
    starting revision 92c78327 (`logs/base_pytest2.log`). The remaining GPU time at dt x8: the pressure solves 8.5 ms
    per substep (8.0 ms in their iterations, mostly those of the density solves of the two coupling passes that almost
    every substep takes), density and factor 1.6 ms, the viscosity geometry 1.5 ms.

58. Warm start of the viscosity-pressure coupling, measured and not adopted (development snapshots on 18bf3cc0, diffs
    `worktrees/dev-cw/v1.diff`, `dev-cw/v2.diff`, `dev-cw3/final.diff` in the Genesis data directory; FluidBench
    `sphgoal-cw*t-n1` and receipts `*__cw_*`, `*__cw2_*`, `*__cw3_*`, `*__cw0_*`, `*__cw20_*`). At dt x8 almost every
    substep takes two coupling passes (1926 of 1974 in `sphgoal-ect-n1`), since the first viscosity solve sees no
    pressure. Three variants started the first pass from the pressure of the previous substep: (v1) the viscosity solve
    and the density solve both start from it; (v2) the same, with the viscous impulse of v* alone scaling the tolerances
    of the first pass; (v3) the viscosity solve reads it and the density solve keeps its usual start (half the last
    pressure). v1 and v2 run at RTF 0.206 and 0.202 against 0.174 (41 density iterations per substep against 92), but
    still take two passes in 1648 substeps: the gain is the full warm start of the density solve, which the Performance
    section already rejected at 0.75 and 1.0 of the last pressure for the noise it leaves. Here the stirring balance
    goes from -1.0 % to -7.7 and -8.6 % (bar 5 %), the mean rms speed of thick-wall water at rest at R1 from 0.008-0.009
    to 0.029 and 0.022 m/s, and the coupling of 5 and 8 of the 3000 substeps of the held plate stops unconverged. v3
    keeps every check (stirring balance -1.5 %, rest rms 0.0083 m/s, no unconverged solve, momentum, decay at nu 1e-3
    and 1e-2, Couette and held joints unchanged; unit tests as in finding 57) and lets 438 substeps finish after one
    pass, but runs at RTF 0.167: the second pass costs 14-16 density iterations against 41-44 for the first (worst
    substeps of `sphgoal-sbt-n1`), which the prediction kernel and the longer first viscosity solve offset. Not adopted.
    H100 duplicates of these runs (`cwh*`) were discarded: the worktree was edited while they compiled.

59. Kernel tasks and boundary neighbor lists (FluidBench timing runs `sphgoal-ect-n1`, `sphgoal-mot-n1`,
    `sphgoal-blt-n1`, thick-wall S1 f2 R2 at dt x8, three repeats each; profiles `ec_*_dt8`, `df_kp_dt8`, `mo_*_dt8`,
    `bl_*_dt8`; `adapters/genesis/sph_validation/dev/hash_probe.py`). Every pressure iteration ran a task of its own to
    reset the error of the environments and the wrenches of the coupled links (7.6 us, 0.57 ms per substep at dt x8);
    the first thread of the pressure acceleration loop now resets the error and the wrenches are reset at the start of
    the solve and after the links have read them. The Gram matrix of the directional viscosity correction accumulates
    its upper triangle only. RTF 0.174 to 0.177 (walls 5.75, 5.70, 5.71 s). The density kernel spends 1.55 ms of GPU
    time per substep in its boundary loop against 0.08 ms in the fluid search that builds the neighbor list (development
    split, `df_kp_dt8`); a boundary search scans 78 candidates per fluid particle for 8.1 neighbors (fluid search: 180
    for 30), at most 449, since 403 boundary particles of the stirring rod lie outside the hash grid of the domain and
    share its slots. The boundary neighbors are now listed by the first boundary search of the density kernel (up to 48,
    a particle with more searches the grid), and the second pass of the density kernel, both passes of the viscosity
    geometry, the isotropic pressure correction and the pressure readout read the list: RTF 0.180 (walls 5.64, 5.60,
    5.61 s), GPU time per substep 14.41, 14.14 and 14.04 ms. The lists barely change the time of those kernels (density
    and factor 1.53 ms, viscosity geometry 0.59 and 0.85 ms): the search is not their cost. With 19172 particles a
    kernel runs as a single wave of threads and lasts as long as its slowest thread, and a particle in a corner of the
    cup runs its many boundary neighbors, each with 8 ghost points of the wall column, in sequence. Spreading the
    boundary pairs over threads is the remaining opportunity in these kernels. Equivalence (`*__mo_*`, `*__bl_*` against
    `*__ec_*`): momentum drift 4.0-4.6e-6 and 7.9-8.0e-6 per s, mode decay error 0.189, Couette torques and profile
    within the spread of repeated runs (profile 0.143), stirring balance -0.76 and -0.91 %, held joints within 0.8 % of
    the readout, thick-wall rest at R1 rms 0.0083 m/s and largest speed 0.071 m/s without crossing; the unit tests as in
    finding 57.

60. Accuracy over time steps and resolutions on Genesis a7f11ce9 (FluidBench `sphgoal-tr-dt{1,2,4,8}` full-horizon runs
    and `sphgoal-trt-dt{1,4,8}` timing runs, thick walls, velocity baseline, every solve converged except T07a at dt x4;
    FluidBench `report/sph_goal.md`). S1 water (Stage B) keeps its errors from dt x1 to x8: at R2 the angular momentum
    error 0.23 to 0.25 (bar 0.20), the spin-down time -65.4 to -67.4 % (bar 30 %), the volume drift 1.3 to 0.81 %; at R1
    0.36 to 0.42, -65.5 to -68.1 %, 0.88 to 0.31 %. The water dynamics, not the time step, set the S1 verdict. S2
    pouring is usable at every time step at R1 and R2 with the same onset and transfer as at dt x1 (output RTF 0.043 to
    0.193 at R1, 0.026 to 0.091 at R2). The floating box T07a is usable at dt x1 and ends in NaN at dt x4 (45
    unconverged solves); the run at dt x8 slowed to 0.4 frames per second and was stopped by its time limit of 3 h
    without a record (Slurm job 767085): dynamic links need the FluidBench time step. Timing runs (one environment,
    RTF): R1 0.077 / 0.191 / 0.302, R2 0.049 / 0.117 / 0.172, R3 0.017 / 0.039 / 0.072 at dt x1 / x4 / x8, against the
    0.5 target.

61. Shared neighbor loops in the pressure iterations (FluidBench `sphgoal-gpt-n1`, thick-wall S1 f2 R2 at dt x8, three
    repeats; profile `gp_kp_dt8`; receipts `*__gp_*`). The two neighbor loops of every divergence and density iteration
    split the listed neighbors of each particle among 4 threads, whose partial sums a loop over the particles adds in a
    fixed order (the result stays independent of the thread order). With 19172 particles a loop over the particles is a
    single wave of threads that lasts as long as its slowest thread: the pressure acceleration loop went from 35.8 to
    21.8 us and the predicted source loop from 52.5 to 39.4 us, the update now a loop of its own of 16.0 us. RTF 0.180
    to 0.195 (walls 5.22, 5.17, 5.15 s). Equivalence against `*__bl_*`: momentum drift 4.5e-6 and 7.9e-6 per s, mode
    decay error 0.189, Couette torques and profile within the spread of repeated runs (profile 0.142), stirring balance
    -0.63 %, held joints within 0.1 % of the readout, thick-wall rest at R1 rms 0.0080 m/s against 0.0080-0.0088 m/s in
    the four earlier runs, its largest speed 0.117 m/s in a single one of 62 steps (0.071-0.106 m/s in the earlier
    runs); the unit tests as in finding 57.

62. Error of the pressure iterations summed only when read (FluidBench `sphgoal-ert-n1`, thick-wall S1 f2 R2 at dt x8,
    three repeats; profile `er_kp_dt8`; receipts `*__er_*`). Each particle added its error to the error of its
    environment in every iteration, an atomic addition per particle on a single address: after the shared neighbor loops
    of finding 61 the update loop of the iteration took 39 us, more than either neighbor loop. The particles now store
    their error, and the solve sums it in chunks of 64 particles when it reads it (after the first iteration and every
    fourth). The update loop takes 7.7 us per iteration, GPU time per substep 13.96 to 11.61 ms, RTF 0.195 to 0.232
    (walls 4.36, 4.34, 4.35 s). Equivalence against `*__gp_*`: momentum drift 4.2e-6 and 7.0e-6 per s, mode decay error
    0.189, Couette torques and profile within the spread of repeated runs (profile 0.146), stirring balance -0.67 %,
    held joints within 0.1 % of the readout, thick-wall rest at R1 rms 0.0079 m/s and largest speed 0.047 m/s.
    `test_fluid_load_readout_is_link_impulse` failed once in seven runs of this revision (mean load 0.081 N against its
    floor of 0.094 N, in one of two runs of the whole suite) and passed in four separate runs of it and in the four
    separate runs on 676daa52: a margin of that test to the round-off of the GPU reductions, not a change of the load;
    the other unit tests as in finding 57.

63. The other sums over the environments in chunks (FluidBench `sphgoal-vrt-n1`, `sphgoal-g8t-n1`, thick-wall S1 f2 R2
    at dt x8, three repeats; receipts `*__vr_*`, `*__g8_*`). The conjugate gradient iterations of the viscosity solve
    (two sums per iteration), its initialization (three sums and the stiffness bound of the coupling acceleration), the
    coupling residual (three sums) and the dissipation readout added one term per particle to the sums of the
    environment; the particles now store their terms and a loop over chunks of 64 particles adds them, as for the error
    of the pressure iterations (finding 62). RTF 0.232 to 0.246 (walls 4.09, 4.10, 4.10 s). Equivalence against
    `*__er_*`: momentum drift 4.3e-6 and 7.4e-6 per s, mode decay error 0.189, Couette torques and profile within the
    spread of repeated runs (profile 0.146), stirring balance -0.66 %, held joints within 0.1 % of the readout,
    thick-wall rest at R1 rms 0.0082 m/s and largest speed 0.054 m/s; the unit tests as in finding 57. Eight instead of
    four threads per particle in the neighbor loops of the pressure iterations (finding 61) gave RTF 0.227 with the same
    checks (rest rms 0.0077 m/s): not adopted.

64. Two more task reductions (FluidBench `sphgoal-rft-n1`, `sphgoal-vgt-n1`, `sphgoal-cbt-n1`, thick-wall S1 f2 R2 at dt
    x8, three repeats; receipts `*__rf_*`, `*__vg_*`, `*__cb_*`). The response of the coupled links in the pressure
    iterations runs in the first threads of the loop that adds the shares of the pressure accelerations instead of a
    loop of its own (RTF 0.246 to 0.248), and the wall viscosity tensors of the viscosity geometry share the listed
    boundary neighbors of a particle among 4 threads as the pressure iterations do (finding 61), their slot tensors
    added in a fixed order (RTF 0.252). Both together: RTF 0.254 (walls 3.98, 3.95, 3.97 s). Equivalence of the
    combination against `*__vr_*`: momentum drift 4.3e-6 and 7.2e-6 per s, mode decay error 0.189, stirring balance
    -0.68 %, held joints within 0.1 % of the readout, thick-wall rest at R1 rms 0.0080 m/s and largest speed 0.050 m/s,
    Couette torques as before and profile error 0.153 (0.137-0.147 in the eleven earlier runs since finding 57, 0.144
    and 0.145 for each change alone); the unit tests as in finding 57.

65. Speed of the Stage A baseline on 138b2c5f (FluidBench `sphgoal-fin-*` timing runs: thick-wall S1 f2 and S2 T4, RTX
    PRO 6000, three repeats each, every solve converged; `sphgoal-bis-*` batch repeats). One environment, RTF of S1 at
    dt x1 / x4 / x8: R1 0.090 / 0.237 / 0.389, R2 0.060 / 0.163 / 0.257, R3 0.025 / 0.067 / 0.120; S2 at R2 0.031 at dt
    x1 and 0.130 at dt x8. Independent batches of S1 R2 at dt x8 (`env_time_shift` 0.25 s): 4 environments 0.444
    environment-seconds per second (RTF 0.111 each, peak GPU memory 2079 MB); 16 environments either 0.40-0.42 (about
    2.0 coupling passes and 114-132 density iterations per substep; 138b2c5f twice, 219e1dfb twice, 48127a19, 18bf3cc0
    0.385) or 0.16-0.18 (about 5 coupling passes and 224-233 density iterations per substep; 138b2c5f twice, 219e1dfb,
    676daa52, 3203a547), on the same code from run to run (peak 2561 MB). The solves of a batch iterate until its worst
    environment converges, and in four of ten runs one environment needs about five coupling passes in most substeps:
    the round-off of the GPU reductions selects the branch. Letting environments stop independently (candidate 9) is the
    remaining batch opportunity.

66. Environments stop their solves independently (candidate 9; FluidBench `sphgoal-iet-*` timing runs on the development
    snapshot of 3123360d with the change, receipts `*__ie_*`). An environment takes part in the coupling passes of a
    substep until its own fluid and rigid coupling residual is at most 1, and in the iterations of a pressure solve
    until its own average error meets its own target (the tolerance, or for a later coupling pass a tenth of its own
    first error); the kernels of the passes and iterations skip the other environments, which keep the state of their
    last pass or iteration. Before, every environment iterated until the worst one converged. Batches of S1 R2 at dt x8:
    16 environments 0.461, 0.491 and 0.350 environment-seconds per second (2.9, 2.4 and 5.5 coupling passes per substep
    in the worst environment) against 0.40-0.42 and 0.16-0.18 (finding 65); 4 environments 0.442 against 0.444; one
    environment RTF 0.248 against 0.254 (walls 4.04, 4.10, 4.06 s against 3.98, 3.95, 3.97 s: the activation, the error
    of the solve that ended and the coupling flags are kernels of their own). Equivalence against `*__cb_*`: momentum
    drift 4.4e-6 and 7.4e-6 per s, mode decay error 0.189, Couette torques and profile within the spread of repeated
    runs (profile 0.138), stirring balance -0.94 %, held joints within 0.1 % of the readout, thick-wall rest at R1 rms
    0.0082 m/s and largest speed 0.053 m/s; the unit tests as in finding 57.

67. The gap between a wall and the fluid nearest to it as an integer (FluidBench `sphgoal-gat-n1`, `sphgoal-hmt-n1`,
    thick-wall S1 f2 R2 at dt x8; profiles `df2_kp_dt8`, `df3_kp_dt8`, `hm_kp_dt8`, `ga_kp_dt8`; receipts `*__ga_*`,
    `*__hm_*`). Split into its loops (development snapshot `dev-df2`), the density kernel spent 1.34 ms per substep in
    the boundary search that assigns the slots and 0.24 and 0.08 ms in the boundary and fluid density sums; without the
    atomic minimum of the gap of the boundary particles (`boundary_particles_fluid_gap`, a profiling variant) the search
    took 0.20 ms. A floating-point atomic minimum runs as a compare-and-swap loop on the GPU. The gap is now held in
    2^24 units of the support radius and its minimum taken by an integer atomic: the density kernel takes 0.53 ms
    instead of 1.55 ms, GPU time per substep 10.70 to 9.30 ms, RTF 0.248 to 0.282 (walls 3.59, 3.57, 3.58 s). Hashing
    only the boundary particles within a support radius of the domain (the stirring rod leaves it; development snapshot
    `dev-hm`) left the density kernel at 1.55 ms and the RTF at 0.253: the candidates of the search were not its cost;
    not adopted. Equivalence of the integer gap against `*__ie_*`: momentum drift 4.3e-6 and 7.8e-6 per s, mode decay
    error 0.189, Couette torques and profile within the spread of repeated runs (profile 0.140), stirring balance -0.75
    %, held joints within 0.1 % of the readout, thick-wall rest at R1 rms 0.0083 m/s and largest speed 0.055 m/s; the
    unit tests as in finding 57.

68. Speed on cc03c40b and four changes measured and not adopted (FluidBench `sphgoal-lad-*`, `sphgoal-lst-n1`,
    `sphgoal-gmt-n1`, `sphgoal-p0t-n1`, thick-wall S1 f2 at dt x8; profiles `lad_*`, `ls_*`, `gm_*`, `p0_*`, `hm3_*`;
    receipts `*__ls_*`, `*__gm_*`, `*__p0_*`). One environment at dt x8: RTF 0.408 / 0.282 / 0.126 at R1 / R2 / R3
    (walls 2.49, 2.47, 2.41 s at R1). A substep takes about the same time at R1 (7157 particles, 7.8 ms) as at R2 (19172
    particles, 8.1 ms): the GPU runs some 560-610 tasks per substep, most of them at a floor of 6-8 us, and the kernel
    profiler sums to the wall time (7.83 ms at R1), so the solver is bound by the number and the latency of its GPU
    tasks rather than by their arithmetic. Not adopted: (a) hashing only the boundary particles within reach of the
    domain on top of finding 67 (density kernel 0.54 ms either way, R1 RTF 0.42-0.43 against 0.41-0.43); (b) holding the
    boundary slots in registers through the two boundary passes of the density kernel (RTF 0.279, density kernel 0.62
    ms); (c) sharing the neighbors and ghost points of the viscosity moments among 4 threads (RTF 0.277, viscosity
    geometry 1.29 ms against 1.10 ms at R2); (d) the first coupling pass of a viscous fluid working to the tolerance
    instead of a tenth of it, since the next pass continues from its pressure: 62 instead of 92 density iterations per
    substep, RTF 0.298 at R2 and 0.45 at R1, but the thick-wall cup at rest at R1 reaches an rms speed of 0.044 m/s
    (0.008 m/s before) and a largest speed of 0.22 m/s, the particle-scale noise of finding 15. All four kept the other
    checks (momentum, decay, Couette, stirring balance -0.75 to -1.07 %, held joints, unit tests).

69. The iterations of the pressure solves as a CUDA graph, measured and not adopted (development snapshot `dev-gr` on
    4f781e61, diff `worktrees/dev-gr/final.diff`; FluidBench `sphgoal-grt-n1`, profiles `gr_*`, receipts `*__gr_*`). The
    iterations of a solve ran inside `qd.graph.do_while` of a `@qd.kernel(graph=True)`, as the constraint solver of the
    rigid solver does: the iteration count, the Chebyshev weight and the convergence test on the GPU, the error read
    every fourth iteration on the GPU, one host read of the iteration count per solve. Every check kept its value
    (momentum drift 4.0e-6 and 6.4e-6 per s, mode decay 0.189, Couette profile 0.139, stirring balance -1.0 %, held
    joints within 0.4 %, thick-wall rest largest speed 0.049 m/s, unit tests as in finding 57), but the speed did not
    change: RTF 0.275 at R2 (0.282 without) and 0.427-0.440 at R1 (0.408-0.425). The host launches and synchronizations
    were not the cost; the GPU tasks themselves are, each at a floor of 6-8 us (finding 68), which a graph does not
    shorten. RTF 0.5 at R2 therefore needs fewer GPU tasks per substep or fewer iterations, and the one measured way to
    fewer iterations raises the rest noise (finding 68).

70. The first coupling pass to the tolerance with two plain Jacobi iterations at the end of every accelerated density
    solve, measured and not adopted (development snapshot `dev-sm` on b1fd757c, diff `worktrees/dev-sm/final.diff`;
    FluidBench `sphgoal-smt-n1`, receipts `*__sm_*`). Plain Jacobi damps the particle-scale error first, so the
    smoothing was to remove the noise of change (d) of finding 68 at its lower iteration count: 70 density iterations
    per substep, RTF 0.289 at R2 and 0.424-0.437 at R1, but the thick-wall cup at rest at R1 keeps an rms speed of 0.051
    m/s (largest 0.21 m/s) against 0.008 m/s, and `test_fluid_load_readout_is_link_impulse` failed in the run of the
    unit tests. The noise of a first pass stopped at the tolerance is not removed by two plain iterations at the end of
    the solves.

71. Three instead of four GPU tasks per pressure iteration, measured and not adopted (development snapshot `dev-f3` on
    c00833b2, diff `worktrees/dev-f3/final.diff`; FluidBench `sphgoal-f3t-n1`, receipts `*__f3_*`). The second neighbor
    loop read the pressure accelerations as the sums of their four shares and the response of the coupled links from
    their wrenches, so that the loop adding the shares was dropped. Every check kept its value (rest rms 0.0083 m/s,
    unit tests as in finding 57), but reading four shares per neighbor costs more than the task it saves: RTF 0.248 at
    R2 (0.282 with four tasks) and 0.428-0.445 at R1 (0.408-0.425).

## Validation at 47499e5c (v5)

Receipts tagged `v5*` (FluidBench `report/sph_goal.md`, acceptance matrix): `v5` default, `v5i` isotropic pressure
and density, `v5s` volume maps with them, `v5v` volume maps alone, `v5p` drift correction of the positions. The
deciding receipt is the finest resolution where the design asks for it; a receipt with capped solves is no valid
sample.

| Case | v5 | v5i | v5s | v5v | v5p |
| --- | --- | --- | --- | --- | --- |
| V1 Stokes first problem | fail | fail (torque) | fail | fail | fail |
| V2 Couette | fail | pass | fail | fail | fail |
| V3 pipe | fail (17-18 %) | fail | pass | pass | fail |
| V4 decay, nu 1e-2 / 1e-3 / 1e-4 / 1e-6 | fail (16.5-25 %, 1e-6 unmeasurable) | fail | fail | fail | fail (10.1-12.6 %) |
| V5 spin-down | pass | pass | pass | pass | pass |
| V6 momentum | pass | pass | pass | pass | pass |
| V7 hydrostatics (bulk) | pass | fail | fail | pass | fail |
| V8 plank / light box at dx 1.75 mm | pass / pass (4.9 / 9.7 %) | pass / fail | fail | fail | capped solves |
| V9 settling sphere | fail | capped solves | fail | fail | fail |
| V10 held plate at dx 2.5 mm | pass | fail | fail | fail | fail |
| V10 stirring | fail at 47499e5c, pass at 97aebd75 (finding 34) | fail | fail | pass | pass |
| V11 thin-wall cup | fail | not run | not run | fail | fail |
| V12 state on RTX PRO 6000 | pass | fail | fail | fail | not run |

## Validation at cab47d96 (final campaign)

The validation receipts are tagged `v4_*`, `v4i_*` and `v4s_*` (FluidBench `results/sph_validation/`), with the full
rows in FluidBench `report/sph_goal.md`. There are three configurations:

- default: boundary particles, no isotropic pressure;
- isotropic: boundary particles with `isotropic_pressure=True, isotropic_density=True`;
- volume maps: `boundary_model="volume_map"` with the isotropic pressure and density.

The validation runs ran on NVIDIA H100, except the RTX row of V12. Errors are relative to the reference of
SPH_SOLVER_DESIGN.md section 9.

| Case (bar) | Default | Isotropic | Volume maps |
| --- | --- | --- | --- |
| V1 Stokes first problem, nu 1e-4 (profile 10 %, torque 15 %) | profile 8-34 %, torque x1.36-2.64: fail | profile 1.3-8.9 % at dx 2.5 mm, torque x0.51-0.97: fail | profile 6-16 %, torque x0.29-0.76: fail |
| V2 Couette dx 3.5 mm (torque 15 %, profile 10 %) | 12-45 %, profile 15 %: fail | 0.02-6.7 %, profile 2.7 %: pass | 14-39 %, profile 11 %: fail |
| V3 pipe, nu 1e-2, dx 2.5 / 1.875 mm (centerline 10 %) | 17.1 / 17.8 %: fail | 17.8 / 18.9 %: fail | 4.4 / 5.0 %: pass |
| V4 mode decay, nu 1e-3, dx 2.5 mm (rate 10 %) | 17.4 %: fail | 15.8 %: fail | 11.7 %: fail |
| V5 spin-down, nu 1e-2 (rate 20 %) | omega rms 2.1 %: pass | 3.2 %: pass | 1.35 %: pass |
| V6 momentum drift (1e-4 per s) | 5.1e-6 / 7.6e-6: pass | 1.2e-4 / 1.5e-4: fail | 1.2e-4 / 1.6e-4: fail |
| V7 sphere resting on the floor (no bar) | buoyancy 24.5 % short, balance 0.85 % | 16.2 %, -0.37 % | 7.1 %, -0.006 % |
| V8 plank of density 500, dx 3.5 / 2.5 mm (draft 10 %) | 14.3 / 8.8 %: fail / pass | 6.7 / 0.9 %: pass | 14.4 / 14.9 %: fail |
| V8 light box of density 100 (draft 10 %) | 29.5 %: fail | not run | 27.6 %, tilt 6.7 deg: fail |
| V9 settling sphere, nu 1e-2, dx 3.5 / 2.5 mm (terminal velocity 20 %) | 56 / 90 % slow: fail | 23 / 18.5 % slow: fail / pass | 97 / 99 % fast: fail |
| V10 held plate (controller vs readout 5 %) | 19 / 31 %: fail | 0.9 / 7.8 %: fail | 4.2 / 5.5 %: fail |
| V10 stirring paddle (energy balance 5 %) | -14.6 %, torque readout 0.25 %: fail | -878 %: fail | -761 %: fail |
| V10 paddle at rest, nu 1e-3 | 1.9e-4 J dissipated in 1.5 s | 0.21 J, 57 unconverged solves | 0.17 J |
| V11 thin-wall cup, 4 mm wall, 1 s from rest (no crossing, start-up speed 0.05 m/s) | R1 (dx 3.5 mm), offset 1.35 mm: 4607 of 7157 particles through the wall, 6.0 m/s in the first step (cup turned 10 deg: 4603, 5.4 m/s); settled 0.5 s first: 9, 1.6 m/s; offset 0.7 particle sizes: 4850, all but 6 out of the domain, 115 m/s; R0 (dx 5 mm, wider than the wall): 1985 from rest (turned: 2013), 200 settled, 0.7 particle sizes 261 m/s: fail | not run | R1: 1 from rest, 1.6 m/s, 7 settled; R0: 451 from rest, 570 settled: fail |
| V12 restart and selected reset | H100: first replayed step 1.4e-3 against 2.8e-3 without warm start: fail; RTX PRO 6000: 1.8e-4 against 3.0e-3: pass | H100: pass | H100: pass |
| V13 FluidBench S1-S3 | see the S1-S3 table below | not run | see below |

The V11 receipts are `thinwall__v4_thin_*` and `thinwall__v4s_thin_*`; the speed is the largest
particle speed over the run. The plank at dx 1.75 mm did not finish within the 4 h limit of its job in either
configuration. The isotropic pressure heats the stirred liquid (stirring paddle balance) and breaks the momentum
conservation of V6. This is the non-conservative correction of finding 22.

FluidBench S1-S3 on cab47d96 (records `gs-dfsph__*@sphgoal-v4` and `@sphgoal-v4s` in FluidBench `results/index`, NVIDIA
H100). Both configurations sample the lattice of the spec with a particle size of dx and a particle volume factor of 1,
and let it settle quasi-statically for 0.5 s before t = 0 (finding 20). The default configuration uses a boundary
offset of 1.35 mm. Bar: the usable bar of FluidBench (volume 1 %, no leakage, S1 L_z 20 % and spin-down time 30 %,
S2 onset 5 deg and transferred share 0.05, S3 mixing index 0.1).

| Scene | Default | Volume maps |
| --- | --- | --- |
| S1 f2 R1 | volume 2.7 %, leakage, L_z 35 %: not usable | volume 1.8 %, leakage, L_z 30 %: not usable |
| S1 f2 R2 | leakage, L_z 24 %, spin-down -70.9 %: not usable | volume 0.78 %, L_z within 20 %, leakage, spin-down -54.2 %: not usable |
| S1 f2 R3 | volume 1.0 %, L_z 28 %, spin-down -75.2 %: not usable | volume 0.93 %, L_z within 20 %, no leakage, spin-down -56.4 %: not usable |
| S2 T4 R1 | usable | usable |
| S2 T4 R2 | unstable (27.9 m/s, above 10 times the characteristic speed) | usable |
| S2 T4 R3 | did not finish within 4 h | did not finish within 4 h |
| S3 M0 R1 | unstable (0.34 % of the liquid out of the domain) | volume 1.2 %, mixing index fails |
| S3 M0 R2 | mixing index fails (M2 5.0 / 4.9 tolerances) | mixing index fails (4.1 / 4.4) |
| S3 M0 R3 | mixing index fails (2.9 / 2.9) | mixing index fails (2.1 / 2.4) |

No run had an unconverged pressure solve (iteration cap 100).

## Measured validation at development snapshots (superseded by the section above)

Full boundary gradient at a boundary spacing of 0.5 particle sizes, isotropic pressure off (receipts tagged `fe_*`), and
the same with `isotropic_pressure=True` (`fi_iso_*`, `fgE_*`).

| Case | Default | With isotropic pressure | Receipt |
| --- | --- | --- | --- |
| V2 Couette dx 3.5 mm, depth 0.05 / 0.08 m, t 0.05-1 s | torque +10-27 % / +26-48 % | 0.3-3.4 % / 2-10 % | `couette__fe_couette_dx0.0035`, `couette__fi_iso_couette_dx0.0035` |
| V4 mode decay nu = 1e-3, dx 2.5 mm | rate -23 %, dissipation readout 0.995 of the energy loss | -23 % | `decay__fe_decay_nu1e-3_dx0.0025`, `decay__fgE_decay` |
| V5 spin-down nu = 1e-2, dx 5 mm | decay rate -11.6 %, omega rms error 2.5 % | -17 %, 3.0 % | `spindown__fe_spin_arm5e-4`, `spindown__fi_iso_spin_arm5e-4` |
| V6 momentum, nu = 1e-3 | 9.2e-6 / 9.0e-6 per s (limit 1e-4) | | `momentum__fe_mom_nu1e-3` |
| V8 plank of density 500, dx 3.5 mm | draft -14 %, fluid load +2.6 % of the weight, tilt 0.5 deg | | `floating__fe_plank500_dx0.0035` |
| Contact, sphere of density 2000 on the floor | fluid + contact = weight within 0.4 %, buoyancy -23 % (no fluid under the contact cap) | | `contact__fe_contact` |
| V10 paddle at rest | dissipation 2.0e-4 J over 1.5 s | | `articulation__fe_stir0` |
| Water at rest, closed column | kinetic energy decays (3.3e-4 to 1.3e-4 J in 0.9 s) | grows (2.3e-3 to 3.7e-3 J); at nu = 1e-3 held at 3 mm/s rms | `logs/fe_cn.log`, `logs/fg_cn_full50_iso.log`, `logs/fi_cn_iso_nu1e-3.log` |

Earlier snapshots (projected boundary gradient, development record): V2 2-9 % at depth 0.05 m and dx 5 mm
(`couette__g3_pbs_dx*`), V4 -39 %, -17 %, -13 % at dx 3.5, 2.5, 1.75 mm (`decay__face2_decay_nu1e-3_dx*`), V10 hold
joint forces vs readout within 1 % (`articulation__g3_hold`), stir joint torque vs fluid torque 1.4-3.5 %
(`articulation__mid2_stir`).

FluidBench S1 (stirred water, f2, R2, 19172 particles; OpenFOAM spin-down time 7.86 s and steady angular momentum
7.86e-4; the FluidBench usability bar asks for 30 % and 20 %):

| Configuration | Status | tau_sd (s) | Lz steady | RTF (output run) |
| --- | --- | --- | --- | --- |
| default, dt 0.31 ms (52 substeps / 16 ms) | 0.1 % of the liquid left the domain at 4.4 s | 1.97 | 9.38e-4 | 0.033 |
| default, dt x2 / x4 / x8 | ok | 1.73 / 1.66 / 1.66 | 9.09e-4 / 8.99e-4 / 8.86e-4 | 0.057 / 0.086 / 0.137 |
| isotropic pressure, dt 0.31 ms | a particle reached 20 m/s at 0.16 s | 2.78 | 8.08e-4 | 0.026 |

Records `gs-dfsph__S1__f2@sphgoal-fe*`, `@sphgoal-fgE` in FluidBench `results/index`. No configuration meets the
spin-down bar: in water the spin-down is set by the numerical friction of the walls and of the pressure pairs, and by
the particle noise.

## Performance (M5)

Profiles of FluidBench S1 f2 R2 (19172 particles, 45280 boundary particles, one environment) with
`adapters/genesis/tools/dfsph_profile.py`; receipts in FluidBench `results/profiles/sphgoal/`. Times are per substep of
the plain (uninstrumented) windows, two windows each; `--kernel-profiler` adds the GPU time of every task.

Where the time goes (`nl3_kp_dt1`, GPU time per substep at dt 0.31 ms): viscosity geometry 1.9 ms (its two boundary
passes over the ghost columns, 0.95 ms each), one Jacobi iteration of the density solve 0.165 ms (two neighbor loops of
67 and 40 us, the rest in small serial and coupled-link tasks of about 6 us each), density and factor 0.64 ms. Over a
full S1 run the density solve takes 47 iterations per substep at the FluidBench tolerance (average density error
0.05 %), 94, 114 and 170 at time steps x2, x4, x8 (records `@sphgoal-nl1-it100-dt*`); with the default cap of 10
iterations those solves stop unconverged at the larger steps (`fe_coarse_dt8`: all 882 density solves).

Uninstrumented timings of the committed revisions (FluidBench timing runs: S1 f2 R2, one environment, window 1-2 s of
the stirring, three repeats, iteration cap 100 so that every solve converges; records
`gs-dfsph__S1__f2@sphgoal-abl-<revision>-dt<x>` in FluidBench `results/index`):

| Revision | RTF at dt 0.31 ms (x1) | wall per repeat (s) | RTF at dt 2.5 ms (x8) | wall per repeat (s) | density iterations per substep (x1 / x8) |
| --- | --- | --- | --- | --- | --- |
| 26832080b (model) | 0.0162 | 62.2, 66.2, 59.3 | 0.0438 | 23.0, 23.0, 23.2 | 47.9 / 165.6 |
| cbc79cd98 neighbor lists | 0.0241 | 41.8, 43.1, 40.6 | 0.0735 | 13.6, 13.7, 13.7 | 51.0 / 165.2 |
| 51e488d49 fused iteration | 0.0290 | 37.8, 34.7, 33.6 | 0.0818 | 12.2, 12.3, 12.5 | 47.8 / 165.5 |
| 0ea1569d5 Chebyshev | 0.0412 | 24.5, 24.7, 24.3 | 0.156 | 6.4, 6.5, 6.5 | 19.5 / 67.9 |
| dc03db16e reentrant edges closed | 0.0521 | 19.3, 19.6, 19.4 | 0.151 | 6.6, 6.7, 6.8 | 10.1 / 69.9 |
| cab47d96 default, settled lattice at rest density (final campaign, `sphgoal-t4-dt*`) | 0.0394 | 25.6, 25.7, 25.6 | 0.142 | 7.1, 7.0, 7.1 | 12.5 / 92.0 |
| cab47d96 volume maps with isotropic pressure and density (`sphgoal-t4s-dt*`) | 0.0321 | 31.4, 31.4, 31.1 | 0.139 | 7.3, 7.3, 7.2 | 23.5 / 50.5 |

The target RTF of 0.5 is not reached. The fastest converged configuration ran at 0.156 (Chebyshev revision, dt x8).
The final configurations run at 0.142 (default) and 0.139 (volume maps) at dt x8, and 0.084 / 0.097 at dt x4. On
the settled lattice at rest density the density solves of the default configuration take 92 iterations per substep at
dt x8, against 69.9 at dc03db16e on the adapter lattice. S1 is not accuracy-qualified at any time step (section
"Validation at cab47d96").

| Candidate (SPH_SOLVER_GOAL.md) | State | Evidence |
| --- | --- | --- |
| 1 boundary geometry per fluid particle and link | implemented: gradient, wall tensor and clearance per boundary slot, computed once per substep and read by every iteration | solver design (M2) |
| 2 slot allocation of the hash grid | measured, not changed: reordering takes 0.07 ms per substep (1 %) | `fe_fine` |
| 3 host readback in the iterations | proposed: one scalar per iteration, synchronization 16 us per call | `fe_fine` (`sync_s_per_call`) |
| 4 static and moving boundary indexing | proposed: the boundary update takes 0.02 ms per substep | `nl3_kp_dt1` |
| 5 cached neighbor indices | implemented, validated, measured: 9.0 to 6.6 ms per substep at dt x1, 12.8 to 8.9 ms at dt x8 (1.36x, 1.44x) | `fe_coarse_dt*`, `nl1_coarse_dt*`; equivalence `momentum__nl1_mom_nu1e-3`, `decay__nl1_decay_nu1e-3_dx0.0025`, `couette__nl1_couette_dx0.0035` |
| 6 reorder and copy traffic | measured, not changed: under 2 % | `fe_fine` |
| 7 preconditioner reuse | proposed | |
| 8 block reductions | implemented, measured: the link wrench accumulates in the pressure acceleration pass, and only particles with an error add it to their environment; the predicted-source task of an iteration went from 67 to 50 us, a substep at dt x8 from 31.1 to 28.6 ms (iteration cap 100) | `nl3_kp_dt1`, `it1_kp_dt1`, `nl3_it100_coarse_dt8`, `it1_it100_coarse_dt8` |
| 10 prescribed-boundary path | proposed | |
| packed moment matrix of the viscosity correction | measured, not adopted: 1.99 to 1.93 ms | `nl2_fine_dt1` |
| compile-time constants in the iteration kernel | implemented, measured: statements outside the loops of a kernel run as serial GPU tasks of 6-7 us; one fewer per iteration, 28.6 to 28.1 ms per substep at dt x8 | `it2_kp_dt1`, `it2_it100_coarse_dt8` |
| Chebyshev semi-iterative density solves | implemented, validated, measured: bound 0.98 (best of 0.95 to 0.998 at every time step), on the density solves of the first two coupling passes, which work to a tenth of the tolerance (finding 15); 21.5, 56.1 and 69.3 iterations per substep at dt x1, x4, x8 against 51.1, 123.2 and 166.8, RTF 0.041, 0.099, 0.155 against 0.027, 0.048, 0.076, all solves converged; paddle at rest (nu = 1e-3) 57 s against 130 s with 6.9 coupling passes per substep and none unconverged | `ch3_dt*`, `cheb0_dt*`, `cheb*_dt*`; validation `*__ch3_*`, `logs/ch3_cn.log` |
| warm start of the density solve at 0.75 or 1.0 of the last pressure instead of 0.5 | measured, not adopted: 45 to 41 and 35 iterations per substep at dt x1, but water at rest stays noisier (closed column kinetic energy 3.7e-4 to 2.6e-4 J over 0.9 s against 3.3e-4 to 1.3e-4 J) | `gs-dfsph__S1__f2@sphgoal-ws*`, `logs/ws1.0_cn.log` |
| chunked reductions of the viscosity projection and the clearance guard | implemented, validated, measured: RTF 0.123 to 0.153 on thick-wall S1 R2 at dt x8 (finding 57) | `sphgoal-rct-n1`, `rc_kp_dt8`; equivalence `*__rc_*` |
| error of the pressure solves read every fourth iteration | implemented, validated, measured: RTF 0.153 to 0.174 (finding 57) | `sphgoal-ect-n1`, `ec_kp_dt8`; equivalence `*__ec_*` |
| kernel task of the iteration resets folded into its loops, upper triangle of the viscosity Gram matrix | implemented, validated, measured: RTF 0.174 to 0.177 (finding 59) | `sphgoal-mot-n1`, `mo_kp_dt8`; equivalence `*__mo_*` |
| boundary neighbor lists | implemented, validated, measured: RTF 0.177 to 0.180; the boundary kernels are bound by their slowest thread, not the search (finding 59) | `sphgoal-blt-n1`, `bl_kp_dt8`, `df_kp_dt8`; equivalence `*__bl_*` |
| neighbors of a particle shared among 4 threads in the pressure iterations | implemented, validated, measured: RTF 0.180 to 0.195 (finding 61) | `sphgoal-gpt-n1`, `gp_kp_dt8`; equivalence `*__gp_*` |
| error of the pressure iterations summed in chunks when read | implemented, validated, measured: RTF 0.195 to 0.232 (finding 62) | `sphgoal-ert-n1`, `er_kp_dt8`; equivalence `*__er_*` |
| sums of the viscosity solve, coupling residual and dissipation readout in chunks | implemented, validated, measured: RTF 0.232 to 0.246 (finding 63) | `sphgoal-vrt-n1`; equivalence `*__vr_*` |
| 8 instead of 4 threads per particle in the pressure iterations | measured, not adopted: RTF 0.227 (finding 63) | `sphgoal-g8t-n1`, `*__g8_*` |
| link response folded into the share loop; wall tensors over shared boundary neighbors | implemented, validated, measured: RTF 0.246 to 0.254 (finding 64) | `sphgoal-rft-n1`, `sphgoal-vgt-n1`, `sphgoal-cbt-n1`; equivalence `*__rf_*`, `*__vg_*`, `*__cb_*` |
| 9 independent stop of environments | implemented, validated, measured: batches of 16 at 0.35-0.49 environment-seconds per second against 0.16-0.42, one environment 2 % slower (finding 66) | `sphgoal-iet-*`; equivalence `*__ie_*` |
| integer atomic minimum of the wall gap | implemented, validated, measured: RTF 0.248 to 0.282 (finding 67) | `sphgoal-gat-n1`, `ga_kp_dt8`; equivalence `*__ga_*` |
| boundary particles hashed only within reach of the domain | measured, not adopted: RTF 0.253 (finding 67) | `sphgoal-hmt-n1`, `*__hm_*` |
| warm start of the viscosity-pressure coupling | measured, not adopted: RTF 0.206 with the full warm start of the density solve, which breaks the stirring balance and the rest noise; 0.167 without it (finding 58) | `sphgoal-cw*t-n1`, `*__cw*_*` |
| 4 instead of 8 ghost points per wall column | measured, not adopted: 6.65 to 6.08 ms per substep at dt x1, mode decay rate -24.0 % against -22.9 % | `g4_coarse_dt1`, `decay__g4_decay_nu1e-3_dx0.0025` |

## Remaining limitations (failing acceptance criteria at 76f37038)

The goal is not met. These are the acceptance criteria that fail on the committed revision, with the best values of the
configurations measured since the final campaign (FluidBench `report/sph_goal.md`, section "Acceptance by case and
configuration"; tags `v6w`: wall shear consistency, `v6wp`: with the drift correction of the positions as well, `wi` /
`wip`: with the isotropic pressure and density in addition, findings 36-44). Two families of configurations bound what
the solver reaches:

- With the symmetric pressure pairs (`v6w`, `v6wp`) the solver conserves energy and momentum (V6, the V10 stirring
  balance within 2.1 %, water at rest decaying from 1.4e-4 J) and passes V3, V5, V6, V7 (bulk, `v6w`), V8 at dx 2.5 mm
  (`v6wp`), V10 (held plate within 4.9 % with `v6w`) and V12, but the pressure pairs carry a shear stress in proportion
  to the pressure (finding 11): V9 settles 50-66 % slow, the V2 torque is 5-38 % off, and the V1 profile 9-32 %.
- With the isotropic pressure (`wi`, `wip`) the friction is gone (V2 and V9 pass, the V1 profile within 8 %), but the
  corrected pairs do work on the fluid (V10 stirring balance -2.5 to -4.9 times the work, water at rest at 1.2-1.6e-3
  J), and V5, V10 and V11 fail.

No variant tried against the friction keeps the energy (findings 31, 33, 40, 41, 43).

- V1 Stokes first problem: no configuration meets the torque bar of 15 % (1.23-1.91 times the reference with `v6w`,
  0.56-0.86 with `wi`); the profile passes with `wi` only.
- V2 Taylor-Couette: passes with `wi` / `wip` only.
- V4 mode decay: within 10 % at nu 1e-2, 1e-3 and 1e-4 with `v6wp` only (8.2, 5.9, 9.6 %); at nu 1e-6 the mode decays
  2.0-3.2 times faster than the analytic rate in every configuration (finding 39).
- V7 hydrostatics: the cup load matches the weight within 1e-4 and the bulk pressure follows rho g (H - z) within 3.7 %
  at dx 2.5 mm with the virial readout (finding 42; 5.6 % at 3.5 mm); the layer on the floor reads 24 % high and the one
  below the top layer 8 % low, so the bar on every layer below the top one fails.
- V8: the light box passes at dx 2.5 mm with `v6wp` and `wi` (6.1 %, 5.5 %), not with `v6w` (18.5 %); at dx 3.5 mm no
  configuration is within 10 %.
- V9 settling sphere: passes with `wi` / `wip` (0.3-12 %), fails with the symmetric pairs (50-66 %); the run of `wi` at
  dx 1.75 mm ends in NaN in the rigid solver.
- V10: the held plate and the stirring balance pass with `v6w` and fail with the isotropic pressure.
- V11 thin-wall cup: with the liquid sampled clear of the walls, the drift correction and a density tolerance of 0.02 %,
  no particle crosses at R0 or R1, from the lattice or settled (findings 47-49); the largest speed of the settled liquid
  is 0.062 m/s at R0 and 0.106 m/s at R1, above the 0.05 m/s bar, and a lattice is not at rest in a round cup.
- V12: passes with `v6wp` on both GPUs; `v6w` failed its envelope once on the RTX PRO 6000 (2.5e-4 against two replays
  of 2.4e-4).
- V13 FluidBench: S1 is not usable in any configuration (spin-down time 54-86 % short, finding 21 and the V4 dissipation
  at nu 1e-6); S2 is usable at R2 with `v6w`; S3 misses the mixing index.
- Speed: on S1 R2 (RTX PRO 6000, all solves converged) the RTF is 0.039 at the FluidBench time step and 0.142 at dt x8
  for the default configuration (section "Performance (M5)"), against the 0.5 target. The Stage A velocity baseline on
  thick walls runs at 0.282 at dt x8 (0.408 at R1; findings 55, 57-68); dt x12 is unstable, and a floating link fails at
  dt x4. SPH_SOLVER_DESIGN.md section 10 calls S1 R2 accuracy-qualified when V1-V8 pass at that resolution; V1, V2 (or
  V5) and V4 fail in every configuration, so there is no accuracy-qualified speed to report.
