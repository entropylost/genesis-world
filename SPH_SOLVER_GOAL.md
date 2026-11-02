# DFSPH solver development goal

Branch: `fluidbench/sph-solver`
Repository: https://github.com/Roushelfy/Genesis
Owner requirements confirmed: 2026-10-27
Starting revision: `92c783279a9d6d8882bf06dce540c07fdf2a38a6`
Status: implementation in progress. Current delivery scope and acceptance are defined in the dated update below.

## Objective

Deliver a complete, physically consistent, stable and efficient divergence-free smoothed particle hydrodynamics (DFSPH)
solver with implicit liquid viscosity and strong two-way coupling to Genesis rigid links. Optimize single-environment
latency and batched throughput while preserving the accepted physical model and numerical accuracy.

Implicit viscosity is a required part of the final implementation. Pressure and viscous stresses both exchange linear and
angular momentum with rigid links. A diagnostic implementation that applies fluid impulses after an otherwise independent
rigid step is an intermediate milestone. Final acceptance requires the coupled fluid and rigid residuals to converge.

## Current scope and staged acceptance (owner update, 2026-10-30)

The owner prioritizes a useful, stable and fast thick-wall solver and permits moderate engineering accuracy targets.
This section governs the current delivery stage wherever the original full research acceptance below is stricter.
The original criteria and historical pass/fail results remain visible for comparison.

### Stage A: thick-wall engineering solver

- Use thick watertight containers and obstacles. Defer V11 thin-wall penetration and thin-wall/coarse-grid geometry fixes
  from the current critical path. Reopen them after the thick-wall physics and performance campaign.
- Keep the inner fluid domain, liquid mass and prescribed motion fixed when thickening a container outward. Record the
  wall thickness relative to particle spacing and kernel support. Choose and validate a sufficient thickness for each
  resolution rather than assuming thickness alone guarantees stable boundaries. Record changed inertia for dynamic links.
- Keep implicit interior and wall viscosity, two-way pressure and viscous coupling, joint/contact feedback, fluid load
  readout and state/reset support as required features.
- Use symmetric pressure with wall shear consistency and `density_correction="velocity"` as the primary thick-wall
  performance baseline. This keeps the standard DFSPH density correction and disables the extra position-level drift
  solve. Preserve implicit viscosity, two-way coupling and the dissipative wall clearance guard.
- Compare `density_correction="position"` separately on the same committed revision, scene, initialization, time step,
  material properties and tolerances. Report its density/volume error, rest noise, loads, damping, energy, iterations,
  memory and end-to-end cost. Select it only when a measured robustness or accuracy gain justifies the cost.
- Keep wall clearance recovery distinct from density drift correction. Restore state and validate coupling and
  conservation for each variant. Adopt a final option set from unified tests rather than different snapshots.
- Keep force and torque balance, momentum conservation, dissipative behavior, absence of non-finite or growing numerical
  instabilities, and converged accepted samples as required checks. Retain the 5% driven energy-balance and load-readout
  bars and the 1e-4 per second closed-system momentum bar.
- As provisional engineering targets, use 20% for velocity profiles, bulk pressure, draft and viscous decay observables,
  and 30% for transient wall torque or drag. Publish the original tighter verdict beside this staged verdict. These
  bands apply only to cases with a valid, resolved reference and never override conservation or energy checks.
- First qualify the thick-wall cases at nu = 1e-4 to 1e-2 m^2/s. Continue measuring water at nu around 1e-6, report its
  excess numerical damping and its S1/S3 errors, and keep accurate water dynamics as the next research stage. Changing
  a water case to a more viscous liquid creates a separate experiment.
- Allow a documented settling procedure and wall-clear initial sampling. Measure rest-state RMS velocity and energy,
  distinguish isolated particle maxima from sustained growth, and report the original initialization as a control.
  The original thin-wall 0.05 m/s maximum-velocity gate is deferred with V11.
- Run full-horizon stirring, floating/free-link interaction, a constrained link and pouring in the thick-wall geometry.
  Preserve the original FluidBench geometry results. Use separately converged references for materially changed cases.
- Profile and optimize this complete feature set now. Report the accuracy-speed tradeoff over time steps and tolerances.
  RTF >= 0.5 remains the performance target and RTF >= 1 the stretch target. Report engineering usability and measured
  speed separately while that target remains unmet.

### Stage B: higher fidelity and broader geometry

Retain the original research goals for accurate water spin-down and mixing, tighter wall stress and pressure accuracy,
thin-wall handling and broader viscosity/resolution coverage. Stage B failures are documented limitations of Stage A,
rather than reasons to spend the current optimization budget on thin-wall repair.

### Initial Stage A work queue (historical; next queue below)

1. Freeze the thick-wall candidate with position-level drift correction disabled and its public example/configuration.
2. Replay the coupled momentum, energy, load, state and viscosity cases on it. Include a free dynamic link and joint/contact
   feedback, and confirm every accepted sample's pressure, viscosity, drift and constraint residuals.
3. Run thick-wall S1/S2 and the representative dynamic-link cases through the full horizon. Record rest noise, leakage,
   volume, loads, damping and reference errors under both the original and staged targets.
4. Profile that same configuration and pursue the remaining measured acceleration opportunities. Rank them by end-to-end
   benefit. Benchmark single environments and independent batches at matched feature sets.
5. Record Stage A delivery and its limitations before returning to the stricter Stage B research cases.

## Next stage: qualification, coupling robustness and measured acceleration (owner update, 2026-10-31)

This update defines the next work queue. Stage A is a delivered engineering baseline with known failures, not full
acceptance of the original objective. Keep thick walls, implicit interior and wall viscosity, strong two-way rigid
coupling and `density_correction="velocity"` as the baseline. Thin-wall repair and accurate water dynamics remain
deferred; pressure friction in the engineering viscosity range, dynamic-link robustness and state correctness are
active work in this stage. The staged accuracy bands of 2026-10-30 remain in force.

### N1: qualify a clean, committed baseline

- Freeze the current solver code (cc03c40b, recorded by c00833b2), the FluidBench revision, public configuration,
  commands, references and hardware. Use immutable worktrees throughout compilation and execution. Record exact SHAs,
  both repositories' dirty flags, initialization, geometry, time step and every solver tolerance.
- Re-run the representative viscosity, momentum, hydrostatics, floating, held-joint/contact, stirring-energy,
  thick-wall rest and state cases together on that revision. Include full-horizon pouring and free dynamic-link
  scenes. Preserve every existing failure and capped solve in the new report.
- Reproduce the latest speed ladder on clean code at R1/R2/R3 and batches of 1/4/16 where feasible. The current S1
  dt x8 planning baseline is RTF 0.408/0.282/0.126, not a water-accuracy pass. Keep older snapshots for comparison.
- A development snapshot (`genesis_dirty=true` or modified benchmark code) can motivate a change but cannot establish
  final qualification. Re-run adopted changes on their clean committed revision. Mark each case as pass, fail, invalid
  or not run; convergence is required for both accuracy and accepted timing samples.

### N2: make dynamic-link operating limits reliable

- Trace the first unconverged stage and first non-finite state in the thick-wall free-link failures (T07a at dt x4,
  T08a and T08c controls). Separate geometry/advection limits, pressure-viscosity coupling, rigid mobility and
  joint/contact feedback; record normalized residual histories and impulse accounting.
- Test light links, large fluid-to-link inertia ratios, a constrained link and contact feedback. Sweep dt x1/x2/x4/x8
  with fixed physical parameters, and compare loads and trajectories against a refined, converged time-step reference.
- Correct the responsible equations, stage order or solver convergence before raising a supported time step. An
  explicit smaller-step operating range is acceptable; dt x8 from prescribed stirring is not a general rigid-coupling
  guarantee. Raising an iteration cap is a measured cost/robustness tradeoff, not a correction of a wrong settling rate.
- Reject non-finite or unconverged results through the existing error path. Preserve action/reaction, the 1e-4 per
  second closed-system momentum bar and the 5% energy/readout bars.

### N3: establish state and selected-reset correctness

- Audit all persistent warm starts, active/convergence flags, coupling guesses, impulses and cache invalidation.
  Exercise snapshot/restart, repeated reset and selected-environment reset in independent batches, including the
  first replayed step and optional micropolar state where enabled.
- Compare repeated ordinary restarts with full and selected resets using particle RMS, quantiles and maxima, plus
  density, energy, loads and trajectories over the declared horizon. Use deterministic reductions as a diagnostic
  where practical to separate missing state from amplification of floating-point round-off.
- Keep the original V12 envelope verdict. A replacement statistical criterion requires a written numerical rationale
  and thresholds frozen before new validation runs, with both verdicts reported. Increasing a bound after observing a
  failed run does not close the issue.

### N4: reduce pressure-induced shear and qualify physical viscosity

- Isolate the pressure contribution to shear stress and drag in V1 Stokes flow, V2 Couette and V9 settling, including
  its dependence on hydrostatic pressure, particle disorder, resolution and viscosity. Include T08c as an application
  check: a converged but nearly stationary settling sphere is not an accuracy pass.
- Write the density constraint, force operator, rigid response and discrete work balance for each proposed correction.
  Measure interior and wall contributions separately. A pressure-model reformulation is allowed when justified by
  these measurements; no single reformulation or removal of the error is presumed in advance.
- Start at nu = 1e-4 to 1e-2 m^2/s. Target the existing 20% profile/decay and 30% transient-load bands, including the
  currently failing nu = 1e-4 decay case. Retain each analytic case's declared rate/reference criterion; do not reclassify
  a settling-rate failure as a passing transient-load measurement.
- Reject a correction that introduces growing rest energy, breaks the 5% driven balance, loses momentum balance,
  hides errors in the pressure readout or weakens the coupled residual checks. Re-run V3/V5/V6/V7/V8/V10 and pouring
  after adopting a model change. Keep accurate water and thin walls reported as deferred research.
- If the current formulation cannot meet a target, publish the measured limitation and the next justified experiment.
  Do not declare a fundamental DFSPH error floor or silently loosen the owner targets.

### N5: accelerate the qualified feature set

- Profile after the correctness changes. Rank candidates by end-to-end benefit; the current profile points to roughly
  560-610 GPU tasks per substep and their 6-8 us latency floor at R1/R2. Prioritize reducing tasks or justified
  iterations while retaining the global dependencies and converged coupled solution.
- Preserve the useful chunked reductions, shared neighbor loops, independent environment stops and integer wall-gap
  minimum unless new measurements justify a replacement. Revisit CUDA graphs, coupling warm starts or relaxed first
  solves only with new evidence: prior graph trials gave no material gain and fewer first-pass iterations raised rest
  noise or broke the energy balance.
- Measure static/prescribed scenes separately from free or constrained links. Report warmed solver-only and output
  timing, build/compile time, peak memory, residuals, rest noise and three or more timing repeats. For batches report
  total environment throughput, per-environment RTF, spread and heterogeneous trajectories.
- Keep RTF >= 0.5 at R2 as the target and >= 1 as the stretch target. The S1 water timing remains a performance
  diagnostic until its accuracy qualifies. Add an explicitly identified viscous stirring case with a converged
  reference; changing the water viscosity creates a separate case and cannot turn the original S1 verdict into a pass.
- Compare candidates at matched physics, geometry, resolution, time steps and tolerances. A benefit measured only at
  a time step that fails dynamic-link validation cannot establish dynamic-coupling acceleration.

### N6: ablate every accuracy, conservation and boundary correction (owner update, 2026-11-02)

Measure the accuracy benefit, runtime cost and side effects of every correction discussed below. The purpose is to
identify the simplest configuration supported by evidence, not to assume that more corrections are better or that all
can be removed. This is an active next-stage goal alongside N1-N5. Keep the current engineering baseline unchanged
until a replacement passes the matched validation.

#### Components and controls

| Id | Component | Required comparison |
| --- | --- | --- |
| A1 | Directional consistency of the interior viscosity (15 coefficients per particle) | current directional correction versus the original uncorrected pair coefficient and an isotropic scalar correction; retain symmetric nonnegative pair weights in proposed delivery candidates |
| A2 | Ghost-fluid columns for wall viscosity | current 8-point columns versus the original single boundary-particle pair model; compare 1/2/4/8-point quadrature with the same column definition to separate the boundary model from integration cost |
| A3 | Wall shear consistency scaling | enabled versus disabled on identical geometry; report the distribution of the scale, how often its [0.25, 4] bounds activate, corner/free-surface behavior and sensitivity to particle-wall distance |
| A4 | Projection of the net viscosity residual force and torque | projection on/off at matched CG tolerance, plus a tighter-tolerance control without projection; measure the correction magnitude, momentum drift, energy change and extra 6x6 solve/reduction cost |
| A5 | Wall clearance protection and its angular-momentum compensation | full guard versus off; isolate the normal velocity constraint, soft position recovery and the global rotational velocity correction; include position recovery disabled as a control for the need for angular compensation |
| A6 | Isotropic pressure and corrected density (experimental, off in the baseline) | symmetric pressure baseline versus isotropic pressure with corrected density off/on; retain the original force/density definitions and record their work consistency, rest heating and shear benefit |
| A7 | Position-level density drift correction (experimental, off in the baseline) | velocity correction versus position correction at matched configuration; record volume benefit, velocity/rotation changes, added iterations, dynamic-link failures and non-finite onset |

For A1 report regularization, nonnegative clipping and the free-surface blend, including how often they are active.
For A2 report the near-wall distance floor and the column start determined by the nearest fluid gap. For A3 use any
additional parameter sweep to explain sensitivity, not to tune separately to each validation scene.
For A5 quantify position shifts, corrective impulses, the global rotational adjustment, and kinetic and potential
energy changes separately. Its reaction belongs in rigid force/torque accounting. A guard-disabled run that leaks is a
measured failure, not a valid fast replacement; thick walls do not justify assuming the guard is redundant.
For A6 and A7 retain the known negative results as controls and stop divergent experiments at their declared failure
condition. These comparisons do not enable either option by default.

#### Matched protocol and interaction tests

- Freeze clean Genesis and FluidBench revisions, reference solutions, horizons, seeds, initial states, geometry, mass,
  viscosity, time steps, tolerances and iteration caps before the campaign. Keep implicit interior/wall viscosity and
  two-way rigid response in the main comparisons. Add benchmark-only switches where needed and record them explicitly;
  do not change production defaults merely to run an ablation.
- Start with one-component changes from the current baseline. Then test the important dependencies: A1 x A2 x A3 for
  wall/interior consistency, A4 x CG tolerance, A5 position recovery x rotational compensation, A6 x A7, and A7 x A5.
  Use staged screening rather than an indiscriminate full Cartesian product. Confirm promising combinations and any
  simplest viable configuration over the full declared horizons.
- Use a simple documented configuration as a second anchor, with the extra corrections removed where the equations
  still define a valid model. Record why any removal is mathematically undefined or requires a replacement boundary
  model. Include combined removals so that individually small benefits cannot hide an important interaction.
- Separately measure the cost of the coupling structure: converged viscosity-pressure iteration and rigid constraint
  repetitions versus one-pass diagnostic controls. Chebyshev acceleration and warm starts can be solver controls at
  matched residuals. A one-pass or disabled-feedback run is not an accepted simplification unless the original coupled
  residuals, conservation and accuracy criteria pass; keep it distinct from the correction ablations.
- Screen on analytic viscosity/shear, wall and conservation cases, then confirm on V1-V10 as applicable, thick-wall
  rest, V12, full-horizon pouring, stirring and a free/constraint-coupled link. Include R1/R2/R3 and nu = 1e-4 to
  1e-2 m^2/s for final candidates; low-viscosity water remains a diagnostic. Thin-wall repair stays deferred.
  Keep both settled and original initialization controls where startup noise affects the conclusion.
- Report every pressure, viscosity, coupling, drift and constraint residual, iteration count, cap exhaustion, slot
  overflow and failure time. A non-finite, leaking or unconverged run remains visible but cannot supply an accepted
  speedup or accuracy result. Separate static/prescribed walls from dynamic links and their supported time steps.

#### Measurements and delivery decision

For every component and selected combination publish:

- Accuracy: profile L2 and maximum error, effective viscosity/modal and rotational decay, slip, integrated force and
  torque, hydrostatic pressure/load, floating draft and settling rate, volume drift and leakage against the fixed
  analytic or converged references.
- Side effects: rest RMS/quantile/maximum speed, kinetic and potential energy, driven work/dissipation balance,
  linear/angular momentum drift, correction magnitude and activation frequency, stiffness/convergence changes,
  restart/selected-reset sensitivity, and the time and stage of instability.
- Cost: uninstrumented warmed step time and RTF (at least three repeats with spread), instrumented stage/kernel
  breakdown including correction setup and global reductions, GPU task count, iterations, build/compile time and
  peak memory. Compare one environment and independent batches; report both per-environment and total throughput.
- Attribution: implement or verify separate pressure, viscous and clearance force/torque/impulse readouts and their
  sum against the total. The current test-side optional pressure getter is insufficient while the committed solver
  lacks it; empty or unavailable pressure data cannot count as a successful pressure/viscosity separation.

Provide a compact table with baseline, each ablation, important combinations, absolute errors, accuracy changes,
time and memory changes, side effects, convergence and a recommendation: retain, simplify, remove, or inconclusive.
Use the existing staged accuracy and conservation/energy bars, and keep the original stricter verdicts beside them.
Select any default change from this evidence and re-run N1-N3 on the resulting clean commit. If no simpler variant
qualifies, retain the baseline and publish the measured reason. Store commands and receipts in FluidBench and sync
adopted code, tests, examples and documentation to the collaborator repository under the existing requirement.

### Completion and delivery

Track N1-N6 separately as pending, implemented, validated or measured, with exact commits, commands and receipts.
Publish the accepted operating range, original and staged verdicts, remaining failures and reproducible public examples.
Retain historical negative results. Sync every coherent milestone, including goals/status, code, tests and examples,
to `entropylost/genesis-world:main` under the existing owner requirement; verify both heads or the integrated diff.
Do not label all goals complete while correctness failures or the performance target remain open.

## Required behavior

### Implicit viscosity

- Solve Newtonian viscosity implicitly in both the fluid interior and at coupled walls. Support spatially varying material
  viscosity with a consistent interface treatment.
- Define coefficient units explicitly. Use one canonical internal convention and convert between kinematic viscosity
  `nu [m^2/s]` and dynamic viscosity `mu [Pa s]` through `mu = rho * nu`.
- Derive the operator from a dissipative, symmetric stress discretization. Establish its symmetry, sign, null modes and
  boundary conditions before selecting a conjugate-gradient solver. Use a solver appropriate to the actual operator.
- Include stationary and moving no-slip walls. Additional slip behavior requires a defined physical boundary condition.
- Apply equal-and-opposite wall forces and moments to the corresponding rigid link. Account for shear-driven spin-up,
  drag, stirring torque and spin-down.
- Remove the explicit viscosity time-step restriction from the final viscosity update. Advection, changing geometry,
  free surfaces, capillarity, contact and solver convergence still define a documented usable time-step range.
- Treat micropolar rotation as a separate optional model. Validate the Newtonian baseline with micropolar disabled, then
  test its integration separately. Angular smoothing and artificial damping are separate from physical liquid viscosity.

### Strong two-way rigid coupling

- Support fixed links, prescribed kinematic links, free dynamic links and articulated links with joint/contact constraints.
- Include link translational and angular response in the pressure and implicit-viscosity solve. Articulated response uses
  the constrained generalized mass response rather than independent free-link inverse masses.
- Advance pressure, viscosity and rigid response consistently at the same substep. Define the stage order and the state
  each stage reads. Track any outer iteration with residual-based convergence.
- Account for both DFSPH divergence and density corrections in the final exchanged impulse. Accumulate the net applied
  correction once. Internal iteration updates must not create repeated physical impulse application.
- Extend the current boundary-pressure-acceleration-zero assumption and its iteration coefficients for moving dynamic
  boundaries. Recompute a preconditioner whenever its numerical assumptions require it.
- Provide public batched force, torque and impulse readout. Define world/local frames, reference point, signs, units and
  substep/control-step averaging. Distinguish fluid loads from gravity, actuators and rigid contact.
- Maintain prescribed trajectories for kinematic links and report their actuator-equivalent fluid load. Dynamic links
  respond to the exchanged load through their actual mass, inertia and constraints.
- Document the pressure/free-surface model. Verify hydrostatic pressure, buoyancy and compression control without a
  separate hand-tuned buoyancy force.

### Coupled solve

Write the discrete equations and their residuals before implementation. A monolithic formulation or a converged block
iteration is acceptable. A single uncorrected viscosity-then-pressure or fluid-then-rigid pass does not establish the final
strong-coupling requirement.

A useful starting point is a joint velocity space containing particle velocities and rigid generalized velocities, a
mass operator, an implicit viscous dissipation operator, and the DFSPH pressure and rigid constraint operators. A pressure
Schur complement must use the velocity response of the selected mass-plus-viscosity system. If block iterations replace
that response, demonstrate convergence of all coupled residuals and the force/torque balance.

Use backward Euler as the initial temporal reference unless the derivation justifies another scheme. Measure temporal
and spatial convergence. Stability alone does not establish accurate damping, spin-down or load prediction.

### State and failure handling

- Save, restore and selectively reset all persistent state, including pressure warm starts, angular velocity when
  micropolar is enabled, iterative-solver warm starts, coupling impulses and cache validity.
- Keep environments independent, including partial reset, active masks, heterogeneous material properties and loads.
- Use the existing solver error mechanism for non-finite state and invalid solves. Record iteration-cap exhaustion and
  residuals explicitly. An unconverged state cannot count as a successful accuracy or performance sample.
- Follow `CLAUDE.md`, `CODING_GUIDELINES.md` and the repository contribution instructions. Backend-specific acceleration
  implements the same physical model and numerical algorithm as the portable path.

## Performance work

Establish an instrumented profile and an uninstrumented timing baseline at the starting revision. Keep the original
one-way explicit-viscosity results as historical evidence. Measure the cost of the new physical features separately, then
compare optimizations against a functionally identical coupled implicit-viscosity baseline.

Prioritize the following measured candidates:

1. Cache boundary geometry per fluid-particle/link pair. Include gradient and moment aggregates needed for force/torque
   exchange. Freeze geometry only over iterations where that approximation is part of the derived scheme. Update the
   rigid velocity response during coupled iterations. Invalidate geometry through the existing change mechanisms.
2. Replace contended slot allocation in spatial hashing with a measured parallel scan or occupied-slot method. Reduce
   unnecessary full-slot clearing. Validate neighbor completeness for negative coordinates, hashing collisions and
   multiple environments.
3. Remove per-iteration host scalar readback from pressure, viscosity and outer coupling convergence where backend support
   permits. Retain the required global dependency barriers between acceleration and residual evaluation.
4. Separate static and moving boundary indexing. Reuse fixed-link geometry and rebuild the moving portion as needed.
5. Evaluate cached neighbor indices and kernel gradients against recomputation. Report memory use, cache build time and
   iteration cost. Reuse geometry across pressure and viscosity operators where their mathematical definitions allow it.
6. Reduce scratch-field reorder/copy traffic. Preserve public particle ordering, persistent warm starts, rendering and
   state restoration.
7. Reuse preconditioner structure and allocations where valid. Report iteration counts and refresh costs. Allocate exact
   known sizes and follow the repository's Quadrants memory-lifetime requirements.
8. Use block reductions and link-grouped force accumulation where profiling shows contention. Verify signed linear and
   angular impulse balance.
9. Allow environments to stop work independently when their coupled residuals pass. Keep inactive/reset environments
   isolated. Benchmark independent trajectories as well as identical replicated trajectories.
10. Measure a prescribed-boundary execution path for scenes without dynamic rigid response. Use the full rigid solver
    whenever joints, dynamics or contact require it. Report this scene specialization separately from general coupling.

Choose launch and execution configurations from device and workload properties. Check generality across particle counts,
boundary fractions, viscosity, link count and environment count. A single stirring scene cannot establish general speedup.

## Validation and acceptance

Freeze the case parameters, references, horizons and justified numerical tolerances before optimization comparisons.
Existing FluidBench tolerances remain authoritative for its cases. For additional tests, derive tolerances from analytic
solutions, reference convergence and numerical residuals. Preserve failure cases in the report.

| Requirement | Evidence |
| --- | --- |
| Correct implicit viscosity | Couette and Poiseuille profiles, analytic viscous mode decay, and time/grid refinement across low and high viscosity |
| Wall viscosity | Moving-wall shear, no-slip error, integrated wall force and torque, and viscous dissipation balance |
| Momentum exchange | Closed isolated fluid/link cases with linear and angular momentum balance at the solver tolerance |
| Rigid response | Hydrostatic floating draft, rising/settling links where a valid reference exists, and light-link/high-fluid-inertia cases |
| Strong coupling | Pressure, viscosity, rigid and outer residual histories, plus temporal refinement for loads and trajectories |
| Articulation | A constrained joint with fluid loading and a robot-driven stirring case with joint torque readout |
| Boundary robustness | Thin-wall and coarse-resolution cup cases, geometry offsets/orientations, moving walls and leakage measurements |
| FluidBench dynamics | S1 stirring and spin-down, S2 pouring, and applicable S3 mixing cases against the frozen references |
| State correctness | Snapshot/restart, repeated reset and selected-environment reset including warm starts and micropolar state |
| Performance generality | Several particle/link counts, viscosity values and boundary fractions, single environment and independent batched scenes |

A fixed or kinematic wall exchanges momentum and work with an external support/actuator. Include those exchanges in its
balance. For a closed stationary setup, viscosity dissipates kinetic energy. For driven cases, compare energy change with
external work and viscous dissipation. A physical instability must be identified separately from numerical blow-up.

Keep concise analytic and algebraic checks in the unit suite. Put long-horizon stability, convergence and performance
sweeps in benchmark campaigns, following the repository's test-step budget.

Report full S1 horizons, not only short build/run smoke tests. Preserve the declared water properties. Increasing
viscosity to suppress noise, changing particle volume to hide drift, disabling divergence correction, loosening solver
tolerances or reducing spatial resolution changes the experiment and requires a separate accuracy tradeoff report.

## Performance measurements and targets

Measure warmed solver-only time, output/readback time, build/compile time, peak memory, iterations and residuals. Report
pressure, viscosity, outer coupling, neighbor/hash work, rigid work, reductions, synchronization and output separately.
Use uninstrumented repeated timings for speed claims and include dispersion.

Real-time factor (RTF) is simulated seconds divided by wall seconds. The initial single-environment target is RTF >= 0.5
on the declared representative GPU and accuracy-qualified S1 configuration. RTF >= 1 is a stretch target. Record achieved
speed if these targets remain unmet. Test resolution ladders rather than selecting an unstable coarse configuration.

For batching, report environment-transitions/s, per-environment RTF and memory at feasible batch sizes. Compare
optimizations at matched physical parameters, resolution, tolerances and feature set. Record representative hardware and
all execution-path choices. Define numerical acceptance before low-precision experiments.

The earlier FluidBench profile reports approximately 52% pressure solving and 28% rigid work. Its RTF around 0.069 is a
historical planning reference, not the implicit-viscosity/two-way baseline. Performance targets require new measurements.

## Milestones

All items start pending. Each completed milestone needs a commit, runnable command, configuration and evidence path.

- [ ] M0: Freeze equations, units, baseline cases, numerical acceptance criteria and performance measurement boundaries.
- [ ] M1: Establish dissipative implicit interior/wall viscosity with analytic validation and force/torque accounting.
- [ ] M2: Complete pressure/viscous reaction readout and the rigid stage integration. A split prototype is development
      evidence only.
- [ ] M3: Complete strong pressure-viscosity-rigid coupling, articulated response and residual-based convergence.
- [ ] M4: Pass thin-wall, coarse-resolution, light-link, high-viscosity and reset/state robustness campaigns.
- [ ] M5: Run optimization ablations at a matched full-feature baseline, including single and multiple environments.
- [ ] M6: Re-run FluidBench, publish accuracy/performance evidence, document public APIs and record remaining limitations.

Completion means all required physics, stability and optimization evidence is present. A stable animation, an explicit
fallback, a pressure-only reaction force, a viscosity-only solve or a fast one-way configuration is insufficient.

## Synchronization with the collaborator repository (owner requirement, 2026-10-30)

Development branch: https://github.com/Roushelfy/Genesis/tree/fluidbench/sph-solver
Collaborator repository: https://github.com/entropylost/genesis-world
Default integration target: `entropylost/genesis-world:main`, which contains the DFSPH starting implementation.
Use a collaborator-designated DFSPH branch instead when that destination is explicitly recorded. The separate
`ipbf-sph-solver` branch is a different work stream and is not the default destination.

- Synchronize each coherent solver feature, correctness fix, performance change and delivery milestone with the
  collaborator repository. Include the associated tests, public examples, option/readout documentation, design,
  development goal and implementation status. Keep large raw evidence in FluidBench and link its exact revisions.
- Before synchronization, fetch the current destination and preserve changes made in either repository. Resolve shared
  solver changes by merging or porting the relevant commits. Preserve unrelated collaborator work.
- Deliver through a pull request from `Roushelfy:fluidbench/sph-solver` to the recorded destination branch, or update the
  existing synchronization pull request. The owner-authorized fast-forward workflow already recorded in the status is
  also acceptable when the destination is an ancestor of the source. Never force-push over collaborator changes.
  Record the source head, destination branch/head, included commits, validation and pull request URL when applicable.
- Track submitted and integrated states separately. A local push alone establishes development progress. Submitted
  means the destination pull request contains the intended changes. Integrated means those changes are present in the
  destination branch, including equivalent commits when squashed or cherry-picked.
- Record synchronization status in `SPH_SOLVER_STATUS.md` at every milestone and identify pending changes or conflicts.
  Keep implementation and validation moving while repository permissions or destination integration remain pending.
- Re-run the affected checks after conflict resolution and verify the published destination or pull request diff.
  Synchronization preserves the same physical model, defaults and numerical acceptance as the validated source.
- Verified before the 2026-10-31 goal update: source and destination both hold c00833b277361c63d78453485868effd238eeca1.
  The existing solver work is integrated; synchronize subsequent milestones and verify their destination separately.

## Evidence locations

Keep derivations, implementation status and compact campaign receipts alongside this goal or in the repository's
established documentation layout. Keep tests and examples in their existing directories. Integrate benchmark records and
analysis with [FluidBench](https://github.com/Roushelfy/fluidbench/tree/bench/auto-research), preserving source SHAs and
physical parameters. Avoid duplicating large raw trajectories in this repository.

Record an optimization as proposed, implemented, validated or measured, with its supporting artifact. Keep the current
implementation status separate from this target specification.
