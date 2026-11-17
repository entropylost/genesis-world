from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base, SamplerType


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rest_density : float, optional
        The rest density (kg/m³). Default is 1000.0.
    viscosity : float, optional
        Kinematic viscosity of the liquid in m^2/s (water is about 1e-6). Higher values damp the flow and drag on the
        walls more. The default solver integrates a viscosity too high for an explicit integration at the substep time
        step implicitly (see `viscosity_integration`), so any value is stable, but very viscous liquids need more solver
        iterations per substep. Default is 0.005.
    surface_tension : float, optional
        The surface tension of the liquid. Default is 0.01.
    vorticity : float, optional
        Coupling strength between the flow and the micro-rotation of the particles, which only applies when the SPH
        solver models micro-rotation (`micropolar=True`). Higher values inject more small-scale swirl, but too high
        values make the flow noisy. Default is 0.01.
    angular_viscosity : float, optional
        Viscosity of the micro-rotation field, which only applies when the SPH solver models micro-rotation. Higher
        values smooth the micro-rotation between neighbors and keep the swirls coherent, but damp them faster.
        Default is 0.1.
    microinertia : float, optional
        Micro-inertia of the particles, which only applies when the SPH solver models micro-rotation. Lower values let
        the micro-rotation follow the flow faster, but require a smaller time step to stay stable. Default is 2.0.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability with SPH.
    """

    rest_density: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    viscosity: NonNegativeFloat = 0.005
    surface_tension: NonNegativeFloat = 0.01
    vorticity: NonNegativeFloat = 0.01
    angular_viscosity: NonNegativeFloat = 0.1
    microinertia: PositiveFloat = 2.0
    sampler: SamplerType = "regular"
