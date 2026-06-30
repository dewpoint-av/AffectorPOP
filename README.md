# Particle Affector POP (Dew Affector) — TouchDesigner CUDA Custom POP

Suite item §A5. Applies dynamic **forces** to particle velocities — the Notch-style affector layer.
Distinct from the **Effector** (which modulates static clone transforms); the Affector drives a
particle *simulation*.

## Status — v1 BUILT (2026-06-29)
Compiles + links + exports correctly. NOT yet TD-verified.

- **Op:** `Dewaffector` / "Dew Affector" / icon AFF. 1 input POP (particles with `P`, optional `v`).
- **Force types:** Turbulence (divergence-free **curl noise**), Vortex (around an axis), Attractor /
  Repulsor (signed), Drag (−strength·v), Wind (uniform), **Field Force** (reads upstream `fieldDir`×
  `field` — so a Field POP, or later the fluid's velocity field, drives the particles).
- **Behavior:** `v += force·dt`; with **Integrate Position** on, also `P += v·dt` (a one-node stepper
  for a **Feedback POP** loop). Off = a force field that composes with TD's native Particle POP.
- **Params:** Force Type, Strength (signed), Center, Axis/Direction, Turbulence Frequency + Anim
  Offset, Attractor Radius, Timestep, Integrate Position.
- Reads `P`/`v`/`fieldDir`/`field` as CUDA buffers; outputs `v` (and `P` if integrating), passes the
  rest through (creates `v` if absent).

## Typical use
```
Particle source (POP)  →  [Feedback POP]  →  Dew Affector (Integrate on)  →  loop back
                                              (stack several: Turbulence + Vortex + Drag)
```
Or as a force field: native Particle POP ← Dew Affector (Integrate off) adds to velocity.
**Field Force** type + a Field POP (or the coming fluid velocity field) = field-driven particles.

## Build / deploy
```sh
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release      # -> build/Release/AffectorPOP.dll
```
Copy into `%USERPROFILE%/Documents/Derivative/Plugins/`. Windows-only (CUDA POP).

## Next
- The flagship **GPU fluid/smoke solver** — its velocity field feeds the **Field Force** affector.
- More: collision (vs SDF/geo), per-particle mass/age, vorticity-confinement turbulence.
