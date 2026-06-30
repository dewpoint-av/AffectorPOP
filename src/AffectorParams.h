// Shared particle-affector parameters (host fills, device reads). Plain POD.
#pragma once

struct AffectorParams
{
    int   type = 0;        // 0 Turbulence, 1 Vortex, 2 Attractor, 3 Drag, 4 Wind, 5 FieldForce
    float cx = 0, cy = 0, cz = 0;   // center (vortex/attractor)
    float ax = 0, ay = 1, az = 0;   // axis (vortex) / direction (wind)
    float strength = 1.0f;          // signed (attractor: + pull, - push)
    float freq   = 0.5f;            // turbulence noise frequency
    float radius = 0.0f;            // attractor falloff radius (0 => 1/d^2)
    float animT  = 0.0f;            // turbulence animation offset
    float dt     = 1.0f / 60.0f;    // timestep
    int   integrate = 0;            // 1 => also integrate P += v*dt (feedback-loop stepper)
};
