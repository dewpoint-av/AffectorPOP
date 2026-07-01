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
    int   fieldMask = 0;            // 1 => multiply the force by the upstream 'field' scalar (spatial mask)
    float damping   = 0.0f;         // per-step velocity loss [0..1] — bounds accumulation (vortex won't spiral out forever)
    float maxSpeed  = 0.0f;         // hard velocity clamp (0 = off) — stability under big param changes
    // interactive point force (mouse/touch): radial pull(+)/push(-) at a driven position, within a radius
    float interactStrength = 0.0f, interactRadius = 0.5f; float ipx=0,ipy=0,ipz=0;
};
