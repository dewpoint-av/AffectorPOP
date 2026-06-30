// Particle Affector POP — CUDA kernel. Applies a force to each particle's velocity
// (v += force*dt), optionally integrating position (P += v*dt) for a feedback-loop stepper.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_functions.h>
#include <cmath>
#include "AffectorParams.h"

__device__ __forceinline__ float3 operator+(const float3&a,const float3&b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ __forceinline__ float3 operator-(const float3&a,const float3&b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ __forceinline__ float3 operator*(const float3&a,float s){return make_float3(a.x*s,a.y*s,a.z*s);}
__device__ __forceinline__ float  dot3(const float3&a,const float3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ __forceinline__ float3 cross3(const float3&a,const float3&b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ __forceinline__ float  len3(const float3&a){return sqrtf(dot3(a,a));}
__device__ __forceinline__ float3 norm3(const float3&a){float l=len3(a);return l>1e-8f?a*(1.0f/l):make_float3(0,0,0);}

// hash value-noise (scalar)
__device__ __forceinline__ float hashf(float x,float y,float z){float n=sinf(x*12.9898f+y*78.233f+z*37.719f)*43758.5453f;return n-floorf(n);}
__device__ float vnoise(float3 p){
    float3 i=make_float3(floorf(p.x),floorf(p.y),floorf(p.z));
    float3 f=make_float3(p.x-i.x,p.y-i.y,p.z-i.z);
    float3 u=make_float3(f.x*f.x*(3-2*f.x),f.y*f.y*(3-2*f.y),f.z*f.z*(3-2*f.z));
    float n000=hashf(i.x,i.y,i.z),n100=hashf(i.x+1,i.y,i.z),n010=hashf(i.x,i.y+1,i.z),n110=hashf(i.x+1,i.y+1,i.z);
    float n001=hashf(i.x,i.y,i.z+1),n101=hashf(i.x+1,i.y,i.z+1),n011=hashf(i.x,i.y+1,i.z+1),n111=hashf(i.x+1,i.y+1,i.z+1);
    float x00=n000+(n100-n000)*u.x,x10=n010+(n110-n010)*u.x,x01=n001+(n101-n001)*u.x,x11=n011+(n111-n011)*u.x;
    float y0=x00+(x10-x00)*u.y,y1=x01+(x11-x01)*u.y;
    return y0+(y1-y0)*u.z;
}
// divergence-free curl noise from a 3-component potential (finite-difference curl)
__device__ float3 curlNoise(float3 p){
    const float e=0.35f;
    float3 dx=make_float3(e,0,0),dy=make_float3(0,e,0),dz=make_float3(0,0,e);
    // potential samples (offset seeds keep the 3 channels independent)
    float3 o1=make_float3(0,0,0),o2=make_float3(31.4f,17.2f,9.6f),o3=make_float3(-8.1f,52.3f,23.7f);
    float p1y=vnoise(p+dy+o1)-vnoise(p-dy+o1), p1z=vnoise(p+dz+o1)-vnoise(p-dz+o1);
    float p2x=vnoise(p+dx+o2)-vnoise(p-dx+o2), p2z=vnoise(p+dz+o2)-vnoise(p-dz+o2);
    float p3x=vnoise(p+dx+o3)-vnoise(p-dx+o3), p3y=vnoise(p+dy+o3)-vnoise(p-dy+o3);
    // curl = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy)
    float inv=1.0f/(2.0f*e);
    return make_float3((p3y - p2z)*inv, (p1z - p3x)*inv, (p2x - p1y)*inv);
}

__global__ void affectorKernel(const float3* Pin, float3* Pout, const float3* Vin, float3* Vout,
                               const float3* fieldDir, const float* field,
                               unsigned int n, AffectorParams ap)
{
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float3 P=Pin[i];
    float3 V=Vin?Vin[i]:make_float3(0,0,0);
    float3 C=make_float3(ap.cx,ap.cy,ap.cz);
    float3 A=make_float3(ap.ax,ap.ay,ap.az);
    float3 F=make_float3(0,0,0);

    switch(ap.type){
        case 0: { // Turbulence (curl noise)
            F = curlNoise(make_float3(P.x*ap.freq+ap.animT, P.y*ap.freq, P.z*ap.freq)) * ap.strength;
            break; }
        case 1: { // Vortex around axis A through C
            float3 r=P-C; float3 axis=norm3(A);
            float3 rp=r - axis*dot3(r,axis);          // radial component
            F = cross3(axis, rp) * ap.strength;        // tangential
            break; }
        case 2: { // Attractor (+ pull) / Repulsor (- )
            float3 d=C-P; float dist=len3(d);
            float fall = (ap.radius>0.0f) ? fmaxf(0.0f, 1.0f-dist/ap.radius) : 1.0f/fmaxf(dist*dist,1e-3f);
            F = norm3(d) * (ap.strength*fall);
            break; }
        case 3: { // Drag
            F = V * (-ap.strength);
            break; }
        case 4: { // Wind (uniform)
            F = A * ap.strength;
            break; }
        case 5: { // Field Force (from upstream Field POP)
            float w = field?field[i]:1.0f;
            float3 dir = fieldDir?fieldDir[i]:A;       // use fieldDir if present, else the Direction param
            F = dir * (w*ap.strength);
            break; }
    }

    V = V + F*ap.dt;
    if(Vout) Vout[i]=V;
    if(Pout) Pout[i] = ap.integrate ? (P + V*ap.dt) : P;
}

static int divUp(int a,int b){return (a+b-1)/b;}

extern "C" cudaError_t
launchAffector(const void* Pin, void* Pout, const void* Vin, void* Vout,
               const void* fieldDir, const void* field, unsigned int n,
               const AffectorParams& ap, cudaStream_t stream)
{
    if(n==0) return cudaSuccess;
    dim3 b(128),g(divUp((int)n,128));
    affectorKernel<<<g,b,0,stream>>>((const float3*)Pin,(float3*)Pout,(const float3*)Vin,(float3*)Vout,
                                     (const float3*)fieldDir,(const float*)field,n,ap);
    return cudaGetLastError();
}
