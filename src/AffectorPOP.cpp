/* Particle Affector POP — implementation. See AffectorPOP.h. */
#include "AffectorPOP.h"
#include "AffectorParams.h"
#include <algorithm>

extern "C" cudaError_t
launchAffector(const void* Pin, void* Pout, const void* Vin, void* Vout,
               const void* fieldDir, const void* field, unsigned int n,
               const AffectorParams& ap, cudaStream_t stream);

// ---------------------------------------------------------------- plugin exports
extern "C"
{
DLLEXPORT void
FillPOPPluginInfo(POP_PluginInfo* info)
{
    if (!info->setAPIVersion(POPCPlusPlusAPIVersion))
        return;
    info->customOPInfo.opType->setString("Dewaffector");
    info->customOPInfo.opLabel->setString("Dew Affector");
    info->customOPInfo.opIcon->setString("AFF");
    info->customOPInfo.authorName->setString("Dewpoint");
    info->customOPInfo.authorEmail->setString("anthony@dewpoint.live");
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;
}
DLLEXPORT POP_CPlusPlusBase*
CreatePOPInstance(const OP_NodeInfo* info, POP_Context* context) { return new AffectorPOP(info, context); }
DLLEXPORT void
DestroyPOPInstance(POP_CPlusPlusBase* instance, POP_Context*) { delete (AffectorPOP*)instance; }
};

// ---------------------------------------------------------------- ctor / dtor
AffectorPOP::AffectorPOP(const OP_NodeInfo* info, POP_Context* context) :
    myNodeInfo(info), myExecuteCount(0), myStream(0),
    myContext(context), myError(nullptr), myLastNumPoints(0)
{
    cudaStreamCreate(&myStream);
}
AffectorPOP::~AffectorPOP()
{
    if (myStateP) cudaFree(myStateP);
    if (myStateV) cudaFree(myStateV);
    if (myStream) cudaStreamDestroy(myStream);
}

void AffectorPOP::getGeneralInfo(POP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
    // Self-Simulate must advance every frame even when the input is static — otherwise a still
    // point source never re-cooks and the sim looks frozen.
    ginfo->cookEveryFrame = inputs->getParInt("Selfsim") ? true : false;
}

void AffectorPOP::pulsePressed(const char* name, void*)
{
    if (name && std::string(name) == "Reset") myReseed = true;   // re-seed state from the input
}

std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>
AffectorPOP::getAllAttributes(POP_AttributeClass c, const OP_POPInput* input)
{
    std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>> res;
    for (uint32_t i = 0; i < input->getNumAttributes(c); i++)
    {
        const POP_Attribute* attr = input->getAttribute(c, i, nullptr);
        if (!attr) continue;
        // Skip ARRAY attributes (e.g. the step3 Neighbor POP's 'Nebr', 64-wide neighbor indices):
        // fetching/passing a wide per-point array buffer through the CUDA-POP interop hangs TD, and the
        // Affector doesn't need them (the Neighbor POP recomputes them downstream every cook).
        if (attr->info.arraySize > 1) continue;
        // Skip the kernel-handled point attrs (P/v/fieldDir/field): they are fetched ONCE as CUDA in
        // execute() and reused for both the kernel and the P pass-through. Fetching them here too
        // (CPUOrCUDA) double-maps the same GL/SSBO buffer per cook — the interop conflict that hangs TD
        // on cook-context changes (zoom out / node switch). The CudaPOP SDK sample fetches each once.
        {
            const std::string nm = attr->info.name;
            if (c == POP_AttributeClass::Point &&
                (nm == "P" || nm == "v" || nm == "fieldDir" || nm == "field")) continue;
        }
        // CPUOrCUDA: leave buffers in place for safe pass-through (no forced GPU download).
        POP_GetBufferInfo info; info.location = POP_BufferLocation::CPUOrCUDA; info.stream = myStream;
        res.emplace_back(std::string(attr->info.name), attr->getBuffer(info, nullptr));
    }
    return res;
}
void
AffectorPOP::copyAllAttributes(POP_AttributeClass c,
                               std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>& attrs,
                               const POP_SetBufferInfo& sinfo, const OP_POPInput* input,
                               POP_Output* output, const std::vector<std::string>& skip)
{
    for (auto& p : attrs)
    {
        if (std::find(skip.begin(), skip.end(), p.first) != skip.end()) continue;
        const POP_Attribute* attr = input->getAttribute(c, p.first.c_str(), nullptr);
        if (!attr || !p.second) continue;
        output->setAttribute(&p.second, attr->info, sinfo, nullptr);
    }
}

// ---------------------------------------------------------------- execute
void
AffectorPOP::execute(POP_Output* output, const OP_Inputs* inputs, void*)
{
    myError = nullptr; myExecuteCount++;
    if (inputs->getNumInputs() < 1) return;
    const OP_POPInput* input = inputs->getInputPOP(0);
    if (!input) return;

    auto pointAttrs = getAllAttributes(POP_AttributeClass::Point, input);
    auto vertAttrs  = getAllAttributes(POP_AttributeClass::Vertex, input);
    auto primAttrs  = getAllAttributes(POP_AttributeClass::Primitive, input);

    POP_GetBufferInfo gi; gi.location = POP_BufferLocation::CPUOrCUDA;
    POP_InfoBuffers infoBufs; input->getAllInfoBuffers(&infoBufs, gi, nullptr);
    POP_MaxInfo maxInfo; input->getMaxInfo(&maxInfo, nullptr);

    POP_GetBufferInfo igi; igi.location = POP_BufferLocation::CPUOrCUDA; igi.stream = myStream;
    OP_SmartRef<POP_Buffer> indexBuf = input->getIndexBuffer(nullptr)->getBuffer(igi, nullptr);

    POP_GetBufferInfo cgi; cgi.location = POP_BufferLocation::CPU;
    OP_SmartRef<POP_Buffer> piCPU = input->getPointInfo(cgi, nullptr);
    uint32_t n = piCPU ? ((POP_PointInfo*)piCPU->getData(nullptr))->numPoints : maxInfo.points;
    myLastNumPoints = n;

    // Kernel attributes are intentionally EXCLUDED from pointAttrs (see getAllAttributes) so they are
    // fetched exactly once; query them directly here.
    const POP_Attribute* pA  = input->getAttribute(POP_AttributeClass::Point, "P", nullptr);
    const POP_Attribute* vA  = input->getAttribute(POP_AttributeClass::Point, "v", nullptr);
    const POP_Attribute* fdA = input->getAttribute(POP_AttributeClass::Point, "fieldDir", nullptr);
    const POP_Attribute* flA = input->getAttribute(POP_AttributeClass::Point, "field", nullptr);
    if (!pA) { myError = "Input has no 'P' point attribute."; }

    AffectorParams ap;
    ap.type = inputs->getParInt("Forcetype");
    double cx, cy, cz; inputs->getParDouble3("Center", cx, cy, cz);
    ap.cx=(float)cx; ap.cy=(float)cy; ap.cz=(float)cz;
    double ax, ay, az; inputs->getParDouble3("Axis", ax, ay, az);
    ap.ax=(float)ax; ap.ay=(float)ay; ap.az=(float)az;
    ap.strength = (float)inputs->getParDouble("Strength");
    ap.freq     = (float)inputs->getParDouble("Frequency");
    ap.radius   = (float)inputs->getParDouble("Radius");
    ap.animT    = (float)inputs->getParDouble("Animoffset");
    ap.dt       = (float)inputs->getParDouble("Timestep");
    ap.fieldMask= inputs->getParInt("Fieldmask");
    ap.damping  = (float)inputs->getParDouble("Damping");
    ap.maxSpeed = (float)inputs->getParDouble("Maxspeed");
    ap.interactStrength = (float)inputs->getParDouble("Interact");
    ap.interactRadius   = (float)inputs->getParDouble("Interactradius");
    { double a,b,c; inputs->getParDouble3("Interactpos", a,b,c); ap.ipx=(float)a; ap.ipy=(float)b; ap.ipz=(float)c; }
    bool selfSim = inputs->getParInt("Selfsim") != 0;
    ap.integrate= selfSim ? 1 : inputs->getParInt("Integrate");   // self-sim always integrates internally

    auto makeBuf = [&](uint64_t elemSize) -> OP_SmartRef<POP_Buffer>
    {
        POP_BufferInfo bi; bi.size = (uint64_t)n * elemSize; bi.mode = POP_BufferMode::SequentialWrite;
        bi.usage = POP_BufferUsage::Attribute; bi.location = POP_BufferLocation::CUDA; bi.stream = myStream;
        return myContext->createBuffer(bi, nullptr);
    };

    OP_SmartRef<POP_Buffer> Vout, Pout, Pin;   // Pin hoisted so P can pass through when not integrating
    if (pA && n > 0)
    {
        // Fetch the kernel's inputs ONCE, as CUDA. getBuffer({CUDA}) (the interop map) is called OUTSIDE
        // begin/end (the known-good pattern — moving it inside froze faster); device access is getData()
        // INSIDE begin/end.
        POP_GetBufferInfo cgia; cgia.location = POP_BufferLocation::CUDA; cgia.stream = myStream;
        Pin = pA->getBuffer(cgia, nullptr);
        OP_SmartRef<POP_Buffer> Vin  = vA  ? vA->getBuffer(cgia, nullptr)  : OP_SmartRef<POP_Buffer>();
        OP_SmartRef<POP_Buffer> FDin = fdA ? fdA->getBuffer(cgia, nullptr) : OP_SmartRef<POP_Buffer>();
        OP_SmartRef<POP_Buffer> FLin = flA ? flA->getBuffer(cgia, nullptr) : OP_SmartRef<POP_Buffer>();

        if (Pin)
        {
            Vout = makeBuf(3 * sizeof(float));                 // always output velocity
            if (ap.integrate) Pout = makeBuf(3 * sizeof(float));

            myContext->beginCUDAOperations(nullptr);
            if (selfSim && Pout)
            {
                // Self-contained sim: persistent P/V device state, (re)seeded from the input, stepped in
                // place each frame, then copied to the TD output. No external Feedback POP required.
                const size_t bytes = (size_t)n * 3 * sizeof(float);
                bool reseed = myReseed || myStateN != n || !myStateP || !myStateV;
                if (reseed)
                {
                    if (myStateP) { cudaFree(myStateP); myStateP = nullptr; }
                    if (myStateV) { cudaFree(myStateV); myStateV = nullptr; }
                    cudaError_t ea = cudaMalloc(&myStateP, bytes);
                    cudaError_t eb = cudaMalloc(&myStateV, bytes);
                    if (ea != cudaSuccess || eb != cudaSuccess) { myError = "Self-sim allocation failed"; myStateN = 0; }
                    else
                    {
                        cudaMemcpyAsync(myStateP, Pin->getData(nullptr), bytes, cudaMemcpyDeviceToDevice, myStream);
                        if (Vin) cudaMemcpyAsync(myStateV, Vin->getData(nullptr), bytes, cudaMemcpyDeviceToDevice, myStream);
                        else     cudaMemsetAsync(myStateV, 0, bytes, myStream);
                        myStateN = n; myReseed = false;
                    }
                }
                if (myStateP && myStateV)
                {
                    // step state IN PLACE (the kernel reads P/V into registers before writing, so aliasing
                    // in==out is safe), then copy state -> output for downstream.
                    launchAffector(myStateP, myStateP, myStateV, myStateV,
                                   FDin ? FDin->getData(nullptr) : nullptr, FLin ? FLin->getData(nullptr) : nullptr,
                                   n, ap, myStream);
                    cudaMemcpyAsync(Pout->getData(nullptr), myStateP, bytes, cudaMemcpyDeviceToDevice, myStream);
                    cudaMemcpyAsync(Vout->getData(nullptr), myStateV, bytes, cudaMemcpyDeviceToDevice, myStream);
                }
            }
            else
            {
                launchAffector(Pin->getData(nullptr), Pout ? Pout->getData(nullptr) : nullptr,
                               Vin ? Vin->getData(nullptr) : nullptr, Vout ? Vout->getData(nullptr) : nullptr,
                               FDin ? FDin->getData(nullptr) : nullptr, FLin ? FLin->getData(nullptr) : nullptr,
                               n, ap, myStream);
            }
            myContext->endCUDAOperations(nullptr);
        }
    }

    POP_SetBufferInfo sinfo; sinfo.stream = myStream;
    // pointAttrs already excludes P/v/fieldDir/field, so no skip list is needed.
    copyAllAttributes(POP_AttributeClass::Point, pointAttrs, sinfo, input, output, {});

    auto setAttr = [&](OP_SmartRef<POP_Buffer>& b, const char* name)
    {
        POP_AttributeInfo ai; ai.name = name; ai.numComponents = 3; ai.numColumns = 1; ai.arraySize = 0;
        ai.type = POP_AttributeType::Float;
        ai.qualifier = (name[0]=='v') ? POP_AttributeQualifier::Direction : POP_AttributeQualifier::None;
        ai.attribClass = POP_AttributeClass::Point;
        output->setAttribute(&b, ai, sinfo, nullptr);
    };
    if (Vout) setAttr(Vout, "v");                                          // affector velocity
    if (Pout) setAttr(Pout, "P");                                          // integrated position, OR:
    else if (Pin && pA) output->setAttribute(&Pin, pA->info, sinfo, nullptr); // pass input P through unchanged

    copyAllAttributes(POP_AttributeClass::Vertex, vertAttrs, sinfo, input, output, {});
    copyAllAttributes(POP_AttributeClass::Primitive, primAttrs, sinfo, input, output, {});
    output->setIndexBuffer(&indexBuf, input->getIndexBuffer(nullptr)->info, sinfo, nullptr);
    output->setInfoBuffers(&infoBufs, sinfo, nullptr);
}

// ---------------------------------------------------------------- info / params
int32_t AffectorPOP::getNumInfoCHOPChans(void*) { return 2; }
void AffectorPOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
    if (index == 0) { chan->name->setString("executeCount"); chan->value = (float)myExecuteCount; }
    else if (index == 1) { chan->name->setString("numPoints"); chan->value = (float)myLastNumPoints; }
}
void AffectorPOP::getErrorString(OP_String* error, void*) { error->setString(myError); }

void
AffectorPOP::setupParameters(OP_ParameterManager* manager, void*)
{
    {
        OP_StringParameter sp; sp.name = "Forcetype"; sp.label = "Force Type"; sp.page = "Affector";
        sp.defaultValue = "Turbulence";
        const char* n[] = { "Turbulence", "Vortex", "Attractor", "Drag", "Wind", "Fieldforce" };
        const char* l[] = { "Turbulence", "Vortex", "Attractor", "Drag", "Wind", "Field Force" };
        manager->appendMenu(sp, 6, n, l);
    }
    {
        OP_NumericParameter np; np.name = "Strength"; np.label = "Strength"; np.page = "Affector";
        np.defaultValues[0] = 1.0; np.minSliders[0] = -5.0; np.maxSliders[0] = 5.0;
        manager->appendFloat(np);
    }
    {
        OP_NumericParameter np; np.name = "Center"; np.label = "Center"; np.page = "Affector";
        for (int i=0;i<3;++i){ np.minSliders[i]=-10; np.maxSliders[i]=10; }
        manager->appendXYZ(np);
    }
    {
        OP_NumericParameter np; np.name = "Axis"; np.label = "Axis / Direction"; np.page = "Affector";
        np.defaultValues[0]=0; np.defaultValues[1]=1; np.defaultValues[2]=0;
        for (int i=0;i<3;++i){ np.minSliders[i]=-1; np.maxSliders[i]=1; }
        manager->appendXYZ(np);
    }
    auto f = [&](const char* nm, const char* lb, double d, double lo, double hi)
    {
        OP_NumericParameter np; np.name = nm; np.label = lb; np.page = "Affector";
        np.defaultValues[0] = d; np.minSliders[0] = lo; np.maxSliders[0] = hi;
        manager->appendFloat(np);
    };
    f("Frequency", "Turbulence Frequency", 0.5, 0.0, 4.0);
    f("Radius", "Attractor Radius (0=1/d^2)", 0.0, 0.0, 20.0);
    f("Animoffset", "Turbulence Anim Offset", 0.0, 0.0, 100.0);
    f("Timestep", "Timestep (dt)", 1.0/60.0, 0.0, 0.1);
    f("Damping", "Velocity Damping", 0.05, 0.0, 1.0);   // bounds accumulation; 0 = raw/undamped (can spiral out)
    f("Maxspeed", "Max Speed (0=off)", 0.0, 0.0, 100.0); // stability clamp under strong forces / param changes
    {
        // Field Mask: multiply the selected force by the upstream Dew Field's 'field' scalar, so the field
        // spatially sculpts where/how strongly ANY force mode acts. (Field Force mode already uses the field.)
        OP_NumericParameter np; np.name = "Fieldmask"; np.label = "Field Mask"; np.page = "Affector";
        np.defaultValues[0] = 0; np.minValues[0] = 0; np.maxValues[0] = 1;
        manager->appendToggle(np);
    }
    {
        // Default ON: works out-of-the-box on regular point systems (the Affector moves the points).
        // NOTE: turn this OFF when driving the SPH fluid in its feedback loop — the SPH owns position
        // integration, so ON there double-integrates and explodes.
        OP_NumericParameter np; np.name = "Integrate"; np.label = "Integrate Position"; np.page = "Affector";
        np.defaultValues[0] = 1; np.minValues[0] = 0; np.maxValues[0] = 1;
        manager->appendToggle(np);
    }
    {
        // Self-contained simulation: the node keeps its own particle state and steps every frame, so
        // it accumulates motion WITHOUT an external Feedback POP. Forces integration on internally.
        // Leave OFF to use it as a stateless stepper inside an existing feedback loop (e.g. the SPH fluid).
        OP_NumericParameter np; np.name = "Selfsim"; np.label = "Self Simulate (no feedback loop)"; np.page = "Affector";
        np.defaultValues[0] = 0; np.minValues[0] = 0; np.maxValues[0] = 1;
        manager->appendToggle(np);
    }
    {
        OP_NumericParameter np; np.name = "Reset"; np.label = "Reset"; np.page = "Affector";
        manager->appendPulse(np);   // re-seed the self-sim state from the current input
    }
    // Interactive point force (drive Interact Position with a Mouse In / touch CHOP) — additive on top of Force Type.
    f("Interact", "Interact Strength (+/-)", 0.0, -50.0, 50.0);
    f("Interactradius", "Interact Radius", 0.5, 0.01, 10.0);
    { OP_NumericParameter np; np.name="Interactpos"; np.label="Interact Position"; np.page="Affector";
      np.defaultValues[0]=0; np.defaultValues[1]=0; np.defaultValues[2]=0;
      for(int k=0;k<3;++k){ np.minSliders[k]=-5; np.maxSliders[k]=5; }
      manager->appendXYZ(np); }
}
