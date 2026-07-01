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
AffectorPOP::~AffectorPOP() { if (myStream) cudaStreamDestroy(myStream); }

void AffectorPOP::getGeneralInfo(POP_GeneralInfo*, const OP_Inputs*, void*) {}

std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>
AffectorPOP::getAllAttributes(POP_AttributeClass c, const OP_POPInput* input)
{
    std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>> res;
    for (uint32_t i = 0; i < input->getNumAttributes(c); i++)
    {
        const POP_Attribute* attr = input->getAttribute(c, i, nullptr);
        if (!attr) continue;
        // CPUOrCUDA: leave buffers in place for safe pass-through (no forced GPU download, which
        // would require begin/end bracketing even on pass-through paths => freeze).
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

    bool haveP=false, haveV=false, haveFD=false, haveFL=false;
    for (auto& a : pointAttrs)
    {
        if      (a.first == "P")        haveP  = true;
        else if (a.first == "v")        haveV  = true;
        else if (a.first == "fieldDir") haveFD = true;
        else if (a.first == "field")    haveFL = true;
    }
    if (!haveP) { myError = "Input has no 'P' point attribute."; }

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
    ap.integrate= inputs->getParInt("Integrate");

    auto makeBuf = [&](uint64_t elemSize) -> OP_SmartRef<POP_Buffer>
    {
        POP_BufferInfo bi; bi.size = (uint64_t)n * elemSize; bi.mode = POP_BufferMode::SequentialWrite;
        bi.usage = POP_BufferUsage::Attribute; bi.location = POP_BufferLocation::CUDA; bi.stream = myStream;
        return myContext->createBuffer(bi, nullptr);
    };

    OP_SmartRef<POP_Buffer> Vout, Pout;
    if (haveP && n > 0)
    {
        const POP_Attribute* pA  = input->getAttribute(POP_AttributeClass::Point, "P", nullptr);
        const POP_Attribute* vA  = haveV  ? input->getAttribute(POP_AttributeClass::Point, "v", nullptr)        : nullptr;
        const POP_Attribute* fdA = haveFD ? input->getAttribute(POP_AttributeClass::Point, "fieldDir", nullptr) : nullptr;
        const POP_Attribute* flA = haveFL ? input->getAttribute(POP_AttributeClass::Point, "field", nullptr)    : nullptr;

        if (pA)
        {
            Vout = makeBuf(3 * sizeof(float));                 // always output velocity (createBuffer: device alloc)
            if (ap.integrate) Pout = makeBuf(3 * sizeof(float));

            // CRITICAL: the forced-CUDA getBuffer() calls (the GL/SSBO->CUDA interop map) MUST live
            // INSIDE beginCUDAOperations/endCUDAOperations. Calling them outside leaks an interop
            // registration per cook => the driver hangs after ~a couple seconds (esp. with a live GLSL-
            // SPH input re-mapping a fresh SSBO every frame, and outside a feedback loop). See the
            // suite's hard-won CUDA-POP lessons (#3).
            POP_GetBufferInfo cgia; cgia.location = POP_BufferLocation::CUDA; cgia.stream = myStream;
            myContext->beginCUDAOperations(nullptr);
            OP_SmartRef<POP_Buffer> Pin  = pA->getBuffer(cgia, nullptr);
            OP_SmartRef<POP_Buffer> Vin  = vA  ? vA->getBuffer(cgia, nullptr)  : OP_SmartRef<POP_Buffer>();
            OP_SmartRef<POP_Buffer> FDin = fdA ? fdA->getBuffer(cgia, nullptr) : OP_SmartRef<POP_Buffer>();
            OP_SmartRef<POP_Buffer> FLin = flA ? flA->getBuffer(cgia, nullptr) : OP_SmartRef<POP_Buffer>();
            if (Pin)
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
    std::vector<std::string> skip;
    if (Vout) skip.push_back("v");
    if (Pout) skip.push_back("P");
    copyAllAttributes(POP_AttributeClass::Point, pointAttrs, sinfo, input, output, skip);

    auto setAttr = [&](OP_SmartRef<POP_Buffer>& b, const char* name)
    {
        POP_AttributeInfo ai; ai.name = name; ai.numComponents = 3; ai.numColumns = 1; ai.arraySize = 0;
        ai.type = POP_AttributeType::Float;
        ai.qualifier = (name[0]=='v') ? POP_AttributeQualifier::Direction : POP_AttributeQualifier::None;
        ai.attribClass = POP_AttributeClass::Point;
        output->setAttribute(&b, ai, sinfo, nullptr);
    };
    if (Vout) setAttr(Vout, "v");
    if (Pout) setAttr(Pout, "P");

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
    {
        OP_NumericParameter np; np.name = "Integrate"; np.label = "Integrate Position"; np.page = "Affector";
        np.defaultValues[0] = 0; np.minValues[0] = 0; np.maxValues[0] = 1;
        manager->appendToggle(np);
    }
}
