/* Particle Affector POP — a GPU Custom POP that applies a force to each particle's velocity
 * (v += force*dt), optionally integrating position. Forces: Turbulence (curl) / Vortex /
 * Attractor / Drag / Wind / Field Force (from a Field POP). Use in a Feedback POP loop. */
#pragma once

#include "POP_CPlusPlusBase.h"
#include <vector>
#include <string>
#include <utility>
#include "cuda_runtime.h"

using namespace TD;

class AffectorPOP : public POP_CPlusPlusBase
{
public:
    AffectorPOP(const OP_NodeInfo* info, POP_Context* context);
    virtual ~AffectorPOP();

    virtual void    getGeneralInfo(POP_GeneralInfo*, const OP_Inputs*, void*) override;
    virtual void    execute(POP_Output*, const OP_Inputs*, void*) override;
    virtual int32_t getNumInfoCHOPChans(void*) override;
    virtual void    getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
    virtual void    getErrorString(OP_String* error, void*) override;
    virtual void    setupParameters(OP_ParameterManager* manager, void*) override;

private:
    std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>
        getAllAttributes(POP_AttributeClass c, const OP_POPInput* input);
    void copyAllAttributes(POP_AttributeClass c,
                           std::vector<std::pair<std::string, OP_SmartRef<POP_Buffer>>>& attrs,
                           const POP_SetBufferInfo& sinfo, const OP_POPInput* input,
                           POP_Output* output, const std::vector<std::string>& skip);

    const OP_NodeInfo* myNodeInfo;
    int32_t            myExecuteCount;
    cudaStream_t       myStream;
    POP_Context*       myContext;
    const char*        myError;
    uint32_t           myLastNumPoints;
};
