#ifndef MATH_SIMD_STUB_HPP
#define MATH_SIMD_STUB_HPP

#include "numerical/math/SingleInstructionMultipleData.hpp"

namespace math
{
    class SingleInstructionMultipltDataStub
        : public SingleInstructionMultipltData
    {
    public:
        uint32_t Pkhbt(uint32_t value1, uint32_t value2, uint32_t value3) override
        {
            return 0;
        }

        uint32_t Pkhtb(uint32_t value1, uint32_t value2, uint32_t value3) override
        {
            return 0;
        }

        uint32_t Qadd(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qadd16(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qadd8(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qasx(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qsax(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qsub(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qsub16(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Qsub8(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Shadd16(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Shasx(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Shsax(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Shsub16(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Smlad(uint32_t value1, uint32_t value2, uint32_t value3) override
        {
            return 0;
        }

        uint32_t Smladx(uint32_t value1, uint32_t value2, uint32_t value3) override
        {
            return 0;
        }

        uint64_t Smlald(uint32_t value1, uint32_t value2, uint64_t value3) override
        {
            return 0;
        }

        unsigned long long Smlaldx(uint32_t value1, uint32_t value2, unsigned long long value3) override
        {
            return 0;
        }

        uint32_t Smlsdx(uint32_t value1, uint32_t value2, uint32_t value3) override
        {
            return 0;
        }

        uint32_t Smmla(int32_t value1, int32_t value2, int32_t value3) override
        {
            return 0;
        }

        uint32_t Smuad(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Smuadx(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Smusd(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Smusdx(uint32_t value1, uint32_t value2) override
        {
            return 0;
        }

        uint32_t Sxtb16(uint32_t value) override
        {
            return 0;
        }
    };
}

#endif
