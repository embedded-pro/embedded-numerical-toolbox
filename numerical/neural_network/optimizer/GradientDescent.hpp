#ifndef NEURAL_NETWORK_GRADIENT_DESCENT_HPP
#define NEURAL_NETWORK_GRADIENT_DESCENT_HPP

#include "numerical/neural_network/optimizer/Optimizer.hpp"
#include <optional>

namespace neural_network
{
    template<typename QNumberType, size_t NumberOfFeatures>
    class GradientDescent
        : public Optimizer<QNumberType, NumberOfFeatures>
    {
        static_assert(math::is_qnumber<QNumberType>::value || std::is_floating_point<QNumberType>::value,
            "GradientDescent can only be instantiated with math::QNumber types.");

    public:
        using Result = typename Optimizer<QNumberType, NumberOfFeatures>::Result;
        using Vector = typename Loss<QNumberType, NumberOfFeatures>::Vector;

        struct Parameters
        {
            QNumberType learningRate;
            size_t maxIterations;
        };

        explicit GradientDescent(const Parameters& params);
        const Result& Minimize(const Vector& initialGuess, Loss<QNumberType, NumberOfFeatures>& loss) override;

    private:
        Parameters parameters;
        std::optional<Result> result;
    };

    // Implementation //

    template<typename QNumberType, size_t NumberOfFeatures>
    GradientDescent<QNumberType, NumberOfFeatures>::GradientDescent(const Parameters& params)
        : parameters(params)
    {
        really_assert(params.learningRate > QNumberType(0.0f));
        really_assert(params.maxIterations > 0);
    }

    template<typename QNumberType, size_t NumberOfFeatures>
    const typename GradientDescent<QNumberType, NumberOfFeatures>::Result& GradientDescent<QNumberType, NumberOfFeatures>::Minimize(const Vector& initialGuess, Loss<QNumberType, NumberOfFeatures>& loss)
    {
        auto currentParams = initialGuess;
        auto currentCost = loss.Cost(currentParams);
        Vector previousParams;
        size_t iteration = 0;

        while (iteration < parameters.maxIterations)
        {
            previousParams = currentParams;

            auto gradient = loss.Gradient(currentParams);
            currentParams = currentParams - (gradient * parameters.learningRate);

            currentCost = loss.Cost(currentParams);
            ++iteration;
        }

        result.emplace(currentParams, currentCost, iteration);

        return *result;
    }
}

#endif
