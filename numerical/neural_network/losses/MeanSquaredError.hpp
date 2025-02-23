#ifndef NEURAL_NETWORK_LOSSES_MEAN_SQUARED_ERROR_HPP
#define NEURAL_NETWORK_LOSSES_MEAN_SQUARED_ERROR_HPP

#include "numerical/neural_network/losses/Loss.hpp"
#include "numerical/neural_network/regularization/Regularization.hpp"

namespace neural_network
{
    template<typename QNumberType, std::size_t NumberOfFeatures>
    class MeanSquaredError
        : public Loss<QNumberType, NumberOfFeatures>
    {
    public:
        using Vector = typename Loss<QNumberType, NumberOfFeatures>::Vector;

        MeanSquaredError(const Vector& target, const Regularization<QNumberType, NumberOfFeatures>& regularization);

        QNumberType Cost(const Vector& parameters) override;
        Vector Gradient(const Vector& parameters) override;

    private:
        Vector target;
        Regularization<QNumberType, NumberOfFeatures>& regularization;
    };

    // Implementation //

    template<typename QNumberType, std::size_t NumberOfFeatures>
    MeanSquaredError<QNumberType, NumberOfFeatures>::MeanSquaredError(const Vector& target, const Regularization<QNumberType, NumberOfFeatures>& regularization)
        : target(target)
        , regularization(regularization)
    {}

    template<typename QNumberType, std::size_t NumberOfFeatures>
    QNumberType MeanSquaredError<QNumberType, NumberOfFeatures>::Cost(const Vector& parameters)
    {
        QNumberType cost = QNumberType(0);

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            QNumberType diff = parameters[i] - target[i];
            cost += diff * diff;
        }

        cost /= QNumberType(2.0f);

        return cost + regularization.Calculate(parameters);
    }

    template<typename QNumberType, std::size_t NumberOfFeatures>
    typename MeanSquaredError<QNumberType, NumberOfFeatures>::Vector MeanSquaredError<QNumberType, NumberOfFeatures>::Gradient(const Vector& parameters)
    {
        Vector gradient;

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            gradient[i] = parameters[i] - target[i];

        return gradient + regularization.Calculate(parameters);
    }
}

#endif
