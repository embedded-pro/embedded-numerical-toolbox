#ifndef NEURAL_NETWORK_DENSE_HPP
#define NEURAL_NETWORK_DENSE_HPP

#include "numerical/neural_network/activation/ActivationFunction.hpp"
#include "numerical/neural_network/layer/Layer.hpp"
#include <random>

namespace neural_network
{
    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    class Dense
        : public Layer<QNumberType, InputSize, OutputSize, (InputSize * OutputSize) + OutputSize>
    {
        static_assert(std::is_base_of<ActivationFunction<QNumberType>, Activation>::value,
            "Activation has to be derived from ActivationFunction.");

    public:
        using BaseLayer = Layer<QNumberType, InputSize, OutputSize, (InputSize * OutputSize) + OutputSize>;
        using InputVector = typename BaseLayer::InputVector;
        using OutputVector = typename BaseLayer::OutputVector;
        using ParameterVector = typename BaseLayer::ParameterVector;

        Dense();

        void Forward(const InputVector& input) override;
        InputVector& Backward(const OutputVector& output_gradient) override;
        ParameterVector& Parameters() const override;
        void SetParameters(const ParameterVector& parameters) override;

    private:
        Activation activationFunction;

        math::Matrix<QNumberType, OutputSize, InputSize> weights;
        math::Vector<QNumberType, OutputSize> biases;

        InputVector input;
        OutputVector preActivation;
        OutputVector output;

        math::Matrix<QNumberType, OutputSize, InputSize> weightGradients;
        math::Vector<QNumberType, OutputSize> biasGradients;

        mutable ParameterVector parameters;
        InputVector inputGradient;
    };

    // Implementation //

    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    Dense<QNumberType, InputSize, OutputSize, Activation>::Dense()
    {
        std::random_device random;
        std::mt19937 generator(random());
        std::normal_distribution<float> normalDistribution(0.0, std::sqrt(2.0 / InputSize));

        for (std::size_t i = 0; i < OutputSize; ++i)
        {
            for (std::size_t j = 0; j < InputSize; ++j)
                weights.at(i, j) = QNumberType(normalDistribution(generator));

            biases[i] = QNumberType(0.0f);
        }
    }

    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    void Dense<QNumberType, InputSize, OutputSize, Activation>::Forward(const InputVector& input)
    {
        this->input = input;

        for (std::size_t i = 0; i < OutputSize; ++i)
        {
            preActivation[i] = biases[i];
            for (std::size_t j = 0; j < InputSize; ++j)
                preActivation[i] += weights.at(i, j) * input[j];

            output[i] = activationFunction.Forward(preActivation[i]);
        }
    }

    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    typename Dense<QNumberType, InputSize, OutputSize, Activation>::InputVector& Dense<QNumberType, InputSize, OutputSize, Activation>::Backward(const OutputVector& output_gradient)
    {
        OutputVector preActivationGradient;

        for (std::size_t i = 0; i < OutputSize; ++i)
            preActivationGradient[i] = activationFunction.Backward(output_gradient[i]);

        for (std::size_t j = 0; j < InputSize; ++j)
        {
            inputGradient[j] = QNumberType(0.0f);
            for (std::size_t i = 0; i < OutputSize; ++i)
                inputGradient[j] += weights.at(i, j) * preActivationGradient[i];
        }

        for (std::size_t i = 0; i < OutputSize; ++i)
        {
            for (std::size_t j = 0; j < InputSize; ++j)
                weightGradients.at(i, j) = preActivationGradient[i] * input[j];

            biasGradients[i] = preActivationGradient[i];
        }

        return inputGradient;
    }

    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    typename Dense<QNumberType, InputSize, OutputSize, Activation>::ParameterVector& Dense<QNumberType, InputSize, OutputSize, Activation>::Parameters() const
    {
        std::size_t idx = 0;

        for (std::size_t i = 0; i < OutputSize; ++i)
            for (std::size_t j = 0; j < InputSize; ++j)
                parameters[idx++] = weights.at(i, j);

        for (std::size_t i = 0; i < OutputSize; ++i)
            parameters[idx++] = biases[i];

        return parameters;
    }

    template<typename QNumberType, std::size_t InputSize, std::size_t OutputSize, class Activation>
    void Dense<QNumberType, InputSize, OutputSize, Activation>::SetParameters(const ParameterVector& parameters)
    {
        std::size_t idx = 0;

        for (std::size_t i = 0; i < OutputSize; ++i)
            for (std::size_t j = 0; j < InputSize; ++j)
                weights.at(i, j) = parameters[idx++];

        for (std::size_t i = 0; i < OutputSize; ++i)
            biases[i] = parameters[idx++];
    }
}

#endif
