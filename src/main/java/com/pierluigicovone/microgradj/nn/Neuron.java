package com.pierluigicovone.microgradj.nn;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;

import java.util.ArrayList;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.List;

/**
 * The behavior of a neuron is mathematically described as:
 *              ∑_i (x_i * w_i) + b
 * where:
 *  i := number of accepted inputs "x_i"
 *       (and so: the number of weights "w_i")
 *  b := the bias.
 */
public class Neuron {

    // FIELDS

    private final int nin;

    private final List<DiffScalarNode> weights;
    private DiffScalarNode bias;


    // METHODS

    /**
     * Take in input the number "i" of accepted inputs
     * (and so, initialize the proper number of weights),
     * and a Random instance to initialize values of
     * each weight and of the bias, to a Random value
     * between "-1" and "1" [It's a simple Heuristic].
     */
    public Neuron(int nin, Random random) {

        // Check that the neuron works with the proper number of inputs
        if (nin <= 0) {
            throw new IllegalArgumentException("[Err] The number of inputs must be greater than 0");
        }

        // Lambda expression that generates a double value in [-1,1).
        DoubleSupplier randomW = () -> random.nextDouble() * 2 - 1;

        // Initialize fields
        this.nin = nin;

        this.weights = new ArrayList<>(nin);
        for (int i = 0; i < nin; i++) {
            this.weights.add( DiffScalarNode.leaf(randomW.getAsDouble(), "w"+i) );
        }

        this.bias = DiffScalarNode.leaf(randomW.getAsDouble(), "b");
    }


    /**
     * Allows the neuron to perform the forward pass.
     */
    public DiffScalarNode forward(List<DiffScalarNode> inputs) {

        // Check the integrity of inputs
        int inputSize = inputs.size();
        if (inputSize != nin) {
            throw new IllegalArgumentException("[Err] Neuron expects " + nin + " inputs but got " + inputSize);
        }

        // Otherwise: start by performing the dot product
        DiffScalarNode sum = weights.getFirst().mul( inputs.getFirst() );
        for (int i = 1; i < nin; i++) {
            sum = sum.add(
                    weights.get(i).mul( inputs.get(i) )
            );
        }

        // Add the bias: here the net-input is ready
        DiffScalarNode out = sum.add(bias);

        return out.tanh();
    }

}
