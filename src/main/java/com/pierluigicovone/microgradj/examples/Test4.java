package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;
import com.pierluigicovone.microgradj.nn.Neuron;
import com.pierluigicovone.microgradj.viz.GraphRenderer;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Test of a single Neuron, fed with 3 input values.
 */
public class Test4 {


    public static void main(String[] args) throws IOException {

        // To pretty format numbers.
        Locale.setDefault(Locale.US);

        // Fixed seed for reproducibility :)
        Random random = new Random(42);

        // Build a neuron with 3 inputs
        Neuron neuron = new Neuron(3, random);

        // Inputs
        List<DiffScalarNode> inputs = List.of(
                DiffScalarNode.leaf(1.5, "x0"),
                DiffScalarNode.leaf(-2.0, "x1"),
                DiffScalarNode.leaf(0.5, "x2")
        );

        // Forward pass
        DiffScalarNode o = neuron.forward(inputs).withName("o");

        // Backward pass
        //o.backward();

        // Logs
        System.out.println("Test 4");
        System.out.println(" o = tanh( SUM_i(x_i * w_i) + b )");
        System.out.printf(" o = %.4f%n", o.getData());

        GraphRenderer.renderToFile(o, "graph.png");
        System.out.println("Graph rendered to: graph.png");
    }
}