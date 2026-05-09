package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;
import com.pierluigicovone.microgradj.viz.GraphRenderer;

import java.io.IOException;

/**
 * From the definition of a neuron as mathematical expression, we know that
 * a single neuron takes some inputs
 *                      x0, x1, ..., x_n
 * that are propagated through "synapses", each of one has a weight
 *                      w0, w1, ..., w_n
 * as:
 *                          x_i * w_i
 *
 * Inside the cell of a neuron, it is computed the dot sum of each component
 * and the final result is added to a bias "b", that is independent of inputs:
 *                      n := SUM_i ( x_i * w_i ) + b
 *
 * Then is produced an output "o", that is defined as
 *                          o := ƒ(n),
 * where "ƒ" is a non-linear function.
 *
 * Here we are going to test a simple neuron.
 */
public class Test3 {

    public static void main(String[] args) throws IOException {

        // Inputs
        DiffScalarNode x1 = DiffScalarNode.leaf(2.0, "x1");
        DiffScalarNode x2 = DiffScalarNode.leaf(0.0, "x2");

        // Weights
        DiffScalarNode w1 = DiffScalarNode.leaf(-3.0, "w1");
        DiffScalarNode w2 = DiffScalarNode.leaf(1.0, "w2");

        // Bias
        DiffScalarNode b = DiffScalarNode.leaf(6.8813735870195432, "b");

        // Compute the output
        DiffScalarNode x1w1 = x1.mul(w1); x1w1.withName("x1 * w1");
        DiffScalarNode x2w2 = x2.mul(w2); x2w2.withName("x2 * w2");

        DiffScalarNode x1w1x2w2 =x1w1.add(x2w2); x1w1x2w2.withName("x1w1 + x2w2");

        DiffScalarNode n = x1w1x2w2.add(b); n.withName("n");

        DiffScalarNode o = n.tanh();    // :)

        // Logs
        System.out.println("Test 3");
        System.out.println(" o = tanh( (x1 * w1) + (x1 * w1) + b )");

        GraphRenderer.renderToFile(o, "graph.png");
        System.out.println("Graph rendered to: graph.png");
    }
}
