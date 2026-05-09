package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;
import com.pierluigicovone.microgradj.viz.GraphRenderer;

import java.io.IOException;
import java.util.Locale;

/**
 * Testing a simple math expression using "DiffScalarNodes".
 * In particular, this expression uses only the "+" and "-" operations.
 * This is intentionally left simple to test at a first glance,
 * if visualization actually works. The equation is:
 *                  L = (a + b) - (c + 2.0),
 *      with:
 *          a =  2.0,
 *          b = -3.0,
 *          c =  5.0
 */
public class Test2 {

    /**
     * Entry point for the application.
     */
    public static void main(String[] args) throws IOException {

        // To pretty format numbers.
        Locale.setDefault(Locale.US);

        // Define all nodes
        DiffScalarNode a = DiffScalarNode.leaf(2.0, "a");
        DiffScalarNode b = DiffScalarNode.leaf(-3.0, "b");
        DiffScalarNode c = DiffScalarNode.leaf(5.0, "c");


        DiffScalarNode d = a.add(b).withName("d");
        DiffScalarNode e = c.add(2.0).withName("e");
        DiffScalarNode f  = e.neg().withName("f");

        DiffScalarNode L = d.add(f).withName("L");

        System.out.println("Test 2 — L = (a + b) - (c + 2.0)");
        System.out.printf("  a = %.2f   b = %.2f   c = %.2f%n",
                a.getData(), b.getData(), c.getData());
        System.out.printf("  L = %.4f%n", L.getData());

        GraphRenderer.renderToFile(L, "graph.png");
        System.out.println("Graph rendered to: graph.png");
    }

}
