package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;
import com.pierluigicovone.microgradj.viz.GraphRenderer;

import java.io.IOException;

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

        // Define all nodes
        DiffScalarNode a = DiffScalarNode.leaf(2.0);        // a
        DiffScalarNode b = DiffScalarNode.leaf(-3.0);       // b
        DiffScalarNode d = a.add(b);                            // d = a + b

        DiffScalarNode c = DiffScalarNode.leaf(5.0);       // c
        DiffScalarNode e = c.add(2.0);                          // e = c + 2.0

        DiffScalarNode L = d.sub(e);   // L = d + e = ( a + b ) - ( c + 2.0)

        System.out.println("==========================================");
        System.out.println("  TEST 2: Graph Visualization");
        System.out.println("  Equation: L = (a + b) - (c + 2.0)");
        System.out.println("==========================================");
        System.out.println();
        System.out.println("Inputs:");
        System.out.printf("  a = %.4f%n", a.getData());
        System.out.printf("  b = %.4f%n", b.getData());
        System.out.printf("  c = %.4f%n", c.getData());
        System.out.println();
        System.out.println("Intermediate results:");
        System.out.printf("  d = a + b   = %.4f%n", d.getData());
        System.out.printf("  e = c + 2.0 = %.4f%n", e.getData());
        System.out.println();
        System.out.printf("Final result: L = %.4f%n", L.getData());
        System.out.println();

        GraphRenderer.renderToFile(L, "graph.png");
        System.out.println("Graph rendered to: graph.png");
        System.out.println("==========================================");
    }

}
