package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;

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
    public static void main(String[] args) {

        // Define all nodes
        DiffScalarNode a = DiffScalarNode.leaf(2.0);        // a
        DiffScalarNode b = DiffScalarNode.leaf(-3.0);       // b
        DiffScalarNode d = a.add(b);                            // d = a + b

        DiffScalarNode c = DiffScalarNode.leaf(5.0);       // c
        DiffScalarNode e = c.add(2.0);                          // e = c + 2.0

        DiffScalarNode L = d.sub(e);   // L = d + e = ( a + b ) - ( c + 2.0)


    }

}
