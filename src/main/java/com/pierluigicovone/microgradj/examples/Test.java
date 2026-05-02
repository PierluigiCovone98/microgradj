package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;

/**
 * Testing some functions of the "DifferentiableScalarNode" class,
 * while building it.
 */
public class Test {

    public static void main(String[] args) {

        // 1. Testing the "toString()" method.
        DiffScalarNode a = new DiffScalarNode(5.0);
        DiffScalarNode b = new DiffScalarNode(10.0);

        System.out.println(a);
        System.out.println(b);

    }


}
