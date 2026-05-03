package com.pierluigicovone.microgradj.examples;

import com.pierluigicovone.microgradj.autograd.DiffScalarNode;

/**
 * Testing some functions of the "DifferentiableScalarNode" class,
 * while building it.
 */
public class Test {

    public static void main(String[] args) {

        // --- 1. Testing the "toString()" method... --- OK
        DiffScalarNode a = new DiffScalarNode(5.0);
        DiffScalarNode b = new DiffScalarNode(10.0);

        System.out.println(a);
        System.out.println(b);

        // --- 2. Let's try: --- OK
        //   2.1. "+"
        DiffScalarNode c1 = a.add(b);       // a+b
        DiffScalarNode c2 = b.add(a);       // b+a

        System.out.println( c1 );
        System.out.println( c2 );
        System.out.println( c1.equals(c2) );  // Expected: true

        //   2.2. "-"
        DiffScalarNode d1 = a.sub(b);       // a-b
        DiffScalarNode d2 = b.sub(a);       // b-a

        System.out.println( d1 );
        System.out.println( d2 );

        //   2.3. "+" by Number
        DiffScalarNode e1 = a.add(4);
        DiffScalarNode e2 = b.add(-6);

        System.out.println( e1 );
        System.out.println( e2 );

        //   2.4. "-" by Number
        DiffScalarNode f1 = a.sub(4);
        DiffScalarNode f2 = b.sub(-6);

        System.out.println( f1 );
        System.out.println( f2 );


        // --- 3. [...] ----

    }



}
