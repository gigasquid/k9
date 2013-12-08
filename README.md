# k9

A small library using core.matrix to construct Neural Networks

## Usage

Construct simple 3 layer networks with

```clojure
(construct-network n-inputs n-hiddens n-ouputs)
```
Example
```clojure
(construct-network 2 3 2) ;=> [ [0 0] [input-to-hidden-strengths]
[0 0 0] [hidden-to-output-strengths] [0 0]]
```

Feed foward input and get back output neuron values with
```clojure
(ff input network)
```

Example
```clojure
(ff [1 0] (construct-network 2 3 2));=>[0.023969361623158485 0.014886788800864243]
```

Train the network on data in the form of [[input target]
[input target] ... ] => returns a new network

```clojure
(train-data network data learning-rate)
```

Example
```clojure
(def nn (construct-network 2 3 2))
#'user/nn
;; without training
(ff [1 0] nn) ;=> [0.03061049829949632 0.043037351551821625]
(def n1 (train-data nn  [
                         [[1 0] [0 1]]
                         [[0.5 0] [0 0.5]]
                         [[0.25 0] [0 0.25]]]
                     0.2))
(ff [1 0] n1) ;=>
[0.0383350329723964 0.06845383345543034]
````

Another example
```clojure
(defn inverse-data []
  (let [n (rand 1)]
    [[n 0] [0 n]]))

(def n3 (train-data nn (repeatedly 400 inverse-data) 0.5))

(ff [1 0] n3) ;=> [-3.0872502374300364E-4 0.8334331107408276]
````

Can also train the network repeatedly on a set of data for "epochs"
```clojure
(train-epochs n network training-data learning-rate)
```

Example
```clojure
(def n4 (train-epochs 5 nn (repeatedly 200 inverse-data) 0.2))
(ff [1 0] n4) ;=> [-3.794899940782748E-4 0.8105184486966243
```

## Example with Colors
There is another example in the examples directory where the network learns to name colors based on their rgb value.

## Blog Post
I made a blog post about making a simple neural network with an example
here:
[http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/])(http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/)


## License

Copyright © 2013 Carin Meier

Distributed under the Eclipse Public License, the same as Clojure
