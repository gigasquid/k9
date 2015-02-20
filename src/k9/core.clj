(ns k9.core
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

(def activation-fn
  "defines the function implemented by a neuron"
  (fn [x] (Math/tanh x)))
(def dactivation-fn
  "derivative of activation function"
  (fn [y] (- 1.0 (* y y))))

(def range2
  "returns range with interval 2"
  #(take (- %2 %1) (iterate (partial + 2) %1)))

(defn get-weights [network]
  "returns the weights of a network"
  (map first (partition 1 2 (rest network))))

(defn get-layers [network]
  "returns the layers of a network, including input and output"
  (map first (partition 1 2 network)))

(defn layer-activation [inputs strengths]
   "forward propagate the input of a layer"
   (mapv activation-fn
         (mapv #(reduce + %)
               (* inputs (transpose strengths)))))

(defn output-deltas [targets outputs]
  "measures the delta errors for the output layer (Desired value – actual value) and multiplying it by the gradient of the activation function"
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))

(defn hlayer-deltas [deltas [neurons strengths]]
  "measures the delta errors for the hidden layer based on the output deltas"
  (* (mapv dactivation-fn neurons)
     (mapv #(reduce + %)
           (* deltas strengths))))

(defn update-strengths [[deltas neurons strengths lrate]]
  "update the strengths based on the deltas and the learning rate"
  (+ strengths (* lrate
                  (mapv #(* deltas %) neurons))))

(defn feed-forward [input network]
  "feeds input through the network to the output"
  (let [strenghts (get-weights network) ;get weight vectors between input, hidden levels and output
        new-activations (reductions layer-activation input strenghts) ;get the new level activations given the inputs
        activations-indexes (cons 0 (map (partial + 2) (range2 0 (count strenghts))))] ;get the indexes corresponding to the position of level activations in network
    (apply (partial assoc network) ; replace values at indexes with replacement values
           (interleave activations-indexes new-activations)))) ; associate positions with replacement values

(defn update-weights [network target learning-rate]
  "updates the weights based on targets and learning rate with back-prop"
  (let [strenghts (reverse (get-weights network)) ;get weight vectors between input, hidden levels and output
        layers (reverse (get-layers network)) ;get layers values
        o-deltas (output-deltas target (first layers)) ;use the output layer
        h-deltas (->> ;compute the deltas for the hidden layers starting from the output deltas (include the init value: o-deltas)
                  (mapcat #(list [%1 %2]) (rest layers) strenghts) ;list of vectors containing a hidden layer and the weights to the next layer
                  (butlast ,)           ;ignore the input layer
                  (reductions hlayer-deltas o-deltas ,) ;produce the deltas for the hidden layers
                  )
        h-deltas-layer-weights (mapcat #(list [%1 %2 %3 learning-rate]) h-deltas (rest layers) strenghts) ;collect values from the three parameter vectors in a single list of vectors
        n-strenghts (map update-strengths h-deltas-layer-weights) ;produce the new strenghts
        strenghts-indexes (reverse (map inc (range2 0 (count strenghts))))] ;get the indexes corresponding to the position of weights in network
    (apply (partial assoc network) ; replace values at indexes with replacement values
           (interleave strenghts-indexes n-strenghts)))) ; associate positions with replacement values

(defn train-network [network input target learning-rate]
  "train network with one set of target data"
  (update-weights (feed-forward input network) target learning-rate))

(defn train-data [network data learning-rate]
  "train network with a set of data in the form of [[input1 target1] [input2 target2]]"
  (if-let [[input target] (first data)]
    (recur
     (train-network network input target learning-rate)
     (rest data)
     learning-rate)
    network))

(defn train-epochs [n network training-data learning-rate]
  "train repeatedly n times over the same tranining data in the form of [[input1 target1] [input2 target2]]  "
  (if (zero? n)
    network
    (recur (dec n)
           (train-data network training-data learning-rate)
           training-data
           learning-rate)))

(defn ff [input network]
  "Feed forward and return output neurons"
  (last (feed-forward input network)))

(defn gen-strengths [to from]
  "generate random strengths for layer"
  (let [l (* to from)]
    (map vec (partition from (repeatedly l #(rand (/ 1 l)))))))

(defn construct-network
  ([num-in num-hidden num-out]
  "construct a three layer neural network"
  (construct-network num-in num-hidden 0 num-out))
  ([size-in size-hidden num-hidden size-out]
  "construct a N layer neural network"
  (vec (map vec (concat
                 [(repeat size-in 0)
                  (gen-strengths size-in size-hidden)
                  (repeat size-hidden 0)]
                 (->>
                  (cons (gen-strengths size-hidden size-hidden) [(repeat size-hidden 0)])
                  (repeat (dec num-hidden))
                  (apply concat))
                 [(gen-strengths size-hidden size-out)
                  (repeat size-out 0)])))))
