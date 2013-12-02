(ns k9.core
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

(def activation-fn (fn [x] (Math/tanh x)))
(def dactivation-fn (fn [y] (- 1.0 (* y y))))

(defn layer-activation [inputs strengths]
  "forward propogate the input of a layer"
  (mapv activation-fn
        (mapv #(reduce + %)
              (* inputs (transpose strengths)))))

(defn output-deltas [targets outputs]
  "measures the delta errors for the output layer (Desired value – actual value) and multiplying it by the gradient of the activation function"
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))

(defn hlayer-deltas [odeltas neurons strengths]
  "measures the delta errors for the hidden layer based on the output deltas"
  (* (mapv dactivation-fn neurons)
     (mapv #(reduce + %)
           (* odeltas strengths))))

(defn update-strengths [deltas neurons strengths lrate]
  "update the strengths based on the deltas and the learning rate"
  (+ strengths (* lrate
                  (mapv #(* deltas %) neurons))))

(defn feed-forward [input network]
  "feeds input through the network to the output"
  (let [[in i-h-strengths h h-o-strengths out] network
        new-h (layer-activation input i-h-strengths)
        new-o (layer-activation new-h h-o-strengths)]
    [input i-h-strengths new-h h-o-strengths new-o]))

(defn update-weights [network target learning-rate]
  "updates the weights based on targets and learning rate with back-prop"
  (let [[ in i-h-strengths h h-o-strengths out] network
        o-deltas (output-deltas target out)
        h-deltas (hlayer-deltas o-deltas h h-o-strengths)
        n-h-o-strengths (update-strengths
                         o-deltas
                         h
                         h-o-strengths
                         learning-rate)
        n-i-h-strengths (update-strengths
                         h-deltas
                         in
                         i-h-strengths
                         learning-rate)]
    [in n-i-h-strengths h n-h-o-strengths out]))

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
  (last (feed-forward input network)))





