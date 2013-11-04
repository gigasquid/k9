(ns k9.core)

;; 3 layer back propagation neural network
;; layer 1 - input layer will have RGB color ex: [255 0 0 ] FF0000
;; layer 2 - hidden layer
;; layer 3 - output layer -> either "Red" "Green" "Blue"


;;Our neuron values
;;Our weights
;;Our weight changes
;;Our error gradients

{:value 0 :weight 0 :delta-weight 0 :error 0}

;; input 3 neurons for R G B [255 0 0]

;;random weight -0.5 to 0.5
(defn rand-weight []
  (if (> (rand) 0.49)
    (rand 0.5)
    (rand -0.5)))

(defn gen-neuron []
  {:value 0 :weight (rand-weight) :delta-weight 0 :error 0})

(for [i (range 0 3)]  (gen-neuron))

(defn gen-network [n-input n-hidden n-output]
  [(vec (for [i (range 0 n-input)] (gen-neuron)))
   (vec (for [i (range 0 n-hidden)] (gen-neuron)))
   (vec (for [i (range 0 n-output)] (gen-neuron)))])

(defn feed-input [input network]
  (map #(assoc %1 :value %2) (first network) input))

(defn update-neuron [input neuron]
  (let [weight (:weight neuron)
        new-value (Math/tanh (* input weight))]
    (assoc neuron :value new-value)))

(defn feed-layer [in-layer layer]
  (let [in-values (map :value in-layer)
        sum-in (apply + in-values)]
    (map #(update-neuron sum-in %1) layer)))

(defn feed-forward [input network]
  (let [input-row (feed-input input network)
        hidden-layer (feed-layer input-row (second network))
        output-layer (feed-layer hidden-layer (last network))]
    [input-row hidden-layer output-layer]))

(defn dtanh [x]
  (- 1.0 (* x x)))

;; next step is backward propogating the errors

(dtanh 3)

(feed-forward [1 0 0] n1)

(def input (feed-input [1 0 1] n1))
(def hidden (second n1))
input




(first n1)
(def x [{:x 1} {:x 2} {:x 3}])
(def y [5 6 7])
(assoc {:x 2} :x 4)
(map #(assoc %1 :x %2) x y)
;; r g b -> hidden -> red? 
(def n1 (gen-network 3 3 1))
(second n1)

1 0
0 0
0 0

(defn activation [v]
  (Math/tanh 0.4))