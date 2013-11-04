(ns k9.jets)

;; Name	  Gang	Age	Education	Marital Status	Occupation
;; Robi   Jets	30's	College	Single	Pusher
;; Bill	  Jets	40's	College	Single	Pusher
;; Mike	  Jets	20's	H.S.	Single	Pusher
;; Joan	  Jets	20's	J.H.	Single	Pusher
;; Cath   Jets	20's	College	Married	Pusher
;; John	  Jets	20's	College	Divorced Pusher
;; Josh   Jets	20's	College	Single	Bookie
;; Bert	  Jets	20's	College	Single	Burglar
;; Marg   Sharks 30's	J.H	Married	Bookie
;; Janet  Sharks 20's	J.H.	Married	Bookie
;; Alfred Sharks 40's	H.S.	Married	Bookie
;; Gerry  Sharks 40's	College	Married	Bookie
;; Brett  Sharks 40's	J.H.	Single	Bookie
;; Sandra Sharks 40's	J.H	DivorcedBookie
;; Beth	  Sharks 40's	J.H.	Married	Pusher
;; Maria  Sharks 40's	J.H.	Married	Burglar

;; The network applied to this training data is composed of 12 binary
;; inputs (representing the different characteristics of gang
;; members), 4 hidden units, and 2 output units (Jets or Sharks)

;;Sample input is
;; 20's 30's 40's HS JH College Single Married Divorced Pusher Bookie Burglar
(def training-input-1 [0 1 0 0 0 1 1 0 0 1 0 0])
(count training-input-1)


;;random weight -0.5 to 0.5
(defn rand-weight []
  (if (> (rand) 0.49)
    (rand 0.5)
    (rand -0.5)))

(defn gen-neuron []
  {:value 0 :weight (rand-weight) :error 0})

(for [i (range 0 3)]  (gen-neuron))

(defn gen-network [n-hidden n-output]
  [(vec (for [i (range 0 n-hidden)] (gen-neuron)))
   (vec (for [i (range 0 n-output)] (gen-neuron)))])

(defn feed-input [input network]
  (map #(assoc %1 :value %2) (first network) input))

(defn update-neuron [input neuron]
  (let [weight (:weight neuron)
        new-value (Math/tanh (* input weight))]
    (assoc neuron :value new-value)))


(defn feed-layer [in-layer layer]
  (let [sum-in (apply + in-layer)]
    (map #(update-neuron sum-in %1) layer)))

(defn feed-forward [input network]
  (let [new-hidden-layer (feed-layer input (first network))
        new-hidden-values (map :value new-hidden-layer)
        new-output-layer (feed-layer new-hidden-values (last network))]
    [new-hidden-layer new-output-layer]))

(defn dtanh [y]
  (- 1 (* y y)))

(defn error-grad-output [val desired-val]
  (* (dtanh val) (- desired-val val)))

(defn error-grad-hidden [val weight-sum-output-errors]
  (* (dtanh val) weight-sum-output-errors))

;;; trying it out

(def network (gen-network 4 2))

(def v1 (feed-forward training-input-1 network))
;; the answer should be [1 0] for Jets

(defn update-error-output-neuron [neuron desired-val]
  (let [val (:value neuron)
        error (error-grad-output val desired-val)]
    (assoc neuron :error error)))

(defn backprop-error-output [network expected-vals]
  [(first network)
   (map update-error-output-neuron
        (last network) expected-vals)])

(defn weighted-sum-output-errors [output-layer]
   (reduce  #(+ %1 (* (:value %2) (:error %2)))
            0
            output-layer))

(defn update-error-hidden-neuron [neuron sum-oerrors]
  (let [val (:value neuron)
        error (error-grad-hidden val sum-oerrors)]
    (assoc neuron :error error)))


(defn backprop-error-hidden [network]
  (let [outputs (last network)
        hidden (first network)
        sum-oerrors (weighted-sum-output-errors (last network))
        new-hidden (map #(update-error-hidden-neuron %1 sum-oerrors) hidden)]
    [new-hidden outputs]))

(defn update-outer-weight [network]
  (let [outputs (last network)
        hiddens (first network)
        new-outputs ()]))


(defn back-propagate [network expected-vals]
  (let [update-out-errors (backprop-error-output network expected-vals)
        update-in-errors (backprop-error-hidden out-step)
        update-out-weights (reduce #(+ %1 (* (:error %2) 4)) 0 (last e2))
        ]))

(back-propagate v1 [1 0])
(def e2 (back-propagate v1 [1 0]))

(def e1 (backprop-error-output v1 [1 0]))
(backprop-error-hidden e1)


