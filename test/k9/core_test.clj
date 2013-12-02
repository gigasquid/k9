(ns k9.core-test
  (:use clojure.test
        k9.core
        clojure.core.matrix.operators))

;;Neurons
;;  Input Hidden  Output
;;  A     1       C
;;  B     2       D
;;        3


;; Connection Strengths
;; Input to Hidden => [[A1 A2 A3] [B1 B2 B3]]
;; Hidden to Output => [[1C 1D] [2C 2D] [3C 3D]]


(def input-neurons [1 0])
(def input-hidden-strengths [ [0.12 0.2 0.13]
                              [0.01 0.02 0.03]])
(def hidden-neurons [0 0 0])
(def hidden-output-strengths [[0.15 0.16]
                              [0.02 0.03]
                              [0.01 0.02]])

(def output-neurons [0 0])

(deftest layer-test
  (testing "layer-activation-input"
   (is ( == [0.11942729853438588 0.197375320224904 0.12927258360605834]
            (layer-activation input-neurons input-hidden-strengths)))))

(def new-hidden-neurons
  [0.11942729853438588 0.197375320224904 0.12927258360605834])

(deftest layer-hidden-test
  (testing "layer-activation-hidden"
   (is ( == [0.02315019005321053 0.027608061500083565]
            (layer-activation new-hidden-neurons hidden-output-strengths)))))

(def new-output-neurons
  (layer-activation new-hidden-neurons hidden-output-strengths))


;; Calculate the errors
;; Desired output/ targets  will be the inverse of the input

;; Calculate the output layer first
; the error at the output neurons (Desired value – actual value) and multiplying it by the gradient of the sigmoid function

(def targets [0 1])

(deftest output-deltas-test
  (testing "output-deltas"
   (is (== [-0.023137783141771645 0.9716507764442904]
           (output-deltas targets new-output-neurons) ))))

(def odeltas (output-deltas targets new-output-neurons))


;; calc the errors for the hidden layers
;; for the hidden layer the error gradient for each hidden neuron is
;; the gradient of the activation function multiplied by the weighted
;; sum of the errors at the output layer originating from that neuron

(deftest hidden-deltas-test
  (testing "hidden-deltas"
   (is (== [0.14982559238071416 0.027569216735265096 0.018880751432503236]
           (hlayer-deltas
            odeltas
            new-hidden-neurons
            hidden-output-strengths)))))

(def hdeltas (hlayer-deltas
              odeltas
              new-hidden-neurons
              hidden-output-strengths))

;; Update the output weights
;; change = odelta * hidden value
;; weight = weight + learning-rate * change
(def learning-rate 0.2)

(deftest update-strengths-test
  (testing "update strengths for hidden"
   (is (== [[0.14996511847614283 0.20551384334705303 0.13377615028650064]
            [0.01 0.02 0.03]]
           (update-strengths
            hdeltas
            input-neurons
            input-hidden-strengths
            learning-rate)))))

(def new-hidden-output-strengths
  (update-strengths
       odeltas
       new-hidden-neurons
       hidden-output-strengths
       learning-rate))

(def new-input-hidden-strengths
  (update-strengths
       hdeltas
       input-neurons
       input-hidden-strengths
       learning-rate))

;;; now we just need to put all the pieces together

;; create a matrix
;; prop forward input / feed forward
;; calc errors
;; update weights

(def nn [ [0 0] input-hidden-strengths hidden-neurons hidden-output-strengths [0 0]])

(deftest feed-forward-test
  (testing "feed forward"
   (is (== [input-neurons input-hidden-strengths new-hidden-neurons hidden-output-strengths new-output-neurons]
           (feed-forward [1 0] nn)))))

(deftest update-weights-test
  (testing "update-weights"
   (is ( == [input-neurons
             new-input-hidden-strengths
             new-hidden-neurons
             new-hidden-output-strengths
             new-output-neurons]
            (update-weights (feed-forward [1 0] nn) [0 1] 0.2)))))

(deftest train-network-test
  (testing "train-network"
   (is (== [input-neurons
            new-input-hidden-strengths
            new-hidden-neurons
            new-hidden-output-strengths
            new-output-neurons]
           (train-network nn [1 0] [0 1] 0.2)))))

(deftest untrained-network-test
  (testing "untrainined network"
   (is (==  [0.02315019005321053 0.027608061500083565] (ff [1 0] nn)))))

(def n1 (-> nn
     (train-network [1 0] [0 1] 0.5)
     (train-network [0.5 0] [0 0.5] 0.5)
     (train-network [0.25 0] [0 0.25] 0.5)))

(deftest n1-test
  (testing "trained n1 network"
   (is (==  [0.03765676393050254 0.10552175312900794] (ff [1 0] n1)))))

(def n2 (train-data nn [
                        [[1 0] [0 1]]
                        [[0.5 0] [0 0.5]]
                        [[0.25 0] [0 0.25] ]]
                    0.5))

(deftest n2-test
  (testing "trained n2 network"
   (is (==  [0.03765676393050254 0.10552175312900794] (ff [1 0] n2)))))

;;untrained
(ff [1 0] nn) ;=> [0.02315019005321053 0.027608061500083565]
;;trained
(ff [1 0] n1) ;=> [0.03765676393050254 0.10552175312900794]

;; lazy training set of 1000 or something
(defn inverse-data []
  (let [n (rand 1)]
    [[n 0] [0 n]]))

(def n4 (train-data nn (repeatedly 5000 inverse-data) 0.2))


(ff [1 0] n4) ;=> [-4.954958580800465E-4 0.8160149309699489]
(ff [0.5 0] n4) ;=> [2.0984340203290242E-4 0.5577429793364974]
(ff [0.25 0] n4) ;=> [1.3457614704841535E-4 0.3061399408502212]
