(ns k9.simple
  (:use clojure.test)
  (:use clojure.core.matrix)
  (:use clojure.core.matrix.operators))


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

(def activation-fn (fn [x] (Math/tanh x)))
(def dactivation-fn (fn [y] (- 1.0 (* y y))))

;; Propogate the input

(defn layer-activation [inputs strengths]
  "forward propogate the input of a layer"
  (mapv activation-fn
      (mapv #(reduce + %)
       (* inputs (transpose strengths)))))

(testing "layer-activation-input"
  (is ( == [0.11942729853438588 0.197375320224904 0.12927258360605834]
           (layer-activation input-neurons input-hidden-strengths))))

(def new-hidden-neurons
  [0.11942729853438588 0.197375320224904 0.12927258360605834])

(testing "layer-activation-hidden"
  (is ( == [0.02315019005321053 0.027608061500083565]
           (layer-activation new-hidden-neurons hidden-output-strengths))))

(def new-output-neurons
  (layer-activation new-hidden-neurons hidden-output-strengths))


;; Calculate the errors
;; Desired output/ targets  will be the inverse of the input

(def targets [0 1])
(* (mapv dactivation-fn new-output-neurons)
   (- targets new-output-neurons))

(defn output-deltas [targets outputs]
  "measures the delta errors for the output layer"
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))

(testing "output-deltas"
  (is (== [-0.023137783141771645 0.9716507764442904]
          (output-deltas targets new-output-neurons))))




