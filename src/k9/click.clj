(ns k9.click)

;; you'll see this time it will work

;; input is  output
;;  World    World Bank
;; River     River
;; Bank     Earth

(def activation-function (fn [x] (Math/tanh x)))
(def activation-function-derivation (fn [y] (- 1.0 (* y y))))

(defn gen-neuron [id]
  {:id id  :val 0 })

(defn gen-hidden [n]
  (for [i (range n)]
    (gen-neuron (keyword (str "hidden" i)))))

(defn gen-connections [inputs hidden]
  (partition (count hidden)
             (for [x inputs  y hidden]
               [{:connection (keyword (str (name (:id x)) (name (:id y))))
                 :strength (rand (/ 1 (count hidden)))}])))

(def x [{:id :a :val 0} {:id :b :val 0} {:id :c :val 0}])
(def y [{:id 1 :val 0} {:id 2 :val 0} ])
(gen-connections x (gen-hidden 2))

(def a (map gen-neuron [:world :river :bank]))
(def b (gen-hidden (count a)))
(def c (gen-connections a b))
(def d (map gen-neuron [:worldbank :river :earth]))
(def e (gen-connections b d ))

;;ex (gen-network [:world :river :bank] [:worldbank :river :earth])
;; it generates the hidden networks too
(defn gen-network [inputs outputs]
  (let [in-neurons (map gen-neuron inputs)
        hidden-neurons (gen-hidden (count inputs))
        in-hidden (gen-connections in-neurons hidden-neurons)
        out-neurons (map gen-neuron outputs)
        hidden-out (gen-connections hidden-neurons out-neurons)]
    [in-neurons in-hidden hidden-neurons hidden-out out-neurons]))

(def nn (gen-network [:world :river :bank] [:worldbank :river :earth]))
nn

(defn assoc-inputs [in-values network]
  (map #(assoc %1 :val %2) (first network) in-values))

(def a (assoc-inputs [1 0 1] nn))
(def b (second nn))
(def c (nth nn 2))

(reduce + (map #(* (:val %1) 
          (:strength (first (first %2)))
          ) a b))

(map #(conj [] (:id %1)
            (:connection (first (second %2)))
            (:strength (first (second %2)))
            ) a b)
(defn ff-layers [ins connections outs]
  )

(second nn)

(defn feed-forward [in-values network]
  (let [inputs (assoc-inputs in-values network)]
    ))

