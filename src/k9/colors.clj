(ns k9.colors
  (require [k9.simple :refer :all]))

;[red orange yellow green blue purple]
(def red    [255 0 0 0 0 0])
(def orange [0 255 0 0 0 0])
(def yellow [0 0 255 0 0 0])
(def green  [0 0 0 255 0 0])
(def blue   [0 0 0 0 255 0])
(def purple [0 0 0 0 0 255])



(def color-data
  [
   ;;Red
   [178 34 34] red
   [255 48 48] red
   [238 44 44] red
   [205 38 38] red
   [139 26 26]  red
   [255 0 0] red
   [238 0 0] red
   [205 0 0] red
   [139 0 0] red
   [192 0 0] red

   ;;Orange
   [255 140 0] orange
   [255 127 0] orange
   [238 118 0] orange
   [205 102 0] orange
   [255 127 0] orange
   [255 165 0] orange
   [255 165 0] orange
   [238 154 0] orange
   [205 133 0] orange

   ;;Yellow
   [255 255 0] yellow
   [238 238 0] yellow
   [205 205 0] yellow
   [139 139 0] yellow
   [255 215 0] yellow
   [255 215 0] yellow
   [238 201 0] yellow
   [205 173 0] yellow

   ;;Green
   [47 79 47] green
   [0 100 0]  green
   [84 255 159] green
   [78 238 148] green
   [67 205 128] green
   [0 255 127]  green
   [0 255 127] green
   [0 238 118] green
   [0 205 102] green
   [0 139 69] green

   ;Blue
   [0 206 209] blue
   [0 191 255] blue
   [0 191 255] blue
   [0 178 238] blue
   [0 154 205] blue
   [0 104 139] blue
   [30 144 255] blue
   [30 144 255] blue
   [28 134 238] blue
   [24 116 205] blue


   ;;Purple
   [147 112 219] purple
   [171 130 255] purple
   [153 50 205] purple
   [159 121 238] purple
   [137 104 205] purple
   [93 71 139] purple
   [153 50 204] purple
   [191 62 255] purple
   ]
  )

(defn normalize-input [v]
  (mapv #(/ % 255.0) v))

(def color-training-data
  (partition 2 (mapv #(normalize-input %) color-data)))


(def color-nn (construct-network 3 10 6))
(defn train-epochs [n network training-data learning-rate]
  (if (zero? n)
    network
    (recur (dec n)
           (train-data network training-data learning-rate)
           training-data
           learning-rate)))
;; before training
(ff (normalize-input [255 0 0]) color-nn)               ;=> .3

(def nc (train-epochs 100 color-nn color-training-data 0.2))

;; after training
(ff (normalize-input [255 0 0]) nc)
;; [0.3120025079502493
;;  0.2804061413007571
;;  0.22189516312865634
;;  -0.2829161956887886
;;  -0.19619313386399617
;;  0.14397479782011294]
(ff (normalize-input [255 165 0]) nc)
;; [0.13432569198660183
;;  0.38889956053093694
;;  0.43925753652865285
;;  0.17202453927040529
;;  -0.10084259239274129
;;  -0.07614255846373787]


(ff (normalize-input [255 255 0]) nc)
;; [0.0342977314573449
;;  0.43796103500087075
;;  0.5328782635091539
;;  0.3944921605278269
;;  -0.04531791432446118
;;  -0.18661447450991833]


(ff (normalize-input [0 255 127]) nc)
;; [-0.2610373377696042
;;  0.08079658578801456
;;  0.2322513500903176
;;  0.5657442955721695
;;  0.402844694651249
;;  0.0014065678913794128]


(ff (normalize-input [0 191 255]) nc)
;; [-0.175954481872805
;;  -0.08758149998678759
;;  -0.017156154063572772
;;  0.36086407446223373
;;  0.5757102688628224
;;  0.39584597583111636]


(ff (normalize-input [153 50 204]) nc)
;; [0.16830004463062614
;;  0.030995906642651293
;;  -0.025703206654810372
;;  -0.15499642639413008
;;  0.34272848204763423
;;  0.5169574243884011]



