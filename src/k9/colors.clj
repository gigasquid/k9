(ns k9.colors)

;; [red orange yellow green blue purple]
(def red    [1 0 0 0 0 0])
(def orange [0 1 0 0 0 0])
(def yellow [0 0 1 0 0 0])
(def green  [0 0 0 1 0 0])
(def blue   [0 0 0 0 1 0])
(def purple [0 0 0 0 0 1])

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

   ;;Blue
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

(def color-training-data (partition 2 color-data))
