

(define (problem BW-rand-15)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 )
(:init
(arm-empty)
(on b1 b6)
(on b2 b1)
(on b3 b11)
(on b4 b2)
(on b5 b9)
(on b6 b15)
(on b7 b12)
(on b8 b3)
(on b9 b7)
(on-table b10)
(on b11 b5)
(on b12 b10)
(on b13 b4)
(on b14 b13)
(on b15 b8)
(clear b14)
)
(:goal
(and
(on b2 b5)
(on b3 b7)
(on b4 b8)
(on b7 b10)
(on b8 b1)
(on b9 b14)
(on b10 b6)
(on b12 b9)
(on b13 b4)
(on b14 b15)
(on b15 b11))
)
)


