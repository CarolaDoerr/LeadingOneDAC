

(define (problem BW-rand-18)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 )
(:init
(arm-empty)
(on b1 b2)
(on-table b2)
(on-table b3)
(on b4 b6)
(on b5 b1)
(on b6 b12)
(on b7 b17)
(on b8 b5)
(on b9 b3)
(on-table b10)
(on b11 b16)
(on b12 b9)
(on-table b13)
(on b14 b7)
(on b15 b13)
(on b16 b4)
(on b17 b8)
(on b18 b10)
(clear b11)
(clear b14)
(clear b15)
(clear b18)
)
(:goal
(and
(on b1 b16)
(on b2 b5)
(on b5 b11)
(on b6 b14)
(on b7 b15)
(on b8 b6)
(on b9 b8)
(on b10 b2)
(on b11 b18)
(on b12 b1)
(on b13 b9)
(on b15 b13)
(on b17 b3))
)
)

