

(define (problem BW-rand-18)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 )
(:init
(arm-empty)
(on b1 b16)
(on b2 b15)
(on b3 b9)
(on-table b4)
(on b5 b11)
(on b6 b1)
(on b7 b3)
(on b8 b4)
(on-table b9)
(on b10 b14)
(on b11 b17)
(on b12 b7)
(on b13 b12)
(on b14 b2)
(on-table b15)
(on b16 b8)
(on-table b17)
(on b18 b13)
(clear b5)
(clear b6)
(clear b10)
(clear b18)
)
(:goal
(and
(on b2 b17)
(on b3 b6)
(on b4 b3)
(on b5 b12)
(on b6 b16)
(on b8 b13)
(on b9 b1)
(on b11 b5)
(on b12 b2)
(on b13 b9)
(on b14 b18)
(on b16 b10)
(on b17 b8))
)
)


