

(define (problem BW-rand-19)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 )
(:init
(arm-empty)
(on b1 b7)
(on b2 b6)
(on b3 b15)
(on b4 b1)
(on b5 b2)
(on b6 b13)
(on-table b7)
(on b8 b10)
(on b9 b11)
(on b10 b16)
(on b11 b4)
(on-table b12)
(on b13 b3)
(on b14 b9)
(on b15 b19)
(on b16 b17)
(on b17 b14)
(on-table b18)
(on b19 b8)
(clear b5)
(clear b12)
(clear b18)
)
(:goal
(and
(on b1 b3)
(on b2 b13)
(on b3 b6)
(on b4 b7)
(on b6 b16)
(on b7 b1)
(on b8 b15)
(on b9 b14)
(on b10 b9)
(on b11 b10)
(on b12 b18)
(on b14 b17)
(on b15 b5)
(on b16 b11)
(on b17 b8)
(on b18 b4)
(on b19 b2))
)
)


