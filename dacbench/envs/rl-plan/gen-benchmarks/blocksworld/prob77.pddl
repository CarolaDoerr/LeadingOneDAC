

(define (problem BW-rand-17)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 )
(:init
(arm-empty)
(on b1 b16)
(on-table b2)
(on b3 b4)
(on-table b4)
(on b5 b3)
(on-table b6)
(on b7 b15)
(on b8 b12)
(on b9 b5)
(on b10 b6)
(on b11 b13)
(on b12 b17)
(on b13 b7)
(on b14 b2)
(on b15 b8)
(on b16 b10)
(on b17 b9)
(clear b1)
(clear b11)
(clear b14)
)
(:goal
(and
(on b1 b16)
(on b3 b12)
(on b4 b3)
(on b5 b8)
(on b6 b15)
(on b7 b11)
(on b8 b10)
(on b9 b6)
(on b10 b2)
(on b12 b9)
(on b13 b1)
(on b14 b13)
(on b15 b5)
(on b16 b17)
(on b17 b4))
)
)


