

(define (problem BW-rand-21)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 )
(:init
(arm-empty)
(on b1 b11)
(on b2 b16)
(on b3 b2)
(on b4 b7)
(on-table b5)
(on b6 b10)
(on-table b7)
(on b8 b13)
(on b9 b15)
(on-table b10)
(on b11 b21)
(on b12 b9)
(on b13 b12)
(on b14 b4)
(on b15 b5)
(on-table b16)
(on b17 b14)
(on b18 b3)
(on-table b19)
(on b20 b17)
(on b21 b6)
(clear b1)
(clear b8)
(clear b18)
(clear b19)
(clear b20)
)
(:goal
(and
(on b1 b10)
(on b2 b12)
(on b3 b20)
(on b5 b13)
(on b7 b4)
(on b9 b21)
(on b10 b8)
(on b11 b2)
(on b12 b19)
(on b13 b11)
(on b14 b16)
(on b15 b7)
(on b16 b17)
(on b17 b1)
(on b18 b3)
(on b19 b15)
(on b20 b6)
(on b21 b14))
)
)


