

(define (problem BW-rand-23)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 )
(:init
(arm-empty)
(on-table b1)
(on b2 b23)
(on b3 b9)
(on b4 b3)
(on b5 b18)
(on b6 b2)
(on-table b7)
(on b8 b4)
(on b9 b14)
(on b10 b22)
(on b11 b19)
(on b12 b8)
(on b13 b7)
(on b14 b17)
(on b15 b10)
(on b16 b11)
(on b17 b13)
(on b18 b12)
(on b19 b15)
(on-table b20)
(on b21 b20)
(on-table b22)
(on b23 b16)
(clear b1)
(clear b5)
(clear b6)
(clear b21)
)
(:goal
(and
(on b1 b12)
(on b2 b9)
(on b4 b5)
(on b5 b23)
(on b6 b11)
(on b7 b6)
(on b8 b1)
(on b9 b4)
(on b10 b20)
(on b11 b2)
(on b12 b3)
(on b14 b18)
(on b15 b21)
(on b16 b10)
(on b17 b22)
(on b18 b17)
(on b19 b8)
(on b21 b13)
(on b22 b7))
)
)


