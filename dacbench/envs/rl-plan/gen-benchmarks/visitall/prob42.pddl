(define (problem grid-16)
(:domain grid-visit-all)
(:objects 
	loc-x0-y0
	loc-x0-y1
	loc-x0-y2
	loc-x0-y3
	loc-x0-y4
	loc-x0-y5
	loc-x0-y6
	loc-x0-y7
	loc-x0-y8
	loc-x0-y9
	loc-x0-y10
	loc-x0-y11
	loc-x0-y12
	loc-x0-y13
	loc-x0-y14
	loc-x0-y15
	loc-x1-y0
	loc-x1-y1
	loc-x1-y2
	loc-x1-y3
	loc-x1-y4
	loc-x1-y5
	loc-x1-y6
	loc-x1-y7
	loc-x1-y8
	loc-x1-y9
	loc-x1-y10
	loc-x1-y11
	loc-x1-y12
	loc-x1-y13
	loc-x1-y14
	loc-x1-y15
	loc-x2-y0
	loc-x2-y1
	loc-x2-y2
	loc-x2-y3
	loc-x2-y4
	loc-x2-y5
	loc-x2-y6
	loc-x2-y7
	loc-x2-y8
	loc-x2-y9
	loc-x2-y10
	loc-x2-y11
	loc-x2-y12
	loc-x2-y13
	loc-x2-y14
	loc-x2-y15
	loc-x3-y0
	loc-x3-y1
	loc-x3-y2
	loc-x3-y3
	loc-x3-y4
	loc-x3-y5
	loc-x3-y6
	loc-x3-y7
	loc-x3-y8
	loc-x3-y9
	loc-x3-y10
	loc-x3-y11
	loc-x3-y12
	loc-x3-y13
	loc-x3-y14
	loc-x3-y15
	loc-x4-y0
	loc-x4-y1
	loc-x4-y2
	loc-x4-y3
	loc-x4-y4
	loc-x4-y5
	loc-x4-y6
	loc-x4-y7
	loc-x4-y8
	loc-x4-y9
	loc-x4-y10
	loc-x4-y11
	loc-x4-y12
	loc-x4-y13
	loc-x4-y14
	loc-x4-y15
	loc-x5-y0
	loc-x5-y1
	loc-x5-y2
	loc-x5-y3
	loc-x5-y4
	loc-x5-y5
	loc-x5-y6
	loc-x5-y7
	loc-x5-y8
	loc-x5-y9
	loc-x5-y10
	loc-x5-y11
	loc-x5-y12
	loc-x5-y13
	loc-x5-y14
	loc-x5-y15
	loc-x6-y0
	loc-x6-y1
	loc-x6-y2
	loc-x6-y3
	loc-x6-y4
	loc-x6-y5
	loc-x6-y6
	loc-x6-y7
	loc-x6-y8
	loc-x6-y9
	loc-x6-y10
	loc-x6-y11
	loc-x6-y12
	loc-x6-y13
	loc-x6-y14
	loc-x6-y15
	loc-x7-y0
	loc-x7-y1
	loc-x7-y2
	loc-x7-y3
	loc-x7-y4
	loc-x7-y5
	loc-x7-y6
	loc-x7-y7
	loc-x7-y8
	loc-x7-y9
	loc-x7-y10
	loc-x7-y11
	loc-x7-y12
	loc-x7-y13
	loc-x7-y14
	loc-x7-y15
	loc-x8-y0
	loc-x8-y1
	loc-x8-y2
	loc-x8-y3
	loc-x8-y4
	loc-x8-y5
	loc-x8-y6
	loc-x8-y7
	loc-x8-y8
	loc-x8-y9
	loc-x8-y10
	loc-x8-y11
	loc-x8-y12
	loc-x8-y13
	loc-x8-y14
	loc-x8-y15
	loc-x9-y0
	loc-x9-y1
	loc-x9-y2
	loc-x9-y3
	loc-x9-y4
	loc-x9-y5
	loc-x9-y6
	loc-x9-y7
	loc-x9-y8
	loc-x9-y9
	loc-x9-y10
	loc-x9-y11
	loc-x9-y12
	loc-x9-y13
	loc-x9-y14
	loc-x9-y15
	loc-x10-y0
	loc-x10-y1
	loc-x10-y2
	loc-x10-y3
	loc-x10-y4
	loc-x10-y5
	loc-x10-y6
	loc-x10-y7
	loc-x10-y8
	loc-x10-y9
	loc-x10-y10
	loc-x10-y11
	loc-x10-y12
	loc-x10-y13
	loc-x10-y14
	loc-x10-y15
	loc-x11-y0
	loc-x11-y1
	loc-x11-y2
	loc-x11-y3
	loc-x11-y4
	loc-x11-y5
	loc-x11-y6
	loc-x11-y7
	loc-x11-y8
	loc-x11-y9
	loc-x11-y10
	loc-x11-y11
	loc-x11-y12
	loc-x11-y13
	loc-x11-y14
	loc-x11-y15
	loc-x12-y0
	loc-x12-y1
	loc-x12-y2
	loc-x12-y3
	loc-x12-y4
	loc-x12-y5
	loc-x12-y6
	loc-x12-y7
	loc-x12-y8
	loc-x12-y9
	loc-x12-y10
	loc-x12-y11
	loc-x12-y12
	loc-x12-y13
	loc-x12-y14
	loc-x12-y15
	loc-x13-y0
	loc-x13-y1
	loc-x13-y2
	loc-x13-y3
	loc-x13-y4
	loc-x13-y5
	loc-x13-y6
	loc-x13-y7
	loc-x13-y8
	loc-x13-y9
	loc-x13-y10
	loc-x13-y11
	loc-x13-y12
	loc-x13-y13
	loc-x13-y14
	loc-x13-y15
	loc-x14-y0
	loc-x14-y1
	loc-x14-y2
	loc-x14-y3
	loc-x14-y4
	loc-x14-y5
	loc-x14-y6
	loc-x14-y7
	loc-x14-y8
	loc-x14-y9
	loc-x14-y10
	loc-x14-y11
	loc-x14-y12
	loc-x14-y13
	loc-x14-y14
	loc-x14-y15
	loc-x15-y0
	loc-x15-y1
	loc-x15-y2
	loc-x15-y3
	loc-x15-y4
	loc-x15-y5
	loc-x15-y6
	loc-x15-y7
	loc-x15-y8
	loc-x15-y9
	loc-x15-y10
	loc-x15-y11
	loc-x15-y12
	loc-x15-y13
	loc-x15-y14
	loc-x15-y15
- place 
        
)
(:init
	(at-robot loc-x10-y12)
	(visited loc-x10-y12)
	(connected loc-x0-y0 loc-x1-y0)
 	(connected loc-x0-y0 loc-x0-y1)
 	(connected loc-x0-y1 loc-x1-y1)
 	(connected loc-x0-y1 loc-x0-y0)
 	(connected loc-x0-y1 loc-x0-y2)
 	(connected loc-x0-y2 loc-x1-y2)
 	(connected loc-x0-y2 loc-x0-y1)
 	(connected loc-x0-y2 loc-x0-y3)
 	(connected loc-x0-y3 loc-x1-y3)
 	(connected loc-x0-y3 loc-x0-y2)
 	(connected loc-x0-y3 loc-x0-y4)
 	(connected loc-x0-y4 loc-x1-y4)
 	(connected loc-x0-y4 loc-x0-y3)
 	(connected loc-x0-y4 loc-x0-y5)
 	(connected loc-x0-y5 loc-x1-y5)
 	(connected loc-x0-y5 loc-x0-y4)
 	(connected loc-x0-y5 loc-x0-y6)
 	(connected loc-x0-y6 loc-x1-y6)
 	(connected loc-x0-y6 loc-x0-y5)
 	(connected loc-x0-y6 loc-x0-y7)
 	(connected loc-x0-y7 loc-x1-y7)
 	(connected loc-x0-y7 loc-x0-y6)
 	(connected loc-x0-y7 loc-x0-y8)
 	(connected loc-x0-y8 loc-x1-y8)
 	(connected loc-x0-y8 loc-x0-y7)
 	(connected loc-x0-y8 loc-x0-y9)
 	(connected loc-x0-y9 loc-x1-y9)
 	(connected loc-x0-y9 loc-x0-y8)
 	(connected loc-x0-y9 loc-x0-y10)
 	(connected loc-x0-y10 loc-x1-y10)
 	(connected loc-x0-y10 loc-x0-y9)
 	(connected loc-x0-y10 loc-x0-y11)
 	(connected loc-x0-y11 loc-x1-y11)
 	(connected loc-x0-y11 loc-x0-y10)
 	(connected loc-x0-y11 loc-x0-y12)
 	(connected loc-x0-y12 loc-x1-y12)
 	(connected loc-x0-y12 loc-x0-y11)
 	(connected loc-x0-y12 loc-x0-y13)
 	(connected loc-x0-y13 loc-x1-y13)
 	(connected loc-x0-y13 loc-x0-y12)
 	(connected loc-x0-y13 loc-x0-y14)
 	(connected loc-x0-y14 loc-x1-y14)
 	(connected loc-x0-y14 loc-x0-y13)
 	(connected loc-x0-y14 loc-x0-y15)
 	(connected loc-x0-y15 loc-x1-y15)
 	(connected loc-x0-y15 loc-x0-y14)
 	(connected loc-x1-y0 loc-x0-y0)
 	(connected loc-x1-y0 loc-x2-y0)
 	(connected loc-x1-y0 loc-x1-y1)
 	(connected loc-x1-y1 loc-x0-y1)
 	(connected loc-x1-y1 loc-x2-y1)
 	(connected loc-x1-y1 loc-x1-y0)
 	(connected loc-x1-y1 loc-x1-y2)
 	(connected loc-x1-y2 loc-x0-y2)
 	(connected loc-x1-y2 loc-x2-y2)
 	(connected loc-x1-y2 loc-x1-y1)
 	(connected loc-x1-y2 loc-x1-y3)
 	(connected loc-x1-y3 loc-x0-y3)
 	(connected loc-x1-y3 loc-x2-y3)
 	(connected loc-x1-y3 loc-x1-y2)
 	(connected loc-x1-y3 loc-x1-y4)
 	(connected loc-x1-y4 loc-x0-y4)
 	(connected loc-x1-y4 loc-x2-y4)
 	(connected loc-x1-y4 loc-x1-y3)
 	(connected loc-x1-y4 loc-x1-y5)
 	(connected loc-x1-y5 loc-x0-y5)
 	(connected loc-x1-y5 loc-x2-y5)
 	(connected loc-x1-y5 loc-x1-y4)
 	(connected loc-x1-y5 loc-x1-y6)
 	(connected loc-x1-y6 loc-x0-y6)
 	(connected loc-x1-y6 loc-x2-y6)
 	(connected loc-x1-y6 loc-x1-y5)
 	(connected loc-x1-y6 loc-x1-y7)
 	(connected loc-x1-y7 loc-x0-y7)
 	(connected loc-x1-y7 loc-x2-y7)
 	(connected loc-x1-y7 loc-x1-y6)
 	(connected loc-x1-y7 loc-x1-y8)
 	(connected loc-x1-y8 loc-x0-y8)
 	(connected loc-x1-y8 loc-x2-y8)
 	(connected loc-x1-y8 loc-x1-y7)
 	(connected loc-x1-y8 loc-x1-y9)
 	(connected loc-x1-y9 loc-x0-y9)
 	(connected loc-x1-y9 loc-x2-y9)
 	(connected loc-x1-y9 loc-x1-y8)
 	(connected loc-x1-y9 loc-x1-y10)
 	(connected loc-x1-y10 loc-x0-y10)
 	(connected loc-x1-y10 loc-x2-y10)
 	(connected loc-x1-y10 loc-x1-y9)
 	(connected loc-x1-y10 loc-x1-y11)
 	(connected loc-x1-y11 loc-x0-y11)
 	(connected loc-x1-y11 loc-x2-y11)
 	(connected loc-x1-y11 loc-x1-y10)
 	(connected loc-x1-y11 loc-x1-y12)
 	(connected loc-x1-y12 loc-x0-y12)
 	(connected loc-x1-y12 loc-x2-y12)
 	(connected loc-x1-y12 loc-x1-y11)
 	(connected loc-x1-y12 loc-x1-y13)
 	(connected loc-x1-y13 loc-x0-y13)
 	(connected loc-x1-y13 loc-x2-y13)
 	(connected loc-x1-y13 loc-x1-y12)
 	(connected loc-x1-y13 loc-x1-y14)
 	(connected loc-x1-y14 loc-x0-y14)
 	(connected loc-x1-y14 loc-x2-y14)
 	(connected loc-x1-y14 loc-x1-y13)
 	(connected loc-x1-y14 loc-x1-y15)
 	(connected loc-x1-y15 loc-x0-y15)
 	(connected loc-x1-y15 loc-x2-y15)
 	(connected loc-x1-y15 loc-x1-y14)
 	(connected loc-x2-y0 loc-x1-y0)
 	(connected loc-x2-y0 loc-x3-y0)
 	(connected loc-x2-y0 loc-x2-y1)
 	(connected loc-x2-y1 loc-x1-y1)
 	(connected loc-x2-y1 loc-x3-y1)
 	(connected loc-x2-y1 loc-x2-y0)
 	(connected loc-x2-y1 loc-x2-y2)
 	(connected loc-x2-y2 loc-x1-y2)
 	(connected loc-x2-y2 loc-x3-y2)
 	(connected loc-x2-y2 loc-x2-y1)
 	(connected loc-x2-y2 loc-x2-y3)
 	(connected loc-x2-y3 loc-x1-y3)
 	(connected loc-x2-y3 loc-x3-y3)
 	(connected loc-x2-y3 loc-x2-y2)
 	(connected loc-x2-y3 loc-x2-y4)
 	(connected loc-x2-y4 loc-x1-y4)
 	(connected loc-x2-y4 loc-x3-y4)
 	(connected loc-x2-y4 loc-x2-y3)
 	(connected loc-x2-y4 loc-x2-y5)
 	(connected loc-x2-y5 loc-x1-y5)
 	(connected loc-x2-y5 loc-x3-y5)
 	(connected loc-x2-y5 loc-x2-y4)
 	(connected loc-x2-y5 loc-x2-y6)
 	(connected loc-x2-y6 loc-x1-y6)
 	(connected loc-x2-y6 loc-x3-y6)
 	(connected loc-x2-y6 loc-x2-y5)
 	(connected loc-x2-y6 loc-x2-y7)
 	(connected loc-x2-y7 loc-x1-y7)
 	(connected loc-x2-y7 loc-x3-y7)
 	(connected loc-x2-y7 loc-x2-y6)
 	(connected loc-x2-y7 loc-x2-y8)
 	(connected loc-x2-y8 loc-x1-y8)
 	(connected loc-x2-y8 loc-x3-y8)
 	(connected loc-x2-y8 loc-x2-y7)
 	(connected loc-x2-y8 loc-x2-y9)
 	(connected loc-x2-y9 loc-x1-y9)
 	(connected loc-x2-y9 loc-x3-y9)
 	(connected loc-x2-y9 loc-x2-y8)
 	(connected loc-x2-y9 loc-x2-y10)
 	(connected loc-x2-y10 loc-x1-y10)
 	(connected loc-x2-y10 loc-x3-y10)
 	(connected loc-x2-y10 loc-x2-y9)
 	(connected loc-x2-y10 loc-x2-y11)
 	(connected loc-x2-y11 loc-x1-y11)
 	(connected loc-x2-y11 loc-x3-y11)
 	(connected loc-x2-y11 loc-x2-y10)
 	(connected loc-x2-y11 loc-x2-y12)
 	(connected loc-x2-y12 loc-x1-y12)
 	(connected loc-x2-y12 loc-x3-y12)
 	(connected loc-x2-y12 loc-x2-y11)
 	(connected loc-x2-y12 loc-x2-y13)
 	(connected loc-x2-y13 loc-x1-y13)
 	(connected loc-x2-y13 loc-x3-y13)
 	(connected loc-x2-y13 loc-x2-y12)
 	(connected loc-x2-y13 loc-x2-y14)
 	(connected loc-x2-y14 loc-x1-y14)
 	(connected loc-x2-y14 loc-x3-y14)
 	(connected loc-x2-y14 loc-x2-y13)
 	(connected loc-x2-y14 loc-x2-y15)
 	(connected loc-x2-y15 loc-x1-y15)
 	(connected loc-x2-y15 loc-x3-y15)
 	(connected loc-x2-y15 loc-x2-y14)
 	(connected loc-x3-y0 loc-x2-y0)
 	(connected loc-x3-y0 loc-x4-y0)
 	(connected loc-x3-y0 loc-x3-y1)
 	(connected loc-x3-y1 loc-x2-y1)
 	(connected loc-x3-y1 loc-x4-y1)
 	(connected loc-x3-y1 loc-x3-y0)
 	(connected loc-x3-y1 loc-x3-y2)
 	(connected loc-x3-y2 loc-x2-y2)
 	(connected loc-x3-y2 loc-x4-y2)
 	(connected loc-x3-y2 loc-x3-y1)
 	(connected loc-x3-y2 loc-x3-y3)
 	(connected loc-x3-y3 loc-x2-y3)
 	(connected loc-x3-y3 loc-x4-y3)
 	(connected loc-x3-y3 loc-x3-y2)
 	(connected loc-x3-y3 loc-x3-y4)
 	(connected loc-x3-y4 loc-x2-y4)
 	(connected loc-x3-y4 loc-x4-y4)
 	(connected loc-x3-y4 loc-x3-y3)
 	(connected loc-x3-y4 loc-x3-y5)
 	(connected loc-x3-y5 loc-x2-y5)
 	(connected loc-x3-y5 loc-x4-y5)
 	(connected loc-x3-y5 loc-x3-y4)
 	(connected loc-x3-y5 loc-x3-y6)
 	(connected loc-x3-y6 loc-x2-y6)
 	(connected loc-x3-y6 loc-x4-y6)
 	(connected loc-x3-y6 loc-x3-y5)
 	(connected loc-x3-y6 loc-x3-y7)
 	(connected loc-x3-y7 loc-x2-y7)
 	(connected loc-x3-y7 loc-x4-y7)
 	(connected loc-x3-y7 loc-x3-y6)
 	(connected loc-x3-y7 loc-x3-y8)
 	(connected loc-x3-y8 loc-x2-y8)
 	(connected loc-x3-y8 loc-x4-y8)
 	(connected loc-x3-y8 loc-x3-y7)
 	(connected loc-x3-y8 loc-x3-y9)
 	(connected loc-x3-y9 loc-x2-y9)
 	(connected loc-x3-y9 loc-x4-y9)
 	(connected loc-x3-y9 loc-x3-y8)
 	(connected loc-x3-y9 loc-x3-y10)
 	(connected loc-x3-y10 loc-x2-y10)
 	(connected loc-x3-y10 loc-x4-y10)
 	(connected loc-x3-y10 loc-x3-y9)
 	(connected loc-x3-y10 loc-x3-y11)
 	(connected loc-x3-y11 loc-x2-y11)
 	(connected loc-x3-y11 loc-x4-y11)
 	(connected loc-x3-y11 loc-x3-y10)
 	(connected loc-x3-y11 loc-x3-y12)
 	(connected loc-x3-y12 loc-x2-y12)
 	(connected loc-x3-y12 loc-x4-y12)
 	(connected loc-x3-y12 loc-x3-y11)
 	(connected loc-x3-y12 loc-x3-y13)
 	(connected loc-x3-y13 loc-x2-y13)
 	(connected loc-x3-y13 loc-x4-y13)
 	(connected loc-x3-y13 loc-x3-y12)
 	(connected loc-x3-y13 loc-x3-y14)
 	(connected loc-x3-y14 loc-x2-y14)
 	(connected loc-x3-y14 loc-x4-y14)
 	(connected loc-x3-y14 loc-x3-y13)
 	(connected loc-x3-y14 loc-x3-y15)
 	(connected loc-x3-y15 loc-x2-y15)
 	(connected loc-x3-y15 loc-x4-y15)
 	(connected loc-x3-y15 loc-x3-y14)
 	(connected loc-x4-y0 loc-x3-y0)
 	(connected loc-x4-y0 loc-x5-y0)
 	(connected loc-x4-y0 loc-x4-y1)
 	(connected loc-x4-y1 loc-x3-y1)
 	(connected loc-x4-y1 loc-x5-y1)
 	(connected loc-x4-y1 loc-x4-y0)
 	(connected loc-x4-y1 loc-x4-y2)
 	(connected loc-x4-y2 loc-x3-y2)
 	(connected loc-x4-y2 loc-x5-y2)
 	(connected loc-x4-y2 loc-x4-y1)
 	(connected loc-x4-y2 loc-x4-y3)
 	(connected loc-x4-y3 loc-x3-y3)
 	(connected loc-x4-y3 loc-x5-y3)
 	(connected loc-x4-y3 loc-x4-y2)
 	(connected loc-x4-y3 loc-x4-y4)
 	(connected loc-x4-y4 loc-x3-y4)
 	(connected loc-x4-y4 loc-x5-y4)
 	(connected loc-x4-y4 loc-x4-y3)
 	(connected loc-x4-y4 loc-x4-y5)
 	(connected loc-x4-y5 loc-x3-y5)
 	(connected loc-x4-y5 loc-x5-y5)
 	(connected loc-x4-y5 loc-x4-y4)
 	(connected loc-x4-y5 loc-x4-y6)
 	(connected loc-x4-y6 loc-x3-y6)
 	(connected loc-x4-y6 loc-x5-y6)
 	(connected loc-x4-y6 loc-x4-y5)
 	(connected loc-x4-y6 loc-x4-y7)
 	(connected loc-x4-y7 loc-x3-y7)
 	(connected loc-x4-y7 loc-x5-y7)
 	(connected loc-x4-y7 loc-x4-y6)
 	(connected loc-x4-y7 loc-x4-y8)
 	(connected loc-x4-y8 loc-x3-y8)
 	(connected loc-x4-y8 loc-x5-y8)
 	(connected loc-x4-y8 loc-x4-y7)
 	(connected loc-x4-y8 loc-x4-y9)
 	(connected loc-x4-y9 loc-x3-y9)
 	(connected loc-x4-y9 loc-x5-y9)
 	(connected loc-x4-y9 loc-x4-y8)
 	(connected loc-x4-y9 loc-x4-y10)
 	(connected loc-x4-y10 loc-x3-y10)
 	(connected loc-x4-y10 loc-x5-y10)
 	(connected loc-x4-y10 loc-x4-y9)
 	(connected loc-x4-y10 loc-x4-y11)
 	(connected loc-x4-y11 loc-x3-y11)
 	(connected loc-x4-y11 loc-x5-y11)
 	(connected loc-x4-y11 loc-x4-y10)
 	(connected loc-x4-y11 loc-x4-y12)
 	(connected loc-x4-y12 loc-x3-y12)
 	(connected loc-x4-y12 loc-x5-y12)
 	(connected loc-x4-y12 loc-x4-y11)
 	(connected loc-x4-y12 loc-x4-y13)
 	(connected loc-x4-y13 loc-x3-y13)
 	(connected loc-x4-y13 loc-x5-y13)
 	(connected loc-x4-y13 loc-x4-y12)
 	(connected loc-x4-y13 loc-x4-y14)
 	(connected loc-x4-y14 loc-x3-y14)
 	(connected loc-x4-y14 loc-x5-y14)
 	(connected loc-x4-y14 loc-x4-y13)
 	(connected loc-x4-y14 loc-x4-y15)
 	(connected loc-x4-y15 loc-x3-y15)
 	(connected loc-x4-y15 loc-x5-y15)
 	(connected loc-x4-y15 loc-x4-y14)
 	(connected loc-x5-y0 loc-x4-y0)
 	(connected loc-x5-y0 loc-x6-y0)
 	(connected loc-x5-y0 loc-x5-y1)
 	(connected loc-x5-y1 loc-x4-y1)
 	(connected loc-x5-y1 loc-x6-y1)
 	(connected loc-x5-y1 loc-x5-y0)
 	(connected loc-x5-y1 loc-x5-y2)
 	(connected loc-x5-y2 loc-x4-y2)
 	(connected loc-x5-y2 loc-x6-y2)
 	(connected loc-x5-y2 loc-x5-y1)
 	(connected loc-x5-y2 loc-x5-y3)
 	(connected loc-x5-y3 loc-x4-y3)
 	(connected loc-x5-y3 loc-x6-y3)
 	(connected loc-x5-y3 loc-x5-y2)
 	(connected loc-x5-y3 loc-x5-y4)
 	(connected loc-x5-y4 loc-x4-y4)
 	(connected loc-x5-y4 loc-x6-y4)
 	(connected loc-x5-y4 loc-x5-y3)
 	(connected loc-x5-y4 loc-x5-y5)
 	(connected loc-x5-y5 loc-x4-y5)
 	(connected loc-x5-y5 loc-x6-y5)
 	(connected loc-x5-y5 loc-x5-y4)
 	(connected loc-x5-y5 loc-x5-y6)
 	(connected loc-x5-y6 loc-x4-y6)
 	(connected loc-x5-y6 loc-x6-y6)
 	(connected loc-x5-y6 loc-x5-y5)
 	(connected loc-x5-y6 loc-x5-y7)
 	(connected loc-x5-y7 loc-x4-y7)
 	(connected loc-x5-y7 loc-x6-y7)
 	(connected loc-x5-y7 loc-x5-y6)
 	(connected loc-x5-y7 loc-x5-y8)
 	(connected loc-x5-y8 loc-x4-y8)
 	(connected loc-x5-y8 loc-x6-y8)
 	(connected loc-x5-y8 loc-x5-y7)
 	(connected loc-x5-y8 loc-x5-y9)
 	(connected loc-x5-y9 loc-x4-y9)
 	(connected loc-x5-y9 loc-x6-y9)
 	(connected loc-x5-y9 loc-x5-y8)
 	(connected loc-x5-y9 loc-x5-y10)
 	(connected loc-x5-y10 loc-x4-y10)
 	(connected loc-x5-y10 loc-x6-y10)
 	(connected loc-x5-y10 loc-x5-y9)
 	(connected loc-x5-y10 loc-x5-y11)
 	(connected loc-x5-y11 loc-x4-y11)
 	(connected loc-x5-y11 loc-x6-y11)
 	(connected loc-x5-y11 loc-x5-y10)
 	(connected loc-x5-y11 loc-x5-y12)
 	(connected loc-x5-y12 loc-x4-y12)
 	(connected loc-x5-y12 loc-x6-y12)
 	(connected loc-x5-y12 loc-x5-y11)
 	(connected loc-x5-y12 loc-x5-y13)
 	(connected loc-x5-y13 loc-x4-y13)
 	(connected loc-x5-y13 loc-x6-y13)
 	(connected loc-x5-y13 loc-x5-y12)
 	(connected loc-x5-y13 loc-x5-y14)
 	(connected loc-x5-y14 loc-x4-y14)
 	(connected loc-x5-y14 loc-x6-y14)
 	(connected loc-x5-y14 loc-x5-y13)
 	(connected loc-x5-y14 loc-x5-y15)
 	(connected loc-x5-y15 loc-x4-y15)
 	(connected loc-x5-y15 loc-x6-y15)
 	(connected loc-x5-y15 loc-x5-y14)
 	(connected loc-x6-y0 loc-x5-y0)
 	(connected loc-x6-y0 loc-x7-y0)
 	(connected loc-x6-y0 loc-x6-y1)
 	(connected loc-x6-y1 loc-x5-y1)
 	(connected loc-x6-y1 loc-x7-y1)
 	(connected loc-x6-y1 loc-x6-y0)
 	(connected loc-x6-y1 loc-x6-y2)
 	(connected loc-x6-y2 loc-x5-y2)
 	(connected loc-x6-y2 loc-x7-y2)
 	(connected loc-x6-y2 loc-x6-y1)
 	(connected loc-x6-y2 loc-x6-y3)
 	(connected loc-x6-y3 loc-x5-y3)
 	(connected loc-x6-y3 loc-x7-y3)
 	(connected loc-x6-y3 loc-x6-y2)
 	(connected loc-x6-y3 loc-x6-y4)
 	(connected loc-x6-y4 loc-x5-y4)
 	(connected loc-x6-y4 loc-x7-y4)
 	(connected loc-x6-y4 loc-x6-y3)
 	(connected loc-x6-y4 loc-x6-y5)
 	(connected loc-x6-y5 loc-x5-y5)
 	(connected loc-x6-y5 loc-x7-y5)
 	(connected loc-x6-y5 loc-x6-y4)
 	(connected loc-x6-y5 loc-x6-y6)
 	(connected loc-x6-y6 loc-x5-y6)
 	(connected loc-x6-y6 loc-x7-y6)
 	(connected loc-x6-y6 loc-x6-y5)
 	(connected loc-x6-y6 loc-x6-y7)
 	(connected loc-x6-y7 loc-x5-y7)
 	(connected loc-x6-y7 loc-x7-y7)
 	(connected loc-x6-y7 loc-x6-y6)
 	(connected loc-x6-y7 loc-x6-y8)
 	(connected loc-x6-y8 loc-x5-y8)
 	(connected loc-x6-y8 loc-x7-y8)
 	(connected loc-x6-y8 loc-x6-y7)
 	(connected loc-x6-y8 loc-x6-y9)
 	(connected loc-x6-y9 loc-x5-y9)
 	(connected loc-x6-y9 loc-x7-y9)
 	(connected loc-x6-y9 loc-x6-y8)
 	(connected loc-x6-y9 loc-x6-y10)
 	(connected loc-x6-y10 loc-x5-y10)
 	(connected loc-x6-y10 loc-x7-y10)
 	(connected loc-x6-y10 loc-x6-y9)
 	(connected loc-x6-y10 loc-x6-y11)
 	(connected loc-x6-y11 loc-x5-y11)
 	(connected loc-x6-y11 loc-x7-y11)
 	(connected loc-x6-y11 loc-x6-y10)
 	(connected loc-x6-y11 loc-x6-y12)
 	(connected loc-x6-y12 loc-x5-y12)
 	(connected loc-x6-y12 loc-x7-y12)
 	(connected loc-x6-y12 loc-x6-y11)
 	(connected loc-x6-y12 loc-x6-y13)
 	(connected loc-x6-y13 loc-x5-y13)
 	(connected loc-x6-y13 loc-x7-y13)
 	(connected loc-x6-y13 loc-x6-y12)
 	(connected loc-x6-y13 loc-x6-y14)
 	(connected loc-x6-y14 loc-x5-y14)
 	(connected loc-x6-y14 loc-x7-y14)
 	(connected loc-x6-y14 loc-x6-y13)
 	(connected loc-x6-y14 loc-x6-y15)
 	(connected loc-x6-y15 loc-x5-y15)
 	(connected loc-x6-y15 loc-x7-y15)
 	(connected loc-x6-y15 loc-x6-y14)
 	(connected loc-x7-y0 loc-x6-y0)
 	(connected loc-x7-y0 loc-x8-y0)
 	(connected loc-x7-y0 loc-x7-y1)
 	(connected loc-x7-y1 loc-x6-y1)
 	(connected loc-x7-y1 loc-x8-y1)
 	(connected loc-x7-y1 loc-x7-y0)
 	(connected loc-x7-y1 loc-x7-y2)
 	(connected loc-x7-y2 loc-x6-y2)
 	(connected loc-x7-y2 loc-x8-y2)
 	(connected loc-x7-y2 loc-x7-y1)
 	(connected loc-x7-y2 loc-x7-y3)
 	(connected loc-x7-y3 loc-x6-y3)
 	(connected loc-x7-y3 loc-x8-y3)
 	(connected loc-x7-y3 loc-x7-y2)
 	(connected loc-x7-y3 loc-x7-y4)
 	(connected loc-x7-y4 loc-x6-y4)
 	(connected loc-x7-y4 loc-x8-y4)
 	(connected loc-x7-y4 loc-x7-y3)
 	(connected loc-x7-y4 loc-x7-y5)
 	(connected loc-x7-y5 loc-x6-y5)
 	(connected loc-x7-y5 loc-x8-y5)
 	(connected loc-x7-y5 loc-x7-y4)
 	(connected loc-x7-y5 loc-x7-y6)
 	(connected loc-x7-y6 loc-x6-y6)
 	(connected loc-x7-y6 loc-x8-y6)
 	(connected loc-x7-y6 loc-x7-y5)
 	(connected loc-x7-y6 loc-x7-y7)
 	(connected loc-x7-y7 loc-x6-y7)
 	(connected loc-x7-y7 loc-x8-y7)
 	(connected loc-x7-y7 loc-x7-y6)
 	(connected loc-x7-y7 loc-x7-y8)
 	(connected loc-x7-y8 loc-x6-y8)
 	(connected loc-x7-y8 loc-x8-y8)
 	(connected loc-x7-y8 loc-x7-y7)
 	(connected loc-x7-y8 loc-x7-y9)
 	(connected loc-x7-y9 loc-x6-y9)
 	(connected loc-x7-y9 loc-x8-y9)
 	(connected loc-x7-y9 loc-x7-y8)
 	(connected loc-x7-y9 loc-x7-y10)
 	(connected loc-x7-y10 loc-x6-y10)
 	(connected loc-x7-y10 loc-x8-y10)
 	(connected loc-x7-y10 loc-x7-y9)
 	(connected loc-x7-y10 loc-x7-y11)
 	(connected loc-x7-y11 loc-x6-y11)
 	(connected loc-x7-y11 loc-x8-y11)
 	(connected loc-x7-y11 loc-x7-y10)
 	(connected loc-x7-y11 loc-x7-y12)
 	(connected loc-x7-y12 loc-x6-y12)
 	(connected loc-x7-y12 loc-x8-y12)
 	(connected loc-x7-y12 loc-x7-y11)
 	(connected loc-x7-y12 loc-x7-y13)
 	(connected loc-x7-y13 loc-x6-y13)
 	(connected loc-x7-y13 loc-x8-y13)
 	(connected loc-x7-y13 loc-x7-y12)
 	(connected loc-x7-y13 loc-x7-y14)
 	(connected loc-x7-y14 loc-x6-y14)
 	(connected loc-x7-y14 loc-x8-y14)
 	(connected loc-x7-y14 loc-x7-y13)
 	(connected loc-x7-y14 loc-x7-y15)
 	(connected loc-x7-y15 loc-x6-y15)
 	(connected loc-x7-y15 loc-x8-y15)
 	(connected loc-x7-y15 loc-x7-y14)
 	(connected loc-x8-y0 loc-x7-y0)
 	(connected loc-x8-y0 loc-x9-y0)
 	(connected loc-x8-y0 loc-x8-y1)
 	(connected loc-x8-y1 loc-x7-y1)
 	(connected loc-x8-y1 loc-x9-y1)
 	(connected loc-x8-y1 loc-x8-y0)
 	(connected loc-x8-y1 loc-x8-y2)
 	(connected loc-x8-y2 loc-x7-y2)
 	(connected loc-x8-y2 loc-x9-y2)
 	(connected loc-x8-y2 loc-x8-y1)
 	(connected loc-x8-y2 loc-x8-y3)
 	(connected loc-x8-y3 loc-x7-y3)
 	(connected loc-x8-y3 loc-x9-y3)
 	(connected loc-x8-y3 loc-x8-y2)
 	(connected loc-x8-y3 loc-x8-y4)
 	(connected loc-x8-y4 loc-x7-y4)
 	(connected loc-x8-y4 loc-x9-y4)
 	(connected loc-x8-y4 loc-x8-y3)
 	(connected loc-x8-y4 loc-x8-y5)
 	(connected loc-x8-y5 loc-x7-y5)
 	(connected loc-x8-y5 loc-x9-y5)
 	(connected loc-x8-y5 loc-x8-y4)
 	(connected loc-x8-y5 loc-x8-y6)
 	(connected loc-x8-y6 loc-x7-y6)
 	(connected loc-x8-y6 loc-x9-y6)
 	(connected loc-x8-y6 loc-x8-y5)
 	(connected loc-x8-y6 loc-x8-y7)
 	(connected loc-x8-y7 loc-x7-y7)
 	(connected loc-x8-y7 loc-x9-y7)
 	(connected loc-x8-y7 loc-x8-y6)
 	(connected loc-x8-y7 loc-x8-y8)
 	(connected loc-x8-y8 loc-x7-y8)
 	(connected loc-x8-y8 loc-x9-y8)
 	(connected loc-x8-y8 loc-x8-y7)
 	(connected loc-x8-y8 loc-x8-y9)
 	(connected loc-x8-y9 loc-x7-y9)
 	(connected loc-x8-y9 loc-x9-y9)
 	(connected loc-x8-y9 loc-x8-y8)
 	(connected loc-x8-y9 loc-x8-y10)
 	(connected loc-x8-y10 loc-x7-y10)
 	(connected loc-x8-y10 loc-x9-y10)
 	(connected loc-x8-y10 loc-x8-y9)
 	(connected loc-x8-y10 loc-x8-y11)
 	(connected loc-x8-y11 loc-x7-y11)
 	(connected loc-x8-y11 loc-x9-y11)
 	(connected loc-x8-y11 loc-x8-y10)
 	(connected loc-x8-y11 loc-x8-y12)
 	(connected loc-x8-y12 loc-x7-y12)
 	(connected loc-x8-y12 loc-x9-y12)
 	(connected loc-x8-y12 loc-x8-y11)
 	(connected loc-x8-y12 loc-x8-y13)
 	(connected loc-x8-y13 loc-x7-y13)
 	(connected loc-x8-y13 loc-x9-y13)
 	(connected loc-x8-y13 loc-x8-y12)
 	(connected loc-x8-y13 loc-x8-y14)
 	(connected loc-x8-y14 loc-x7-y14)
 	(connected loc-x8-y14 loc-x9-y14)
 	(connected loc-x8-y14 loc-x8-y13)
 	(connected loc-x8-y14 loc-x8-y15)
 	(connected loc-x8-y15 loc-x7-y15)
 	(connected loc-x8-y15 loc-x9-y15)
 	(connected loc-x8-y15 loc-x8-y14)
 	(connected loc-x9-y0 loc-x8-y0)
 	(connected loc-x9-y0 loc-x10-y0)
 	(connected loc-x9-y0 loc-x9-y1)
 	(connected loc-x9-y1 loc-x8-y1)
 	(connected loc-x9-y1 loc-x10-y1)
 	(connected loc-x9-y1 loc-x9-y0)
 	(connected loc-x9-y1 loc-x9-y2)
 	(connected loc-x9-y2 loc-x8-y2)
 	(connected loc-x9-y2 loc-x10-y2)
 	(connected loc-x9-y2 loc-x9-y1)
 	(connected loc-x9-y2 loc-x9-y3)
 	(connected loc-x9-y3 loc-x8-y3)
 	(connected loc-x9-y3 loc-x10-y3)
 	(connected loc-x9-y3 loc-x9-y2)
 	(connected loc-x9-y3 loc-x9-y4)
 	(connected loc-x9-y4 loc-x8-y4)
 	(connected loc-x9-y4 loc-x10-y4)
 	(connected loc-x9-y4 loc-x9-y3)
 	(connected loc-x9-y4 loc-x9-y5)
 	(connected loc-x9-y5 loc-x8-y5)
 	(connected loc-x9-y5 loc-x10-y5)
 	(connected loc-x9-y5 loc-x9-y4)
 	(connected loc-x9-y5 loc-x9-y6)
 	(connected loc-x9-y6 loc-x8-y6)
 	(connected loc-x9-y6 loc-x10-y6)
 	(connected loc-x9-y6 loc-x9-y5)
 	(connected loc-x9-y6 loc-x9-y7)
 	(connected loc-x9-y7 loc-x8-y7)
 	(connected loc-x9-y7 loc-x10-y7)
 	(connected loc-x9-y7 loc-x9-y6)
 	(connected loc-x9-y7 loc-x9-y8)
 	(connected loc-x9-y8 loc-x8-y8)
 	(connected loc-x9-y8 loc-x10-y8)
 	(connected loc-x9-y8 loc-x9-y7)
 	(connected loc-x9-y8 loc-x9-y9)
 	(connected loc-x9-y9 loc-x8-y9)
 	(connected loc-x9-y9 loc-x10-y9)
 	(connected loc-x9-y9 loc-x9-y8)
 	(connected loc-x9-y9 loc-x9-y10)
 	(connected loc-x9-y10 loc-x8-y10)
 	(connected loc-x9-y10 loc-x10-y10)
 	(connected loc-x9-y10 loc-x9-y9)
 	(connected loc-x9-y10 loc-x9-y11)
 	(connected loc-x9-y11 loc-x8-y11)
 	(connected loc-x9-y11 loc-x10-y11)
 	(connected loc-x9-y11 loc-x9-y10)
 	(connected loc-x9-y11 loc-x9-y12)
 	(connected loc-x9-y12 loc-x8-y12)
 	(connected loc-x9-y12 loc-x10-y12)
 	(connected loc-x9-y12 loc-x9-y11)
 	(connected loc-x9-y12 loc-x9-y13)
 	(connected loc-x9-y13 loc-x8-y13)
 	(connected loc-x9-y13 loc-x10-y13)
 	(connected loc-x9-y13 loc-x9-y12)
 	(connected loc-x9-y13 loc-x9-y14)
 	(connected loc-x9-y14 loc-x8-y14)
 	(connected loc-x9-y14 loc-x10-y14)
 	(connected loc-x9-y14 loc-x9-y13)
 	(connected loc-x9-y14 loc-x9-y15)
 	(connected loc-x9-y15 loc-x8-y15)
 	(connected loc-x9-y15 loc-x10-y15)
 	(connected loc-x9-y15 loc-x9-y14)
 	(connected loc-x10-y0 loc-x9-y0)
 	(connected loc-x10-y0 loc-x11-y0)
 	(connected loc-x10-y0 loc-x10-y1)
 	(connected loc-x10-y1 loc-x9-y1)
 	(connected loc-x10-y1 loc-x11-y1)
 	(connected loc-x10-y1 loc-x10-y0)
 	(connected loc-x10-y1 loc-x10-y2)
 	(connected loc-x10-y2 loc-x9-y2)
 	(connected loc-x10-y2 loc-x11-y2)
 	(connected loc-x10-y2 loc-x10-y1)
 	(connected loc-x10-y2 loc-x10-y3)
 	(connected loc-x10-y3 loc-x9-y3)
 	(connected loc-x10-y3 loc-x11-y3)
 	(connected loc-x10-y3 loc-x10-y2)
 	(connected loc-x10-y3 loc-x10-y4)
 	(connected loc-x10-y4 loc-x9-y4)
 	(connected loc-x10-y4 loc-x11-y4)
 	(connected loc-x10-y4 loc-x10-y3)
 	(connected loc-x10-y4 loc-x10-y5)
 	(connected loc-x10-y5 loc-x9-y5)
 	(connected loc-x10-y5 loc-x11-y5)
 	(connected loc-x10-y5 loc-x10-y4)
 	(connected loc-x10-y5 loc-x10-y6)
 	(connected loc-x10-y6 loc-x9-y6)
 	(connected loc-x10-y6 loc-x11-y6)
 	(connected loc-x10-y6 loc-x10-y5)
 	(connected loc-x10-y6 loc-x10-y7)
 	(connected loc-x10-y7 loc-x9-y7)
 	(connected loc-x10-y7 loc-x11-y7)
 	(connected loc-x10-y7 loc-x10-y6)
 	(connected loc-x10-y7 loc-x10-y8)
 	(connected loc-x10-y8 loc-x9-y8)
 	(connected loc-x10-y8 loc-x11-y8)
 	(connected loc-x10-y8 loc-x10-y7)
 	(connected loc-x10-y8 loc-x10-y9)
 	(connected loc-x10-y9 loc-x9-y9)
 	(connected loc-x10-y9 loc-x11-y9)
 	(connected loc-x10-y9 loc-x10-y8)
 	(connected loc-x10-y9 loc-x10-y10)
 	(connected loc-x10-y10 loc-x9-y10)
 	(connected loc-x10-y10 loc-x11-y10)
 	(connected loc-x10-y10 loc-x10-y9)
 	(connected loc-x10-y10 loc-x10-y11)
 	(connected loc-x10-y11 loc-x9-y11)
 	(connected loc-x10-y11 loc-x11-y11)
 	(connected loc-x10-y11 loc-x10-y10)
 	(connected loc-x10-y11 loc-x10-y12)
 	(connected loc-x10-y12 loc-x9-y12)
 	(connected loc-x10-y12 loc-x11-y12)
 	(connected loc-x10-y12 loc-x10-y11)
 	(connected loc-x10-y12 loc-x10-y13)
 	(connected loc-x10-y13 loc-x9-y13)
 	(connected loc-x10-y13 loc-x11-y13)
 	(connected loc-x10-y13 loc-x10-y12)
 	(connected loc-x10-y13 loc-x10-y14)
 	(connected loc-x10-y14 loc-x9-y14)
 	(connected loc-x10-y14 loc-x11-y14)
 	(connected loc-x10-y14 loc-x10-y13)
 	(connected loc-x10-y14 loc-x10-y15)
 	(connected loc-x10-y15 loc-x9-y15)
 	(connected loc-x10-y15 loc-x11-y15)
 	(connected loc-x10-y15 loc-x10-y14)
 	(connected loc-x11-y0 loc-x10-y0)
 	(connected loc-x11-y0 loc-x12-y0)
 	(connected loc-x11-y0 loc-x11-y1)
 	(connected loc-x11-y1 loc-x10-y1)
 	(connected loc-x11-y1 loc-x12-y1)
 	(connected loc-x11-y1 loc-x11-y0)
 	(connected loc-x11-y1 loc-x11-y2)
 	(connected loc-x11-y2 loc-x10-y2)
 	(connected loc-x11-y2 loc-x12-y2)
 	(connected loc-x11-y2 loc-x11-y1)
 	(connected loc-x11-y2 loc-x11-y3)
 	(connected loc-x11-y3 loc-x10-y3)
 	(connected loc-x11-y3 loc-x12-y3)
 	(connected loc-x11-y3 loc-x11-y2)
 	(connected loc-x11-y3 loc-x11-y4)
 	(connected loc-x11-y4 loc-x10-y4)
 	(connected loc-x11-y4 loc-x12-y4)
 	(connected loc-x11-y4 loc-x11-y3)
 	(connected loc-x11-y4 loc-x11-y5)
 	(connected loc-x11-y5 loc-x10-y5)
 	(connected loc-x11-y5 loc-x12-y5)
 	(connected loc-x11-y5 loc-x11-y4)
 	(connected loc-x11-y5 loc-x11-y6)
 	(connected loc-x11-y6 loc-x10-y6)
 	(connected loc-x11-y6 loc-x12-y6)
 	(connected loc-x11-y6 loc-x11-y5)
 	(connected loc-x11-y6 loc-x11-y7)
 	(connected loc-x11-y7 loc-x10-y7)
 	(connected loc-x11-y7 loc-x12-y7)
 	(connected loc-x11-y7 loc-x11-y6)
 	(connected loc-x11-y7 loc-x11-y8)
 	(connected loc-x11-y8 loc-x10-y8)
 	(connected loc-x11-y8 loc-x12-y8)
 	(connected loc-x11-y8 loc-x11-y7)
 	(connected loc-x11-y8 loc-x11-y9)
 	(connected loc-x11-y9 loc-x10-y9)
 	(connected loc-x11-y9 loc-x12-y9)
 	(connected loc-x11-y9 loc-x11-y8)
 	(connected loc-x11-y9 loc-x11-y10)
 	(connected loc-x11-y10 loc-x10-y10)
 	(connected loc-x11-y10 loc-x12-y10)
 	(connected loc-x11-y10 loc-x11-y9)
 	(connected loc-x11-y10 loc-x11-y11)
 	(connected loc-x11-y11 loc-x10-y11)
 	(connected loc-x11-y11 loc-x12-y11)
 	(connected loc-x11-y11 loc-x11-y10)
 	(connected loc-x11-y11 loc-x11-y12)
 	(connected loc-x11-y12 loc-x10-y12)
 	(connected loc-x11-y12 loc-x12-y12)
 	(connected loc-x11-y12 loc-x11-y11)
 	(connected loc-x11-y12 loc-x11-y13)
 	(connected loc-x11-y13 loc-x10-y13)
 	(connected loc-x11-y13 loc-x12-y13)
 	(connected loc-x11-y13 loc-x11-y12)
 	(connected loc-x11-y13 loc-x11-y14)
 	(connected loc-x11-y14 loc-x10-y14)
 	(connected loc-x11-y14 loc-x12-y14)
 	(connected loc-x11-y14 loc-x11-y13)
 	(connected loc-x11-y14 loc-x11-y15)
 	(connected loc-x11-y15 loc-x10-y15)
 	(connected loc-x11-y15 loc-x12-y15)
 	(connected loc-x11-y15 loc-x11-y14)
 	(connected loc-x12-y0 loc-x11-y0)
 	(connected loc-x12-y0 loc-x13-y0)
 	(connected loc-x12-y0 loc-x12-y1)
 	(connected loc-x12-y1 loc-x11-y1)
 	(connected loc-x12-y1 loc-x13-y1)
 	(connected loc-x12-y1 loc-x12-y0)
 	(connected loc-x12-y1 loc-x12-y2)
 	(connected loc-x12-y2 loc-x11-y2)
 	(connected loc-x12-y2 loc-x13-y2)
 	(connected loc-x12-y2 loc-x12-y1)
 	(connected loc-x12-y2 loc-x12-y3)
 	(connected loc-x12-y3 loc-x11-y3)
 	(connected loc-x12-y3 loc-x13-y3)
 	(connected loc-x12-y3 loc-x12-y2)
 	(connected loc-x12-y3 loc-x12-y4)
 	(connected loc-x12-y4 loc-x11-y4)
 	(connected loc-x12-y4 loc-x13-y4)
 	(connected loc-x12-y4 loc-x12-y3)
 	(connected loc-x12-y4 loc-x12-y5)
 	(connected loc-x12-y5 loc-x11-y5)
 	(connected loc-x12-y5 loc-x13-y5)
 	(connected loc-x12-y5 loc-x12-y4)
 	(connected loc-x12-y5 loc-x12-y6)
 	(connected loc-x12-y6 loc-x11-y6)
 	(connected loc-x12-y6 loc-x13-y6)
 	(connected loc-x12-y6 loc-x12-y5)
 	(connected loc-x12-y6 loc-x12-y7)
 	(connected loc-x12-y7 loc-x11-y7)
 	(connected loc-x12-y7 loc-x13-y7)
 	(connected loc-x12-y7 loc-x12-y6)
 	(connected loc-x12-y7 loc-x12-y8)
 	(connected loc-x12-y8 loc-x11-y8)
 	(connected loc-x12-y8 loc-x13-y8)
 	(connected loc-x12-y8 loc-x12-y7)
 	(connected loc-x12-y8 loc-x12-y9)
 	(connected loc-x12-y9 loc-x11-y9)
 	(connected loc-x12-y9 loc-x13-y9)
 	(connected loc-x12-y9 loc-x12-y8)
 	(connected loc-x12-y9 loc-x12-y10)
 	(connected loc-x12-y10 loc-x11-y10)
 	(connected loc-x12-y10 loc-x13-y10)
 	(connected loc-x12-y10 loc-x12-y9)
 	(connected loc-x12-y10 loc-x12-y11)
 	(connected loc-x12-y11 loc-x11-y11)
 	(connected loc-x12-y11 loc-x13-y11)
 	(connected loc-x12-y11 loc-x12-y10)
 	(connected loc-x12-y11 loc-x12-y12)
 	(connected loc-x12-y12 loc-x11-y12)
 	(connected loc-x12-y12 loc-x13-y12)
 	(connected loc-x12-y12 loc-x12-y11)
 	(connected loc-x12-y12 loc-x12-y13)
 	(connected loc-x12-y13 loc-x11-y13)
 	(connected loc-x12-y13 loc-x13-y13)
 	(connected loc-x12-y13 loc-x12-y12)
 	(connected loc-x12-y13 loc-x12-y14)
 	(connected loc-x12-y14 loc-x11-y14)
 	(connected loc-x12-y14 loc-x13-y14)
 	(connected loc-x12-y14 loc-x12-y13)
 	(connected loc-x12-y14 loc-x12-y15)
 	(connected loc-x12-y15 loc-x11-y15)
 	(connected loc-x12-y15 loc-x13-y15)
 	(connected loc-x12-y15 loc-x12-y14)
 	(connected loc-x13-y0 loc-x12-y0)
 	(connected loc-x13-y0 loc-x14-y0)
 	(connected loc-x13-y0 loc-x13-y1)
 	(connected loc-x13-y1 loc-x12-y1)
 	(connected loc-x13-y1 loc-x14-y1)
 	(connected loc-x13-y1 loc-x13-y0)
 	(connected loc-x13-y1 loc-x13-y2)
 	(connected loc-x13-y2 loc-x12-y2)
 	(connected loc-x13-y2 loc-x14-y2)
 	(connected loc-x13-y2 loc-x13-y1)
 	(connected loc-x13-y2 loc-x13-y3)
 	(connected loc-x13-y3 loc-x12-y3)
 	(connected loc-x13-y3 loc-x14-y3)
 	(connected loc-x13-y3 loc-x13-y2)
 	(connected loc-x13-y3 loc-x13-y4)
 	(connected loc-x13-y4 loc-x12-y4)
 	(connected loc-x13-y4 loc-x14-y4)
 	(connected loc-x13-y4 loc-x13-y3)
 	(connected loc-x13-y4 loc-x13-y5)
 	(connected loc-x13-y5 loc-x12-y5)
 	(connected loc-x13-y5 loc-x14-y5)
 	(connected loc-x13-y5 loc-x13-y4)
 	(connected loc-x13-y5 loc-x13-y6)
 	(connected loc-x13-y6 loc-x12-y6)
 	(connected loc-x13-y6 loc-x14-y6)
 	(connected loc-x13-y6 loc-x13-y5)
 	(connected loc-x13-y6 loc-x13-y7)
 	(connected loc-x13-y7 loc-x12-y7)
 	(connected loc-x13-y7 loc-x14-y7)
 	(connected loc-x13-y7 loc-x13-y6)
 	(connected loc-x13-y7 loc-x13-y8)
 	(connected loc-x13-y8 loc-x12-y8)
 	(connected loc-x13-y8 loc-x14-y8)
 	(connected loc-x13-y8 loc-x13-y7)
 	(connected loc-x13-y8 loc-x13-y9)
 	(connected loc-x13-y9 loc-x12-y9)
 	(connected loc-x13-y9 loc-x14-y9)
 	(connected loc-x13-y9 loc-x13-y8)
 	(connected loc-x13-y9 loc-x13-y10)
 	(connected loc-x13-y10 loc-x12-y10)
 	(connected loc-x13-y10 loc-x14-y10)
 	(connected loc-x13-y10 loc-x13-y9)
 	(connected loc-x13-y10 loc-x13-y11)
 	(connected loc-x13-y11 loc-x12-y11)
 	(connected loc-x13-y11 loc-x14-y11)
 	(connected loc-x13-y11 loc-x13-y10)
 	(connected loc-x13-y11 loc-x13-y12)
 	(connected loc-x13-y12 loc-x12-y12)
 	(connected loc-x13-y12 loc-x14-y12)
 	(connected loc-x13-y12 loc-x13-y11)
 	(connected loc-x13-y12 loc-x13-y13)
 	(connected loc-x13-y13 loc-x12-y13)
 	(connected loc-x13-y13 loc-x14-y13)
 	(connected loc-x13-y13 loc-x13-y12)
 	(connected loc-x13-y13 loc-x13-y14)
 	(connected loc-x13-y14 loc-x12-y14)
 	(connected loc-x13-y14 loc-x14-y14)
 	(connected loc-x13-y14 loc-x13-y13)
 	(connected loc-x13-y14 loc-x13-y15)
 	(connected loc-x13-y15 loc-x12-y15)
 	(connected loc-x13-y15 loc-x14-y15)
 	(connected loc-x13-y15 loc-x13-y14)
 	(connected loc-x14-y0 loc-x13-y0)
 	(connected loc-x14-y0 loc-x15-y0)
 	(connected loc-x14-y0 loc-x14-y1)
 	(connected loc-x14-y1 loc-x13-y1)
 	(connected loc-x14-y1 loc-x15-y1)
 	(connected loc-x14-y1 loc-x14-y0)
 	(connected loc-x14-y1 loc-x14-y2)
 	(connected loc-x14-y2 loc-x13-y2)
 	(connected loc-x14-y2 loc-x15-y2)
 	(connected loc-x14-y2 loc-x14-y1)
 	(connected loc-x14-y2 loc-x14-y3)
 	(connected loc-x14-y3 loc-x13-y3)
 	(connected loc-x14-y3 loc-x15-y3)
 	(connected loc-x14-y3 loc-x14-y2)
 	(connected loc-x14-y3 loc-x14-y4)
 	(connected loc-x14-y4 loc-x13-y4)
 	(connected loc-x14-y4 loc-x15-y4)
 	(connected loc-x14-y4 loc-x14-y3)
 	(connected loc-x14-y4 loc-x14-y5)
 	(connected loc-x14-y5 loc-x13-y5)
 	(connected loc-x14-y5 loc-x15-y5)
 	(connected loc-x14-y5 loc-x14-y4)
 	(connected loc-x14-y5 loc-x14-y6)
 	(connected loc-x14-y6 loc-x13-y6)
 	(connected loc-x14-y6 loc-x15-y6)
 	(connected loc-x14-y6 loc-x14-y5)
 	(connected loc-x14-y6 loc-x14-y7)
 	(connected loc-x14-y7 loc-x13-y7)
 	(connected loc-x14-y7 loc-x15-y7)
 	(connected loc-x14-y7 loc-x14-y6)
 	(connected loc-x14-y7 loc-x14-y8)
 	(connected loc-x14-y8 loc-x13-y8)
 	(connected loc-x14-y8 loc-x15-y8)
 	(connected loc-x14-y8 loc-x14-y7)
 	(connected loc-x14-y8 loc-x14-y9)
 	(connected loc-x14-y9 loc-x13-y9)
 	(connected loc-x14-y9 loc-x15-y9)
 	(connected loc-x14-y9 loc-x14-y8)
 	(connected loc-x14-y9 loc-x14-y10)
 	(connected loc-x14-y10 loc-x13-y10)
 	(connected loc-x14-y10 loc-x15-y10)
 	(connected loc-x14-y10 loc-x14-y9)
 	(connected loc-x14-y10 loc-x14-y11)
 	(connected loc-x14-y11 loc-x13-y11)
 	(connected loc-x14-y11 loc-x15-y11)
 	(connected loc-x14-y11 loc-x14-y10)
 	(connected loc-x14-y11 loc-x14-y12)
 	(connected loc-x14-y12 loc-x13-y12)
 	(connected loc-x14-y12 loc-x15-y12)
 	(connected loc-x14-y12 loc-x14-y11)
 	(connected loc-x14-y12 loc-x14-y13)
 	(connected loc-x14-y13 loc-x13-y13)
 	(connected loc-x14-y13 loc-x15-y13)
 	(connected loc-x14-y13 loc-x14-y12)
 	(connected loc-x14-y13 loc-x14-y14)
 	(connected loc-x14-y14 loc-x13-y14)
 	(connected loc-x14-y14 loc-x15-y14)
 	(connected loc-x14-y14 loc-x14-y13)
 	(connected loc-x14-y14 loc-x14-y15)
 	(connected loc-x14-y15 loc-x13-y15)
 	(connected loc-x14-y15 loc-x15-y15)
 	(connected loc-x14-y15 loc-x14-y14)
 	(connected loc-x15-y0 loc-x14-y0)
 	(connected loc-x15-y0 loc-x15-y1)
 	(connected loc-x15-y1 loc-x14-y1)
 	(connected loc-x15-y1 loc-x15-y0)
 	(connected loc-x15-y1 loc-x15-y2)
 	(connected loc-x15-y2 loc-x14-y2)
 	(connected loc-x15-y2 loc-x15-y1)
 	(connected loc-x15-y2 loc-x15-y3)
 	(connected loc-x15-y3 loc-x14-y3)
 	(connected loc-x15-y3 loc-x15-y2)
 	(connected loc-x15-y3 loc-x15-y4)
 	(connected loc-x15-y4 loc-x14-y4)
 	(connected loc-x15-y4 loc-x15-y3)
 	(connected loc-x15-y4 loc-x15-y5)
 	(connected loc-x15-y5 loc-x14-y5)
 	(connected loc-x15-y5 loc-x15-y4)
 	(connected loc-x15-y5 loc-x15-y6)
 	(connected loc-x15-y6 loc-x14-y6)
 	(connected loc-x15-y6 loc-x15-y5)
 	(connected loc-x15-y6 loc-x15-y7)
 	(connected loc-x15-y7 loc-x14-y7)
 	(connected loc-x15-y7 loc-x15-y6)
 	(connected loc-x15-y7 loc-x15-y8)
 	(connected loc-x15-y8 loc-x14-y8)
 	(connected loc-x15-y8 loc-x15-y7)
 	(connected loc-x15-y8 loc-x15-y9)
 	(connected loc-x15-y9 loc-x14-y9)
 	(connected loc-x15-y9 loc-x15-y8)
 	(connected loc-x15-y9 loc-x15-y10)
 	(connected loc-x15-y10 loc-x14-y10)
 	(connected loc-x15-y10 loc-x15-y9)
 	(connected loc-x15-y10 loc-x15-y11)
 	(connected loc-x15-y11 loc-x14-y11)
 	(connected loc-x15-y11 loc-x15-y10)
 	(connected loc-x15-y11 loc-x15-y12)
 	(connected loc-x15-y12 loc-x14-y12)
 	(connected loc-x15-y12 loc-x15-y11)
 	(connected loc-x15-y12 loc-x15-y13)
 	(connected loc-x15-y13 loc-x14-y13)
 	(connected loc-x15-y13 loc-x15-y12)
 	(connected loc-x15-y13 loc-x15-y14)
 	(connected loc-x15-y14 loc-x14-y14)
 	(connected loc-x15-y14 loc-x15-y13)
 	(connected loc-x15-y14 loc-x15-y15)
 	(connected loc-x15-y15 loc-x14-y15)
 	(connected loc-x15-y15 loc-x15-y14)
 
)
(:goal
(and 
	(visited loc-x0-y3)
	(visited loc-x0-y4)
	(visited loc-x0-y6)
	(visited loc-x0-y7)
	(visited loc-x0-y11)
	(visited loc-x0-y12)
	(visited loc-x1-y1)
	(visited loc-x1-y8)
	(visited loc-x1-y9)
	(visited loc-x1-y15)
	(visited loc-x2-y0)
	(visited loc-x2-y2)
	(visited loc-x2-y10)
	(visited loc-x2-y13)
	(visited loc-x3-y6)
	(visited loc-x3-y8)
	(visited loc-x3-y10)
	(visited loc-x3-y11)
	(visited loc-x3-y12)
	(visited loc-x4-y9)
	(visited loc-x4-y13)
	(visited loc-x5-y3)
	(visited loc-x5-y4)
	(visited loc-x5-y12)
	(visited loc-x5-y15)
	(visited loc-x6-y14)
	(visited loc-x7-y8)
	(visited loc-x7-y9)
	(visited loc-x7-y10)
	(visited loc-x7-y14)
	(visited loc-x8-y0)
	(visited loc-x8-y5)
	(visited loc-x8-y7)
	(visited loc-x8-y8)
	(visited loc-x8-y11)
	(visited loc-x8-y14)
	(visited loc-x9-y7)
	(visited loc-x10-y0)
	(visited loc-x10-y9)
	(visited loc-x10-y10)
	(visited loc-x10-y11)
	(visited loc-x10-y12)
	(visited loc-x10-y14)
	(visited loc-x11-y1)
	(visited loc-x11-y4)
	(visited loc-x11-y11)
	(visited loc-x12-y1)
	(visited loc-x12-y4)
	(visited loc-x12-y9)
	(visited loc-x12-y15)
	(visited loc-x13-y2)
	(visited loc-x13-y5)
	(visited loc-x13-y6)
	(visited loc-x13-y8)
	(visited loc-x13-y9)
	(visited loc-x13-y12)
	(visited loc-x13-y14)
	(visited loc-x13-y15)
	(visited loc-x14-y1)
	(visited loc-x14-y2)
	(visited loc-x14-y4)
	(visited loc-x14-y5)
	(visited loc-x15-y0)
	(visited loc-x15-y15)
)
)
)
