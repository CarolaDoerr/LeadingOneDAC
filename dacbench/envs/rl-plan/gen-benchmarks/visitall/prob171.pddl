(define (problem grid-31)
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
	loc-x0-y16
	loc-x0-y17
	loc-x0-y18
	loc-x0-y19
	loc-x0-y20
	loc-x0-y21
	loc-x0-y22
	loc-x0-y23
	loc-x0-y24
	loc-x0-y25
	loc-x0-y26
	loc-x0-y27
	loc-x0-y28
	loc-x0-y29
	loc-x0-y30
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
	loc-x1-y16
	loc-x1-y17
	loc-x1-y18
	loc-x1-y19
	loc-x1-y20
	loc-x1-y21
	loc-x1-y22
	loc-x1-y23
	loc-x1-y24
	loc-x1-y25
	loc-x1-y26
	loc-x1-y27
	loc-x1-y28
	loc-x1-y29
	loc-x1-y30
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
	loc-x2-y16
	loc-x2-y17
	loc-x2-y18
	loc-x2-y19
	loc-x2-y20
	loc-x2-y21
	loc-x2-y22
	loc-x2-y23
	loc-x2-y24
	loc-x2-y25
	loc-x2-y26
	loc-x2-y27
	loc-x2-y28
	loc-x2-y29
	loc-x2-y30
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
	loc-x3-y16
	loc-x3-y17
	loc-x3-y18
	loc-x3-y19
	loc-x3-y20
	loc-x3-y21
	loc-x3-y22
	loc-x3-y23
	loc-x3-y24
	loc-x3-y25
	loc-x3-y26
	loc-x3-y27
	loc-x3-y28
	loc-x3-y29
	loc-x3-y30
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
	loc-x4-y16
	loc-x4-y17
	loc-x4-y18
	loc-x4-y19
	loc-x4-y20
	loc-x4-y21
	loc-x4-y22
	loc-x4-y23
	loc-x4-y24
	loc-x4-y25
	loc-x4-y26
	loc-x4-y27
	loc-x4-y28
	loc-x4-y29
	loc-x4-y30
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
	loc-x5-y16
	loc-x5-y17
	loc-x5-y18
	loc-x5-y19
	loc-x5-y20
	loc-x5-y21
	loc-x5-y22
	loc-x5-y23
	loc-x5-y24
	loc-x5-y25
	loc-x5-y26
	loc-x5-y27
	loc-x5-y28
	loc-x5-y29
	loc-x5-y30
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
	loc-x6-y16
	loc-x6-y17
	loc-x6-y18
	loc-x6-y19
	loc-x6-y20
	loc-x6-y21
	loc-x6-y22
	loc-x6-y23
	loc-x6-y24
	loc-x6-y25
	loc-x6-y26
	loc-x6-y27
	loc-x6-y28
	loc-x6-y29
	loc-x6-y30
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
	loc-x7-y16
	loc-x7-y17
	loc-x7-y18
	loc-x7-y19
	loc-x7-y20
	loc-x7-y21
	loc-x7-y22
	loc-x7-y23
	loc-x7-y24
	loc-x7-y25
	loc-x7-y26
	loc-x7-y27
	loc-x7-y28
	loc-x7-y29
	loc-x7-y30
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
	loc-x8-y16
	loc-x8-y17
	loc-x8-y18
	loc-x8-y19
	loc-x8-y20
	loc-x8-y21
	loc-x8-y22
	loc-x8-y23
	loc-x8-y24
	loc-x8-y25
	loc-x8-y26
	loc-x8-y27
	loc-x8-y28
	loc-x8-y29
	loc-x8-y30
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
	loc-x9-y16
	loc-x9-y17
	loc-x9-y18
	loc-x9-y19
	loc-x9-y20
	loc-x9-y21
	loc-x9-y22
	loc-x9-y23
	loc-x9-y24
	loc-x9-y25
	loc-x9-y26
	loc-x9-y27
	loc-x9-y28
	loc-x9-y29
	loc-x9-y30
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
	loc-x10-y16
	loc-x10-y17
	loc-x10-y18
	loc-x10-y19
	loc-x10-y20
	loc-x10-y21
	loc-x10-y22
	loc-x10-y23
	loc-x10-y24
	loc-x10-y25
	loc-x10-y26
	loc-x10-y27
	loc-x10-y28
	loc-x10-y29
	loc-x10-y30
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
	loc-x11-y16
	loc-x11-y17
	loc-x11-y18
	loc-x11-y19
	loc-x11-y20
	loc-x11-y21
	loc-x11-y22
	loc-x11-y23
	loc-x11-y24
	loc-x11-y25
	loc-x11-y26
	loc-x11-y27
	loc-x11-y28
	loc-x11-y29
	loc-x11-y30
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
	loc-x12-y16
	loc-x12-y17
	loc-x12-y18
	loc-x12-y19
	loc-x12-y20
	loc-x12-y21
	loc-x12-y22
	loc-x12-y23
	loc-x12-y24
	loc-x12-y25
	loc-x12-y26
	loc-x12-y27
	loc-x12-y28
	loc-x12-y29
	loc-x12-y30
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
	loc-x13-y16
	loc-x13-y17
	loc-x13-y18
	loc-x13-y19
	loc-x13-y20
	loc-x13-y21
	loc-x13-y22
	loc-x13-y23
	loc-x13-y24
	loc-x13-y25
	loc-x13-y26
	loc-x13-y27
	loc-x13-y28
	loc-x13-y29
	loc-x13-y30
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
	loc-x14-y16
	loc-x14-y17
	loc-x14-y18
	loc-x14-y19
	loc-x14-y20
	loc-x14-y21
	loc-x14-y22
	loc-x14-y23
	loc-x14-y24
	loc-x14-y25
	loc-x14-y26
	loc-x14-y27
	loc-x14-y28
	loc-x14-y29
	loc-x14-y30
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
	loc-x15-y16
	loc-x15-y17
	loc-x15-y18
	loc-x15-y19
	loc-x15-y20
	loc-x15-y21
	loc-x15-y22
	loc-x15-y23
	loc-x15-y24
	loc-x15-y25
	loc-x15-y26
	loc-x15-y27
	loc-x15-y28
	loc-x15-y29
	loc-x15-y30
	loc-x16-y0
	loc-x16-y1
	loc-x16-y2
	loc-x16-y3
	loc-x16-y4
	loc-x16-y5
	loc-x16-y6
	loc-x16-y7
	loc-x16-y8
	loc-x16-y9
	loc-x16-y10
	loc-x16-y11
	loc-x16-y12
	loc-x16-y13
	loc-x16-y14
	loc-x16-y15
	loc-x16-y16
	loc-x16-y17
	loc-x16-y18
	loc-x16-y19
	loc-x16-y20
	loc-x16-y21
	loc-x16-y22
	loc-x16-y23
	loc-x16-y24
	loc-x16-y25
	loc-x16-y26
	loc-x16-y27
	loc-x16-y28
	loc-x16-y29
	loc-x16-y30
	loc-x17-y0
	loc-x17-y1
	loc-x17-y2
	loc-x17-y3
	loc-x17-y4
	loc-x17-y5
	loc-x17-y6
	loc-x17-y7
	loc-x17-y8
	loc-x17-y9
	loc-x17-y10
	loc-x17-y11
	loc-x17-y12
	loc-x17-y13
	loc-x17-y14
	loc-x17-y15
	loc-x17-y16
	loc-x17-y17
	loc-x17-y18
	loc-x17-y19
	loc-x17-y20
	loc-x17-y21
	loc-x17-y22
	loc-x17-y23
	loc-x17-y24
	loc-x17-y25
	loc-x17-y26
	loc-x17-y27
	loc-x17-y28
	loc-x17-y29
	loc-x17-y30
	loc-x18-y0
	loc-x18-y1
	loc-x18-y2
	loc-x18-y3
	loc-x18-y4
	loc-x18-y5
	loc-x18-y6
	loc-x18-y7
	loc-x18-y8
	loc-x18-y9
	loc-x18-y10
	loc-x18-y11
	loc-x18-y12
	loc-x18-y13
	loc-x18-y14
	loc-x18-y15
	loc-x18-y16
	loc-x18-y17
	loc-x18-y18
	loc-x18-y19
	loc-x18-y20
	loc-x18-y21
	loc-x18-y22
	loc-x18-y23
	loc-x18-y24
	loc-x18-y25
	loc-x18-y26
	loc-x18-y27
	loc-x18-y28
	loc-x18-y29
	loc-x18-y30
	loc-x19-y0
	loc-x19-y1
	loc-x19-y2
	loc-x19-y3
	loc-x19-y4
	loc-x19-y5
	loc-x19-y6
	loc-x19-y7
	loc-x19-y8
	loc-x19-y9
	loc-x19-y10
	loc-x19-y11
	loc-x19-y12
	loc-x19-y13
	loc-x19-y14
	loc-x19-y15
	loc-x19-y16
	loc-x19-y17
	loc-x19-y18
	loc-x19-y19
	loc-x19-y20
	loc-x19-y21
	loc-x19-y22
	loc-x19-y23
	loc-x19-y24
	loc-x19-y25
	loc-x19-y26
	loc-x19-y27
	loc-x19-y28
	loc-x19-y29
	loc-x19-y30
	loc-x20-y0
	loc-x20-y1
	loc-x20-y2
	loc-x20-y3
	loc-x20-y4
	loc-x20-y5
	loc-x20-y6
	loc-x20-y7
	loc-x20-y8
	loc-x20-y9
	loc-x20-y10
	loc-x20-y11
	loc-x20-y12
	loc-x20-y13
	loc-x20-y14
	loc-x20-y15
	loc-x20-y16
	loc-x20-y17
	loc-x20-y18
	loc-x20-y19
	loc-x20-y20
	loc-x20-y21
	loc-x20-y22
	loc-x20-y23
	loc-x20-y24
	loc-x20-y25
	loc-x20-y26
	loc-x20-y27
	loc-x20-y28
	loc-x20-y29
	loc-x20-y30
	loc-x21-y0
	loc-x21-y1
	loc-x21-y2
	loc-x21-y3
	loc-x21-y4
	loc-x21-y5
	loc-x21-y6
	loc-x21-y7
	loc-x21-y8
	loc-x21-y9
	loc-x21-y10
	loc-x21-y11
	loc-x21-y12
	loc-x21-y13
	loc-x21-y14
	loc-x21-y15
	loc-x21-y16
	loc-x21-y17
	loc-x21-y18
	loc-x21-y19
	loc-x21-y20
	loc-x21-y21
	loc-x21-y22
	loc-x21-y23
	loc-x21-y24
	loc-x21-y25
	loc-x21-y26
	loc-x21-y27
	loc-x21-y28
	loc-x21-y29
	loc-x21-y30
	loc-x22-y0
	loc-x22-y1
	loc-x22-y2
	loc-x22-y3
	loc-x22-y4
	loc-x22-y5
	loc-x22-y6
	loc-x22-y7
	loc-x22-y8
	loc-x22-y9
	loc-x22-y10
	loc-x22-y11
	loc-x22-y12
	loc-x22-y13
	loc-x22-y14
	loc-x22-y15
	loc-x22-y16
	loc-x22-y17
	loc-x22-y18
	loc-x22-y19
	loc-x22-y20
	loc-x22-y21
	loc-x22-y22
	loc-x22-y23
	loc-x22-y24
	loc-x22-y25
	loc-x22-y26
	loc-x22-y27
	loc-x22-y28
	loc-x22-y29
	loc-x22-y30
	loc-x23-y0
	loc-x23-y1
	loc-x23-y2
	loc-x23-y3
	loc-x23-y4
	loc-x23-y5
	loc-x23-y6
	loc-x23-y7
	loc-x23-y8
	loc-x23-y9
	loc-x23-y10
	loc-x23-y11
	loc-x23-y12
	loc-x23-y13
	loc-x23-y14
	loc-x23-y15
	loc-x23-y16
	loc-x23-y17
	loc-x23-y18
	loc-x23-y19
	loc-x23-y20
	loc-x23-y21
	loc-x23-y22
	loc-x23-y23
	loc-x23-y24
	loc-x23-y25
	loc-x23-y26
	loc-x23-y27
	loc-x23-y28
	loc-x23-y29
	loc-x23-y30
	loc-x24-y0
	loc-x24-y1
	loc-x24-y2
	loc-x24-y3
	loc-x24-y4
	loc-x24-y5
	loc-x24-y6
	loc-x24-y7
	loc-x24-y8
	loc-x24-y9
	loc-x24-y10
	loc-x24-y11
	loc-x24-y12
	loc-x24-y13
	loc-x24-y14
	loc-x24-y15
	loc-x24-y16
	loc-x24-y17
	loc-x24-y18
	loc-x24-y19
	loc-x24-y20
	loc-x24-y21
	loc-x24-y22
	loc-x24-y23
	loc-x24-y24
	loc-x24-y25
	loc-x24-y26
	loc-x24-y27
	loc-x24-y28
	loc-x24-y29
	loc-x24-y30
	loc-x25-y0
	loc-x25-y1
	loc-x25-y2
	loc-x25-y3
	loc-x25-y4
	loc-x25-y5
	loc-x25-y6
	loc-x25-y7
	loc-x25-y8
	loc-x25-y9
	loc-x25-y10
	loc-x25-y11
	loc-x25-y12
	loc-x25-y13
	loc-x25-y14
	loc-x25-y15
	loc-x25-y16
	loc-x25-y17
	loc-x25-y18
	loc-x25-y19
	loc-x25-y20
	loc-x25-y21
	loc-x25-y22
	loc-x25-y23
	loc-x25-y24
	loc-x25-y25
	loc-x25-y26
	loc-x25-y27
	loc-x25-y28
	loc-x25-y29
	loc-x25-y30
	loc-x26-y0
	loc-x26-y1
	loc-x26-y2
	loc-x26-y3
	loc-x26-y4
	loc-x26-y5
	loc-x26-y6
	loc-x26-y7
	loc-x26-y8
	loc-x26-y9
	loc-x26-y10
	loc-x26-y11
	loc-x26-y12
	loc-x26-y13
	loc-x26-y14
	loc-x26-y15
	loc-x26-y16
	loc-x26-y17
	loc-x26-y18
	loc-x26-y19
	loc-x26-y20
	loc-x26-y21
	loc-x26-y22
	loc-x26-y23
	loc-x26-y24
	loc-x26-y25
	loc-x26-y26
	loc-x26-y27
	loc-x26-y28
	loc-x26-y29
	loc-x26-y30
	loc-x27-y0
	loc-x27-y1
	loc-x27-y2
	loc-x27-y3
	loc-x27-y4
	loc-x27-y5
	loc-x27-y6
	loc-x27-y7
	loc-x27-y8
	loc-x27-y9
	loc-x27-y10
	loc-x27-y11
	loc-x27-y12
	loc-x27-y13
	loc-x27-y14
	loc-x27-y15
	loc-x27-y16
	loc-x27-y17
	loc-x27-y18
	loc-x27-y19
	loc-x27-y20
	loc-x27-y21
	loc-x27-y22
	loc-x27-y23
	loc-x27-y24
	loc-x27-y25
	loc-x27-y26
	loc-x27-y27
	loc-x27-y28
	loc-x27-y29
	loc-x27-y30
	loc-x28-y0
	loc-x28-y1
	loc-x28-y2
	loc-x28-y3
	loc-x28-y4
	loc-x28-y5
	loc-x28-y6
	loc-x28-y7
	loc-x28-y8
	loc-x28-y9
	loc-x28-y10
	loc-x28-y11
	loc-x28-y12
	loc-x28-y13
	loc-x28-y14
	loc-x28-y15
	loc-x28-y16
	loc-x28-y17
	loc-x28-y18
	loc-x28-y19
	loc-x28-y20
	loc-x28-y21
	loc-x28-y22
	loc-x28-y23
	loc-x28-y24
	loc-x28-y25
	loc-x28-y26
	loc-x28-y27
	loc-x28-y28
	loc-x28-y29
	loc-x28-y30
	loc-x29-y0
	loc-x29-y1
	loc-x29-y2
	loc-x29-y3
	loc-x29-y4
	loc-x29-y5
	loc-x29-y6
	loc-x29-y7
	loc-x29-y8
	loc-x29-y9
	loc-x29-y10
	loc-x29-y11
	loc-x29-y12
	loc-x29-y13
	loc-x29-y14
	loc-x29-y15
	loc-x29-y16
	loc-x29-y17
	loc-x29-y18
	loc-x29-y19
	loc-x29-y20
	loc-x29-y21
	loc-x29-y22
	loc-x29-y23
	loc-x29-y24
	loc-x29-y25
	loc-x29-y26
	loc-x29-y27
	loc-x29-y28
	loc-x29-y29
	loc-x29-y30
	loc-x30-y0
	loc-x30-y1
	loc-x30-y2
	loc-x30-y3
	loc-x30-y4
	loc-x30-y5
	loc-x30-y6
	loc-x30-y7
	loc-x30-y8
	loc-x30-y9
	loc-x30-y10
	loc-x30-y11
	loc-x30-y12
	loc-x30-y13
	loc-x30-y14
	loc-x30-y15
	loc-x30-y16
	loc-x30-y17
	loc-x30-y18
	loc-x30-y19
	loc-x30-y20
	loc-x30-y21
	loc-x30-y22
	loc-x30-y23
	loc-x30-y24
	loc-x30-y25
	loc-x30-y26
	loc-x30-y27
	loc-x30-y28
	loc-x30-y29
	loc-x30-y30
- place 
        
)
(:init
	(at-robot loc-x22-y18)
	(visited loc-x22-y18)
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
 	(connected loc-x0-y15 loc-x0-y16)
 	(connected loc-x0-y16 loc-x1-y16)
 	(connected loc-x0-y16 loc-x0-y15)
 	(connected loc-x0-y16 loc-x0-y17)
 	(connected loc-x0-y17 loc-x1-y17)
 	(connected loc-x0-y17 loc-x0-y16)
 	(connected loc-x0-y17 loc-x0-y18)
 	(connected loc-x0-y18 loc-x1-y18)
 	(connected loc-x0-y18 loc-x0-y17)
 	(connected loc-x0-y18 loc-x0-y19)
 	(connected loc-x0-y19 loc-x1-y19)
 	(connected loc-x0-y19 loc-x0-y18)
 	(connected loc-x0-y19 loc-x0-y20)
 	(connected loc-x0-y20 loc-x1-y20)
 	(connected loc-x0-y20 loc-x0-y19)
 	(connected loc-x0-y20 loc-x0-y21)
 	(connected loc-x0-y21 loc-x1-y21)
 	(connected loc-x0-y21 loc-x0-y20)
 	(connected loc-x0-y21 loc-x0-y22)
 	(connected loc-x0-y22 loc-x1-y22)
 	(connected loc-x0-y22 loc-x0-y21)
 	(connected loc-x0-y22 loc-x0-y23)
 	(connected loc-x0-y23 loc-x1-y23)
 	(connected loc-x0-y23 loc-x0-y22)
 	(connected loc-x0-y23 loc-x0-y24)
 	(connected loc-x0-y24 loc-x1-y24)
 	(connected loc-x0-y24 loc-x0-y23)
 	(connected loc-x0-y24 loc-x0-y25)
 	(connected loc-x0-y25 loc-x1-y25)
 	(connected loc-x0-y25 loc-x0-y24)
 	(connected loc-x0-y25 loc-x0-y26)
 	(connected loc-x0-y26 loc-x1-y26)
 	(connected loc-x0-y26 loc-x0-y25)
 	(connected loc-x0-y26 loc-x0-y27)
 	(connected loc-x0-y27 loc-x1-y27)
 	(connected loc-x0-y27 loc-x0-y26)
 	(connected loc-x0-y27 loc-x0-y28)
 	(connected loc-x0-y28 loc-x1-y28)
 	(connected loc-x0-y28 loc-x0-y27)
 	(connected loc-x0-y28 loc-x0-y29)
 	(connected loc-x0-y29 loc-x1-y29)
 	(connected loc-x0-y29 loc-x0-y28)
 	(connected loc-x0-y29 loc-x0-y30)
 	(connected loc-x0-y30 loc-x1-y30)
 	(connected loc-x0-y30 loc-x0-y29)
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
 	(connected loc-x1-y15 loc-x1-y16)
 	(connected loc-x1-y16 loc-x0-y16)
 	(connected loc-x1-y16 loc-x2-y16)
 	(connected loc-x1-y16 loc-x1-y15)
 	(connected loc-x1-y16 loc-x1-y17)
 	(connected loc-x1-y17 loc-x0-y17)
 	(connected loc-x1-y17 loc-x2-y17)
 	(connected loc-x1-y17 loc-x1-y16)
 	(connected loc-x1-y17 loc-x1-y18)
 	(connected loc-x1-y18 loc-x0-y18)
 	(connected loc-x1-y18 loc-x2-y18)
 	(connected loc-x1-y18 loc-x1-y17)
 	(connected loc-x1-y18 loc-x1-y19)
 	(connected loc-x1-y19 loc-x0-y19)
 	(connected loc-x1-y19 loc-x2-y19)
 	(connected loc-x1-y19 loc-x1-y18)
 	(connected loc-x1-y19 loc-x1-y20)
 	(connected loc-x1-y20 loc-x0-y20)
 	(connected loc-x1-y20 loc-x2-y20)
 	(connected loc-x1-y20 loc-x1-y19)
 	(connected loc-x1-y20 loc-x1-y21)
 	(connected loc-x1-y21 loc-x0-y21)
 	(connected loc-x1-y21 loc-x2-y21)
 	(connected loc-x1-y21 loc-x1-y20)
 	(connected loc-x1-y21 loc-x1-y22)
 	(connected loc-x1-y22 loc-x0-y22)
 	(connected loc-x1-y22 loc-x2-y22)
 	(connected loc-x1-y22 loc-x1-y21)
 	(connected loc-x1-y22 loc-x1-y23)
 	(connected loc-x1-y23 loc-x0-y23)
 	(connected loc-x1-y23 loc-x2-y23)
 	(connected loc-x1-y23 loc-x1-y22)
 	(connected loc-x1-y23 loc-x1-y24)
 	(connected loc-x1-y24 loc-x0-y24)
 	(connected loc-x1-y24 loc-x2-y24)
 	(connected loc-x1-y24 loc-x1-y23)
 	(connected loc-x1-y24 loc-x1-y25)
 	(connected loc-x1-y25 loc-x0-y25)
 	(connected loc-x1-y25 loc-x2-y25)
 	(connected loc-x1-y25 loc-x1-y24)
 	(connected loc-x1-y25 loc-x1-y26)
 	(connected loc-x1-y26 loc-x0-y26)
 	(connected loc-x1-y26 loc-x2-y26)
 	(connected loc-x1-y26 loc-x1-y25)
 	(connected loc-x1-y26 loc-x1-y27)
 	(connected loc-x1-y27 loc-x0-y27)
 	(connected loc-x1-y27 loc-x2-y27)
 	(connected loc-x1-y27 loc-x1-y26)
 	(connected loc-x1-y27 loc-x1-y28)
 	(connected loc-x1-y28 loc-x0-y28)
 	(connected loc-x1-y28 loc-x2-y28)
 	(connected loc-x1-y28 loc-x1-y27)
 	(connected loc-x1-y28 loc-x1-y29)
 	(connected loc-x1-y29 loc-x0-y29)
 	(connected loc-x1-y29 loc-x2-y29)
 	(connected loc-x1-y29 loc-x1-y28)
 	(connected loc-x1-y29 loc-x1-y30)
 	(connected loc-x1-y30 loc-x0-y30)
 	(connected loc-x1-y30 loc-x2-y30)
 	(connected loc-x1-y30 loc-x1-y29)
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
 	(connected loc-x2-y15 loc-x2-y16)
 	(connected loc-x2-y16 loc-x1-y16)
 	(connected loc-x2-y16 loc-x3-y16)
 	(connected loc-x2-y16 loc-x2-y15)
 	(connected loc-x2-y16 loc-x2-y17)
 	(connected loc-x2-y17 loc-x1-y17)
 	(connected loc-x2-y17 loc-x3-y17)
 	(connected loc-x2-y17 loc-x2-y16)
 	(connected loc-x2-y17 loc-x2-y18)
 	(connected loc-x2-y18 loc-x1-y18)
 	(connected loc-x2-y18 loc-x3-y18)
 	(connected loc-x2-y18 loc-x2-y17)
 	(connected loc-x2-y18 loc-x2-y19)
 	(connected loc-x2-y19 loc-x1-y19)
 	(connected loc-x2-y19 loc-x3-y19)
 	(connected loc-x2-y19 loc-x2-y18)
 	(connected loc-x2-y19 loc-x2-y20)
 	(connected loc-x2-y20 loc-x1-y20)
 	(connected loc-x2-y20 loc-x3-y20)
 	(connected loc-x2-y20 loc-x2-y19)
 	(connected loc-x2-y20 loc-x2-y21)
 	(connected loc-x2-y21 loc-x1-y21)
 	(connected loc-x2-y21 loc-x3-y21)
 	(connected loc-x2-y21 loc-x2-y20)
 	(connected loc-x2-y21 loc-x2-y22)
 	(connected loc-x2-y22 loc-x1-y22)
 	(connected loc-x2-y22 loc-x3-y22)
 	(connected loc-x2-y22 loc-x2-y21)
 	(connected loc-x2-y22 loc-x2-y23)
 	(connected loc-x2-y23 loc-x1-y23)
 	(connected loc-x2-y23 loc-x3-y23)
 	(connected loc-x2-y23 loc-x2-y22)
 	(connected loc-x2-y23 loc-x2-y24)
 	(connected loc-x2-y24 loc-x1-y24)
 	(connected loc-x2-y24 loc-x3-y24)
 	(connected loc-x2-y24 loc-x2-y23)
 	(connected loc-x2-y24 loc-x2-y25)
 	(connected loc-x2-y25 loc-x1-y25)
 	(connected loc-x2-y25 loc-x3-y25)
 	(connected loc-x2-y25 loc-x2-y24)
 	(connected loc-x2-y25 loc-x2-y26)
 	(connected loc-x2-y26 loc-x1-y26)
 	(connected loc-x2-y26 loc-x3-y26)
 	(connected loc-x2-y26 loc-x2-y25)
 	(connected loc-x2-y26 loc-x2-y27)
 	(connected loc-x2-y27 loc-x1-y27)
 	(connected loc-x2-y27 loc-x3-y27)
 	(connected loc-x2-y27 loc-x2-y26)
 	(connected loc-x2-y27 loc-x2-y28)
 	(connected loc-x2-y28 loc-x1-y28)
 	(connected loc-x2-y28 loc-x3-y28)
 	(connected loc-x2-y28 loc-x2-y27)
 	(connected loc-x2-y28 loc-x2-y29)
 	(connected loc-x2-y29 loc-x1-y29)
 	(connected loc-x2-y29 loc-x3-y29)
 	(connected loc-x2-y29 loc-x2-y28)
 	(connected loc-x2-y29 loc-x2-y30)
 	(connected loc-x2-y30 loc-x1-y30)
 	(connected loc-x2-y30 loc-x3-y30)
 	(connected loc-x2-y30 loc-x2-y29)
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
 	(connected loc-x3-y15 loc-x3-y16)
 	(connected loc-x3-y16 loc-x2-y16)
 	(connected loc-x3-y16 loc-x4-y16)
 	(connected loc-x3-y16 loc-x3-y15)
 	(connected loc-x3-y16 loc-x3-y17)
 	(connected loc-x3-y17 loc-x2-y17)
 	(connected loc-x3-y17 loc-x4-y17)
 	(connected loc-x3-y17 loc-x3-y16)
 	(connected loc-x3-y17 loc-x3-y18)
 	(connected loc-x3-y18 loc-x2-y18)
 	(connected loc-x3-y18 loc-x4-y18)
 	(connected loc-x3-y18 loc-x3-y17)
 	(connected loc-x3-y18 loc-x3-y19)
 	(connected loc-x3-y19 loc-x2-y19)
 	(connected loc-x3-y19 loc-x4-y19)
 	(connected loc-x3-y19 loc-x3-y18)
 	(connected loc-x3-y19 loc-x3-y20)
 	(connected loc-x3-y20 loc-x2-y20)
 	(connected loc-x3-y20 loc-x4-y20)
 	(connected loc-x3-y20 loc-x3-y19)
 	(connected loc-x3-y20 loc-x3-y21)
 	(connected loc-x3-y21 loc-x2-y21)
 	(connected loc-x3-y21 loc-x4-y21)
 	(connected loc-x3-y21 loc-x3-y20)
 	(connected loc-x3-y21 loc-x3-y22)
 	(connected loc-x3-y22 loc-x2-y22)
 	(connected loc-x3-y22 loc-x4-y22)
 	(connected loc-x3-y22 loc-x3-y21)
 	(connected loc-x3-y22 loc-x3-y23)
 	(connected loc-x3-y23 loc-x2-y23)
 	(connected loc-x3-y23 loc-x4-y23)
 	(connected loc-x3-y23 loc-x3-y22)
 	(connected loc-x3-y23 loc-x3-y24)
 	(connected loc-x3-y24 loc-x2-y24)
 	(connected loc-x3-y24 loc-x4-y24)
 	(connected loc-x3-y24 loc-x3-y23)
 	(connected loc-x3-y24 loc-x3-y25)
 	(connected loc-x3-y25 loc-x2-y25)
 	(connected loc-x3-y25 loc-x4-y25)
 	(connected loc-x3-y25 loc-x3-y24)
 	(connected loc-x3-y25 loc-x3-y26)
 	(connected loc-x3-y26 loc-x2-y26)
 	(connected loc-x3-y26 loc-x4-y26)
 	(connected loc-x3-y26 loc-x3-y25)
 	(connected loc-x3-y26 loc-x3-y27)
 	(connected loc-x3-y27 loc-x2-y27)
 	(connected loc-x3-y27 loc-x4-y27)
 	(connected loc-x3-y27 loc-x3-y26)
 	(connected loc-x3-y27 loc-x3-y28)
 	(connected loc-x3-y28 loc-x2-y28)
 	(connected loc-x3-y28 loc-x4-y28)
 	(connected loc-x3-y28 loc-x3-y27)
 	(connected loc-x3-y28 loc-x3-y29)
 	(connected loc-x3-y29 loc-x2-y29)
 	(connected loc-x3-y29 loc-x4-y29)
 	(connected loc-x3-y29 loc-x3-y28)
 	(connected loc-x3-y29 loc-x3-y30)
 	(connected loc-x3-y30 loc-x2-y30)
 	(connected loc-x3-y30 loc-x4-y30)
 	(connected loc-x3-y30 loc-x3-y29)
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
 	(connected loc-x4-y15 loc-x4-y16)
 	(connected loc-x4-y16 loc-x3-y16)
 	(connected loc-x4-y16 loc-x5-y16)
 	(connected loc-x4-y16 loc-x4-y15)
 	(connected loc-x4-y16 loc-x4-y17)
 	(connected loc-x4-y17 loc-x3-y17)
 	(connected loc-x4-y17 loc-x5-y17)
 	(connected loc-x4-y17 loc-x4-y16)
 	(connected loc-x4-y17 loc-x4-y18)
 	(connected loc-x4-y18 loc-x3-y18)
 	(connected loc-x4-y18 loc-x5-y18)
 	(connected loc-x4-y18 loc-x4-y17)
 	(connected loc-x4-y18 loc-x4-y19)
 	(connected loc-x4-y19 loc-x3-y19)
 	(connected loc-x4-y19 loc-x5-y19)
 	(connected loc-x4-y19 loc-x4-y18)
 	(connected loc-x4-y19 loc-x4-y20)
 	(connected loc-x4-y20 loc-x3-y20)
 	(connected loc-x4-y20 loc-x5-y20)
 	(connected loc-x4-y20 loc-x4-y19)
 	(connected loc-x4-y20 loc-x4-y21)
 	(connected loc-x4-y21 loc-x3-y21)
 	(connected loc-x4-y21 loc-x5-y21)
 	(connected loc-x4-y21 loc-x4-y20)
 	(connected loc-x4-y21 loc-x4-y22)
 	(connected loc-x4-y22 loc-x3-y22)
 	(connected loc-x4-y22 loc-x5-y22)
 	(connected loc-x4-y22 loc-x4-y21)
 	(connected loc-x4-y22 loc-x4-y23)
 	(connected loc-x4-y23 loc-x3-y23)
 	(connected loc-x4-y23 loc-x5-y23)
 	(connected loc-x4-y23 loc-x4-y22)
 	(connected loc-x4-y23 loc-x4-y24)
 	(connected loc-x4-y24 loc-x3-y24)
 	(connected loc-x4-y24 loc-x5-y24)
 	(connected loc-x4-y24 loc-x4-y23)
 	(connected loc-x4-y24 loc-x4-y25)
 	(connected loc-x4-y25 loc-x3-y25)
 	(connected loc-x4-y25 loc-x5-y25)
 	(connected loc-x4-y25 loc-x4-y24)
 	(connected loc-x4-y25 loc-x4-y26)
 	(connected loc-x4-y26 loc-x3-y26)
 	(connected loc-x4-y26 loc-x5-y26)
 	(connected loc-x4-y26 loc-x4-y25)
 	(connected loc-x4-y26 loc-x4-y27)
 	(connected loc-x4-y27 loc-x3-y27)
 	(connected loc-x4-y27 loc-x5-y27)
 	(connected loc-x4-y27 loc-x4-y26)
 	(connected loc-x4-y27 loc-x4-y28)
 	(connected loc-x4-y28 loc-x3-y28)
 	(connected loc-x4-y28 loc-x5-y28)
 	(connected loc-x4-y28 loc-x4-y27)
 	(connected loc-x4-y28 loc-x4-y29)
 	(connected loc-x4-y29 loc-x3-y29)
 	(connected loc-x4-y29 loc-x5-y29)
 	(connected loc-x4-y29 loc-x4-y28)
 	(connected loc-x4-y29 loc-x4-y30)
 	(connected loc-x4-y30 loc-x3-y30)
 	(connected loc-x4-y30 loc-x5-y30)
 	(connected loc-x4-y30 loc-x4-y29)
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
 	(connected loc-x5-y15 loc-x5-y16)
 	(connected loc-x5-y16 loc-x4-y16)
 	(connected loc-x5-y16 loc-x6-y16)
 	(connected loc-x5-y16 loc-x5-y15)
 	(connected loc-x5-y16 loc-x5-y17)
 	(connected loc-x5-y17 loc-x4-y17)
 	(connected loc-x5-y17 loc-x6-y17)
 	(connected loc-x5-y17 loc-x5-y16)
 	(connected loc-x5-y17 loc-x5-y18)
 	(connected loc-x5-y18 loc-x4-y18)
 	(connected loc-x5-y18 loc-x6-y18)
 	(connected loc-x5-y18 loc-x5-y17)
 	(connected loc-x5-y18 loc-x5-y19)
 	(connected loc-x5-y19 loc-x4-y19)
 	(connected loc-x5-y19 loc-x6-y19)
 	(connected loc-x5-y19 loc-x5-y18)
 	(connected loc-x5-y19 loc-x5-y20)
 	(connected loc-x5-y20 loc-x4-y20)
 	(connected loc-x5-y20 loc-x6-y20)
 	(connected loc-x5-y20 loc-x5-y19)
 	(connected loc-x5-y20 loc-x5-y21)
 	(connected loc-x5-y21 loc-x4-y21)
 	(connected loc-x5-y21 loc-x6-y21)
 	(connected loc-x5-y21 loc-x5-y20)
 	(connected loc-x5-y21 loc-x5-y22)
 	(connected loc-x5-y22 loc-x4-y22)
 	(connected loc-x5-y22 loc-x6-y22)
 	(connected loc-x5-y22 loc-x5-y21)
 	(connected loc-x5-y22 loc-x5-y23)
 	(connected loc-x5-y23 loc-x4-y23)
 	(connected loc-x5-y23 loc-x6-y23)
 	(connected loc-x5-y23 loc-x5-y22)
 	(connected loc-x5-y23 loc-x5-y24)
 	(connected loc-x5-y24 loc-x4-y24)
 	(connected loc-x5-y24 loc-x6-y24)
 	(connected loc-x5-y24 loc-x5-y23)
 	(connected loc-x5-y24 loc-x5-y25)
 	(connected loc-x5-y25 loc-x4-y25)
 	(connected loc-x5-y25 loc-x6-y25)
 	(connected loc-x5-y25 loc-x5-y24)
 	(connected loc-x5-y25 loc-x5-y26)
 	(connected loc-x5-y26 loc-x4-y26)
 	(connected loc-x5-y26 loc-x6-y26)
 	(connected loc-x5-y26 loc-x5-y25)
 	(connected loc-x5-y26 loc-x5-y27)
 	(connected loc-x5-y27 loc-x4-y27)
 	(connected loc-x5-y27 loc-x6-y27)
 	(connected loc-x5-y27 loc-x5-y26)
 	(connected loc-x5-y27 loc-x5-y28)
 	(connected loc-x5-y28 loc-x4-y28)
 	(connected loc-x5-y28 loc-x6-y28)
 	(connected loc-x5-y28 loc-x5-y27)
 	(connected loc-x5-y28 loc-x5-y29)
 	(connected loc-x5-y29 loc-x4-y29)
 	(connected loc-x5-y29 loc-x6-y29)
 	(connected loc-x5-y29 loc-x5-y28)
 	(connected loc-x5-y29 loc-x5-y30)
 	(connected loc-x5-y30 loc-x4-y30)
 	(connected loc-x5-y30 loc-x6-y30)
 	(connected loc-x5-y30 loc-x5-y29)
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
 	(connected loc-x6-y15 loc-x6-y16)
 	(connected loc-x6-y16 loc-x5-y16)
 	(connected loc-x6-y16 loc-x7-y16)
 	(connected loc-x6-y16 loc-x6-y15)
 	(connected loc-x6-y16 loc-x6-y17)
 	(connected loc-x6-y17 loc-x5-y17)
 	(connected loc-x6-y17 loc-x7-y17)
 	(connected loc-x6-y17 loc-x6-y16)
 	(connected loc-x6-y17 loc-x6-y18)
 	(connected loc-x6-y18 loc-x5-y18)
 	(connected loc-x6-y18 loc-x7-y18)
 	(connected loc-x6-y18 loc-x6-y17)
 	(connected loc-x6-y18 loc-x6-y19)
 	(connected loc-x6-y19 loc-x5-y19)
 	(connected loc-x6-y19 loc-x7-y19)
 	(connected loc-x6-y19 loc-x6-y18)
 	(connected loc-x6-y19 loc-x6-y20)
 	(connected loc-x6-y20 loc-x5-y20)
 	(connected loc-x6-y20 loc-x7-y20)
 	(connected loc-x6-y20 loc-x6-y19)
 	(connected loc-x6-y20 loc-x6-y21)
 	(connected loc-x6-y21 loc-x5-y21)
 	(connected loc-x6-y21 loc-x7-y21)
 	(connected loc-x6-y21 loc-x6-y20)
 	(connected loc-x6-y21 loc-x6-y22)
 	(connected loc-x6-y22 loc-x5-y22)
 	(connected loc-x6-y22 loc-x7-y22)
 	(connected loc-x6-y22 loc-x6-y21)
 	(connected loc-x6-y22 loc-x6-y23)
 	(connected loc-x6-y23 loc-x5-y23)
 	(connected loc-x6-y23 loc-x7-y23)
 	(connected loc-x6-y23 loc-x6-y22)
 	(connected loc-x6-y23 loc-x6-y24)
 	(connected loc-x6-y24 loc-x5-y24)
 	(connected loc-x6-y24 loc-x7-y24)
 	(connected loc-x6-y24 loc-x6-y23)
 	(connected loc-x6-y24 loc-x6-y25)
 	(connected loc-x6-y25 loc-x5-y25)
 	(connected loc-x6-y25 loc-x7-y25)
 	(connected loc-x6-y25 loc-x6-y24)
 	(connected loc-x6-y25 loc-x6-y26)
 	(connected loc-x6-y26 loc-x5-y26)
 	(connected loc-x6-y26 loc-x7-y26)
 	(connected loc-x6-y26 loc-x6-y25)
 	(connected loc-x6-y26 loc-x6-y27)
 	(connected loc-x6-y27 loc-x5-y27)
 	(connected loc-x6-y27 loc-x7-y27)
 	(connected loc-x6-y27 loc-x6-y26)
 	(connected loc-x6-y27 loc-x6-y28)
 	(connected loc-x6-y28 loc-x5-y28)
 	(connected loc-x6-y28 loc-x7-y28)
 	(connected loc-x6-y28 loc-x6-y27)
 	(connected loc-x6-y28 loc-x6-y29)
 	(connected loc-x6-y29 loc-x5-y29)
 	(connected loc-x6-y29 loc-x7-y29)
 	(connected loc-x6-y29 loc-x6-y28)
 	(connected loc-x6-y29 loc-x6-y30)
 	(connected loc-x6-y30 loc-x5-y30)
 	(connected loc-x6-y30 loc-x7-y30)
 	(connected loc-x6-y30 loc-x6-y29)
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
 	(connected loc-x7-y15 loc-x7-y16)
 	(connected loc-x7-y16 loc-x6-y16)
 	(connected loc-x7-y16 loc-x8-y16)
 	(connected loc-x7-y16 loc-x7-y15)
 	(connected loc-x7-y16 loc-x7-y17)
 	(connected loc-x7-y17 loc-x6-y17)
 	(connected loc-x7-y17 loc-x8-y17)
 	(connected loc-x7-y17 loc-x7-y16)
 	(connected loc-x7-y17 loc-x7-y18)
 	(connected loc-x7-y18 loc-x6-y18)
 	(connected loc-x7-y18 loc-x8-y18)
 	(connected loc-x7-y18 loc-x7-y17)
 	(connected loc-x7-y18 loc-x7-y19)
 	(connected loc-x7-y19 loc-x6-y19)
 	(connected loc-x7-y19 loc-x8-y19)
 	(connected loc-x7-y19 loc-x7-y18)
 	(connected loc-x7-y19 loc-x7-y20)
 	(connected loc-x7-y20 loc-x6-y20)
 	(connected loc-x7-y20 loc-x8-y20)
 	(connected loc-x7-y20 loc-x7-y19)
 	(connected loc-x7-y20 loc-x7-y21)
 	(connected loc-x7-y21 loc-x6-y21)
 	(connected loc-x7-y21 loc-x8-y21)
 	(connected loc-x7-y21 loc-x7-y20)
 	(connected loc-x7-y21 loc-x7-y22)
 	(connected loc-x7-y22 loc-x6-y22)
 	(connected loc-x7-y22 loc-x8-y22)
 	(connected loc-x7-y22 loc-x7-y21)
 	(connected loc-x7-y22 loc-x7-y23)
 	(connected loc-x7-y23 loc-x6-y23)
 	(connected loc-x7-y23 loc-x8-y23)
 	(connected loc-x7-y23 loc-x7-y22)
 	(connected loc-x7-y23 loc-x7-y24)
 	(connected loc-x7-y24 loc-x6-y24)
 	(connected loc-x7-y24 loc-x8-y24)
 	(connected loc-x7-y24 loc-x7-y23)
 	(connected loc-x7-y24 loc-x7-y25)
 	(connected loc-x7-y25 loc-x6-y25)
 	(connected loc-x7-y25 loc-x8-y25)
 	(connected loc-x7-y25 loc-x7-y24)
 	(connected loc-x7-y25 loc-x7-y26)
 	(connected loc-x7-y26 loc-x6-y26)
 	(connected loc-x7-y26 loc-x8-y26)
 	(connected loc-x7-y26 loc-x7-y25)
 	(connected loc-x7-y26 loc-x7-y27)
 	(connected loc-x7-y27 loc-x6-y27)
 	(connected loc-x7-y27 loc-x8-y27)
 	(connected loc-x7-y27 loc-x7-y26)
 	(connected loc-x7-y27 loc-x7-y28)
 	(connected loc-x7-y28 loc-x6-y28)
 	(connected loc-x7-y28 loc-x8-y28)
 	(connected loc-x7-y28 loc-x7-y27)
 	(connected loc-x7-y28 loc-x7-y29)
 	(connected loc-x7-y29 loc-x6-y29)
 	(connected loc-x7-y29 loc-x8-y29)
 	(connected loc-x7-y29 loc-x7-y28)
 	(connected loc-x7-y29 loc-x7-y30)
 	(connected loc-x7-y30 loc-x6-y30)
 	(connected loc-x7-y30 loc-x8-y30)
 	(connected loc-x7-y30 loc-x7-y29)
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
 	(connected loc-x8-y15 loc-x8-y16)
 	(connected loc-x8-y16 loc-x7-y16)
 	(connected loc-x8-y16 loc-x9-y16)
 	(connected loc-x8-y16 loc-x8-y15)
 	(connected loc-x8-y16 loc-x8-y17)
 	(connected loc-x8-y17 loc-x7-y17)
 	(connected loc-x8-y17 loc-x9-y17)
 	(connected loc-x8-y17 loc-x8-y16)
 	(connected loc-x8-y17 loc-x8-y18)
 	(connected loc-x8-y18 loc-x7-y18)
 	(connected loc-x8-y18 loc-x9-y18)
 	(connected loc-x8-y18 loc-x8-y17)
 	(connected loc-x8-y18 loc-x8-y19)
 	(connected loc-x8-y19 loc-x7-y19)
 	(connected loc-x8-y19 loc-x9-y19)
 	(connected loc-x8-y19 loc-x8-y18)
 	(connected loc-x8-y19 loc-x8-y20)
 	(connected loc-x8-y20 loc-x7-y20)
 	(connected loc-x8-y20 loc-x9-y20)
 	(connected loc-x8-y20 loc-x8-y19)
 	(connected loc-x8-y20 loc-x8-y21)
 	(connected loc-x8-y21 loc-x7-y21)
 	(connected loc-x8-y21 loc-x9-y21)
 	(connected loc-x8-y21 loc-x8-y20)
 	(connected loc-x8-y21 loc-x8-y22)
 	(connected loc-x8-y22 loc-x7-y22)
 	(connected loc-x8-y22 loc-x9-y22)
 	(connected loc-x8-y22 loc-x8-y21)
 	(connected loc-x8-y22 loc-x8-y23)
 	(connected loc-x8-y23 loc-x7-y23)
 	(connected loc-x8-y23 loc-x9-y23)
 	(connected loc-x8-y23 loc-x8-y22)
 	(connected loc-x8-y23 loc-x8-y24)
 	(connected loc-x8-y24 loc-x7-y24)
 	(connected loc-x8-y24 loc-x9-y24)
 	(connected loc-x8-y24 loc-x8-y23)
 	(connected loc-x8-y24 loc-x8-y25)
 	(connected loc-x8-y25 loc-x7-y25)
 	(connected loc-x8-y25 loc-x9-y25)
 	(connected loc-x8-y25 loc-x8-y24)
 	(connected loc-x8-y25 loc-x8-y26)
 	(connected loc-x8-y26 loc-x7-y26)
 	(connected loc-x8-y26 loc-x9-y26)
 	(connected loc-x8-y26 loc-x8-y25)
 	(connected loc-x8-y26 loc-x8-y27)
 	(connected loc-x8-y27 loc-x7-y27)
 	(connected loc-x8-y27 loc-x9-y27)
 	(connected loc-x8-y27 loc-x8-y26)
 	(connected loc-x8-y27 loc-x8-y28)
 	(connected loc-x8-y28 loc-x7-y28)
 	(connected loc-x8-y28 loc-x9-y28)
 	(connected loc-x8-y28 loc-x8-y27)
 	(connected loc-x8-y28 loc-x8-y29)
 	(connected loc-x8-y29 loc-x7-y29)
 	(connected loc-x8-y29 loc-x9-y29)
 	(connected loc-x8-y29 loc-x8-y28)
 	(connected loc-x8-y29 loc-x8-y30)
 	(connected loc-x8-y30 loc-x7-y30)
 	(connected loc-x8-y30 loc-x9-y30)
 	(connected loc-x8-y30 loc-x8-y29)
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
 	(connected loc-x9-y15 loc-x9-y16)
 	(connected loc-x9-y16 loc-x8-y16)
 	(connected loc-x9-y16 loc-x10-y16)
 	(connected loc-x9-y16 loc-x9-y15)
 	(connected loc-x9-y16 loc-x9-y17)
 	(connected loc-x9-y17 loc-x8-y17)
 	(connected loc-x9-y17 loc-x10-y17)
 	(connected loc-x9-y17 loc-x9-y16)
 	(connected loc-x9-y17 loc-x9-y18)
 	(connected loc-x9-y18 loc-x8-y18)
 	(connected loc-x9-y18 loc-x10-y18)
 	(connected loc-x9-y18 loc-x9-y17)
 	(connected loc-x9-y18 loc-x9-y19)
 	(connected loc-x9-y19 loc-x8-y19)
 	(connected loc-x9-y19 loc-x10-y19)
 	(connected loc-x9-y19 loc-x9-y18)
 	(connected loc-x9-y19 loc-x9-y20)
 	(connected loc-x9-y20 loc-x8-y20)
 	(connected loc-x9-y20 loc-x10-y20)
 	(connected loc-x9-y20 loc-x9-y19)
 	(connected loc-x9-y20 loc-x9-y21)
 	(connected loc-x9-y21 loc-x8-y21)
 	(connected loc-x9-y21 loc-x10-y21)
 	(connected loc-x9-y21 loc-x9-y20)
 	(connected loc-x9-y21 loc-x9-y22)
 	(connected loc-x9-y22 loc-x8-y22)
 	(connected loc-x9-y22 loc-x10-y22)
 	(connected loc-x9-y22 loc-x9-y21)
 	(connected loc-x9-y22 loc-x9-y23)
 	(connected loc-x9-y23 loc-x8-y23)
 	(connected loc-x9-y23 loc-x10-y23)
 	(connected loc-x9-y23 loc-x9-y22)
 	(connected loc-x9-y23 loc-x9-y24)
 	(connected loc-x9-y24 loc-x8-y24)
 	(connected loc-x9-y24 loc-x10-y24)
 	(connected loc-x9-y24 loc-x9-y23)
 	(connected loc-x9-y24 loc-x9-y25)
 	(connected loc-x9-y25 loc-x8-y25)
 	(connected loc-x9-y25 loc-x10-y25)
 	(connected loc-x9-y25 loc-x9-y24)
 	(connected loc-x9-y25 loc-x9-y26)
 	(connected loc-x9-y26 loc-x8-y26)
 	(connected loc-x9-y26 loc-x10-y26)
 	(connected loc-x9-y26 loc-x9-y25)
 	(connected loc-x9-y26 loc-x9-y27)
 	(connected loc-x9-y27 loc-x8-y27)
 	(connected loc-x9-y27 loc-x10-y27)
 	(connected loc-x9-y27 loc-x9-y26)
 	(connected loc-x9-y27 loc-x9-y28)
 	(connected loc-x9-y28 loc-x8-y28)
 	(connected loc-x9-y28 loc-x10-y28)
 	(connected loc-x9-y28 loc-x9-y27)
 	(connected loc-x9-y28 loc-x9-y29)
 	(connected loc-x9-y29 loc-x8-y29)
 	(connected loc-x9-y29 loc-x10-y29)
 	(connected loc-x9-y29 loc-x9-y28)
 	(connected loc-x9-y29 loc-x9-y30)
 	(connected loc-x9-y30 loc-x8-y30)
 	(connected loc-x9-y30 loc-x10-y30)
 	(connected loc-x9-y30 loc-x9-y29)
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
 	(connected loc-x10-y15 loc-x10-y16)
 	(connected loc-x10-y16 loc-x9-y16)
 	(connected loc-x10-y16 loc-x11-y16)
 	(connected loc-x10-y16 loc-x10-y15)
 	(connected loc-x10-y16 loc-x10-y17)
 	(connected loc-x10-y17 loc-x9-y17)
 	(connected loc-x10-y17 loc-x11-y17)
 	(connected loc-x10-y17 loc-x10-y16)
 	(connected loc-x10-y17 loc-x10-y18)
 	(connected loc-x10-y18 loc-x9-y18)
 	(connected loc-x10-y18 loc-x11-y18)
 	(connected loc-x10-y18 loc-x10-y17)
 	(connected loc-x10-y18 loc-x10-y19)
 	(connected loc-x10-y19 loc-x9-y19)
 	(connected loc-x10-y19 loc-x11-y19)
 	(connected loc-x10-y19 loc-x10-y18)
 	(connected loc-x10-y19 loc-x10-y20)
 	(connected loc-x10-y20 loc-x9-y20)
 	(connected loc-x10-y20 loc-x11-y20)
 	(connected loc-x10-y20 loc-x10-y19)
 	(connected loc-x10-y20 loc-x10-y21)
 	(connected loc-x10-y21 loc-x9-y21)
 	(connected loc-x10-y21 loc-x11-y21)
 	(connected loc-x10-y21 loc-x10-y20)
 	(connected loc-x10-y21 loc-x10-y22)
 	(connected loc-x10-y22 loc-x9-y22)
 	(connected loc-x10-y22 loc-x11-y22)
 	(connected loc-x10-y22 loc-x10-y21)
 	(connected loc-x10-y22 loc-x10-y23)
 	(connected loc-x10-y23 loc-x9-y23)
 	(connected loc-x10-y23 loc-x11-y23)
 	(connected loc-x10-y23 loc-x10-y22)
 	(connected loc-x10-y23 loc-x10-y24)
 	(connected loc-x10-y24 loc-x9-y24)
 	(connected loc-x10-y24 loc-x11-y24)
 	(connected loc-x10-y24 loc-x10-y23)
 	(connected loc-x10-y24 loc-x10-y25)
 	(connected loc-x10-y25 loc-x9-y25)
 	(connected loc-x10-y25 loc-x11-y25)
 	(connected loc-x10-y25 loc-x10-y24)
 	(connected loc-x10-y25 loc-x10-y26)
 	(connected loc-x10-y26 loc-x9-y26)
 	(connected loc-x10-y26 loc-x11-y26)
 	(connected loc-x10-y26 loc-x10-y25)
 	(connected loc-x10-y26 loc-x10-y27)
 	(connected loc-x10-y27 loc-x9-y27)
 	(connected loc-x10-y27 loc-x11-y27)
 	(connected loc-x10-y27 loc-x10-y26)
 	(connected loc-x10-y27 loc-x10-y28)
 	(connected loc-x10-y28 loc-x9-y28)
 	(connected loc-x10-y28 loc-x11-y28)
 	(connected loc-x10-y28 loc-x10-y27)
 	(connected loc-x10-y28 loc-x10-y29)
 	(connected loc-x10-y29 loc-x9-y29)
 	(connected loc-x10-y29 loc-x11-y29)
 	(connected loc-x10-y29 loc-x10-y28)
 	(connected loc-x10-y29 loc-x10-y30)
 	(connected loc-x10-y30 loc-x9-y30)
 	(connected loc-x10-y30 loc-x11-y30)
 	(connected loc-x10-y30 loc-x10-y29)
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
 	(connected loc-x11-y15 loc-x11-y16)
 	(connected loc-x11-y16 loc-x10-y16)
 	(connected loc-x11-y16 loc-x12-y16)
 	(connected loc-x11-y16 loc-x11-y15)
 	(connected loc-x11-y16 loc-x11-y17)
 	(connected loc-x11-y17 loc-x10-y17)
 	(connected loc-x11-y17 loc-x12-y17)
 	(connected loc-x11-y17 loc-x11-y16)
 	(connected loc-x11-y17 loc-x11-y18)
 	(connected loc-x11-y18 loc-x10-y18)
 	(connected loc-x11-y18 loc-x12-y18)
 	(connected loc-x11-y18 loc-x11-y17)
 	(connected loc-x11-y18 loc-x11-y19)
 	(connected loc-x11-y19 loc-x10-y19)
 	(connected loc-x11-y19 loc-x12-y19)
 	(connected loc-x11-y19 loc-x11-y18)
 	(connected loc-x11-y19 loc-x11-y20)
 	(connected loc-x11-y20 loc-x10-y20)
 	(connected loc-x11-y20 loc-x12-y20)
 	(connected loc-x11-y20 loc-x11-y19)
 	(connected loc-x11-y20 loc-x11-y21)
 	(connected loc-x11-y21 loc-x10-y21)
 	(connected loc-x11-y21 loc-x12-y21)
 	(connected loc-x11-y21 loc-x11-y20)
 	(connected loc-x11-y21 loc-x11-y22)
 	(connected loc-x11-y22 loc-x10-y22)
 	(connected loc-x11-y22 loc-x12-y22)
 	(connected loc-x11-y22 loc-x11-y21)
 	(connected loc-x11-y22 loc-x11-y23)
 	(connected loc-x11-y23 loc-x10-y23)
 	(connected loc-x11-y23 loc-x12-y23)
 	(connected loc-x11-y23 loc-x11-y22)
 	(connected loc-x11-y23 loc-x11-y24)
 	(connected loc-x11-y24 loc-x10-y24)
 	(connected loc-x11-y24 loc-x12-y24)
 	(connected loc-x11-y24 loc-x11-y23)
 	(connected loc-x11-y24 loc-x11-y25)
 	(connected loc-x11-y25 loc-x10-y25)
 	(connected loc-x11-y25 loc-x12-y25)
 	(connected loc-x11-y25 loc-x11-y24)
 	(connected loc-x11-y25 loc-x11-y26)
 	(connected loc-x11-y26 loc-x10-y26)
 	(connected loc-x11-y26 loc-x12-y26)
 	(connected loc-x11-y26 loc-x11-y25)
 	(connected loc-x11-y26 loc-x11-y27)
 	(connected loc-x11-y27 loc-x10-y27)
 	(connected loc-x11-y27 loc-x12-y27)
 	(connected loc-x11-y27 loc-x11-y26)
 	(connected loc-x11-y27 loc-x11-y28)
 	(connected loc-x11-y28 loc-x10-y28)
 	(connected loc-x11-y28 loc-x12-y28)
 	(connected loc-x11-y28 loc-x11-y27)
 	(connected loc-x11-y28 loc-x11-y29)
 	(connected loc-x11-y29 loc-x10-y29)
 	(connected loc-x11-y29 loc-x12-y29)
 	(connected loc-x11-y29 loc-x11-y28)
 	(connected loc-x11-y29 loc-x11-y30)
 	(connected loc-x11-y30 loc-x10-y30)
 	(connected loc-x11-y30 loc-x12-y30)
 	(connected loc-x11-y30 loc-x11-y29)
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
 	(connected loc-x12-y15 loc-x12-y16)
 	(connected loc-x12-y16 loc-x11-y16)
 	(connected loc-x12-y16 loc-x13-y16)
 	(connected loc-x12-y16 loc-x12-y15)
 	(connected loc-x12-y16 loc-x12-y17)
 	(connected loc-x12-y17 loc-x11-y17)
 	(connected loc-x12-y17 loc-x13-y17)
 	(connected loc-x12-y17 loc-x12-y16)
 	(connected loc-x12-y17 loc-x12-y18)
 	(connected loc-x12-y18 loc-x11-y18)
 	(connected loc-x12-y18 loc-x13-y18)
 	(connected loc-x12-y18 loc-x12-y17)
 	(connected loc-x12-y18 loc-x12-y19)
 	(connected loc-x12-y19 loc-x11-y19)
 	(connected loc-x12-y19 loc-x13-y19)
 	(connected loc-x12-y19 loc-x12-y18)
 	(connected loc-x12-y19 loc-x12-y20)
 	(connected loc-x12-y20 loc-x11-y20)
 	(connected loc-x12-y20 loc-x13-y20)
 	(connected loc-x12-y20 loc-x12-y19)
 	(connected loc-x12-y20 loc-x12-y21)
 	(connected loc-x12-y21 loc-x11-y21)
 	(connected loc-x12-y21 loc-x13-y21)
 	(connected loc-x12-y21 loc-x12-y20)
 	(connected loc-x12-y21 loc-x12-y22)
 	(connected loc-x12-y22 loc-x11-y22)
 	(connected loc-x12-y22 loc-x13-y22)
 	(connected loc-x12-y22 loc-x12-y21)
 	(connected loc-x12-y22 loc-x12-y23)
 	(connected loc-x12-y23 loc-x11-y23)
 	(connected loc-x12-y23 loc-x13-y23)
 	(connected loc-x12-y23 loc-x12-y22)
 	(connected loc-x12-y23 loc-x12-y24)
 	(connected loc-x12-y24 loc-x11-y24)
 	(connected loc-x12-y24 loc-x13-y24)
 	(connected loc-x12-y24 loc-x12-y23)
 	(connected loc-x12-y24 loc-x12-y25)
 	(connected loc-x12-y25 loc-x11-y25)
 	(connected loc-x12-y25 loc-x13-y25)
 	(connected loc-x12-y25 loc-x12-y24)
 	(connected loc-x12-y25 loc-x12-y26)
 	(connected loc-x12-y26 loc-x11-y26)
 	(connected loc-x12-y26 loc-x13-y26)
 	(connected loc-x12-y26 loc-x12-y25)
 	(connected loc-x12-y26 loc-x12-y27)
 	(connected loc-x12-y27 loc-x11-y27)
 	(connected loc-x12-y27 loc-x13-y27)
 	(connected loc-x12-y27 loc-x12-y26)
 	(connected loc-x12-y27 loc-x12-y28)
 	(connected loc-x12-y28 loc-x11-y28)
 	(connected loc-x12-y28 loc-x13-y28)
 	(connected loc-x12-y28 loc-x12-y27)
 	(connected loc-x12-y28 loc-x12-y29)
 	(connected loc-x12-y29 loc-x11-y29)
 	(connected loc-x12-y29 loc-x13-y29)
 	(connected loc-x12-y29 loc-x12-y28)
 	(connected loc-x12-y29 loc-x12-y30)
 	(connected loc-x12-y30 loc-x11-y30)
 	(connected loc-x12-y30 loc-x13-y30)
 	(connected loc-x12-y30 loc-x12-y29)
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
 	(connected loc-x13-y15 loc-x13-y16)
 	(connected loc-x13-y16 loc-x12-y16)
 	(connected loc-x13-y16 loc-x14-y16)
 	(connected loc-x13-y16 loc-x13-y15)
 	(connected loc-x13-y16 loc-x13-y17)
 	(connected loc-x13-y17 loc-x12-y17)
 	(connected loc-x13-y17 loc-x14-y17)
 	(connected loc-x13-y17 loc-x13-y16)
 	(connected loc-x13-y17 loc-x13-y18)
 	(connected loc-x13-y18 loc-x12-y18)
 	(connected loc-x13-y18 loc-x14-y18)
 	(connected loc-x13-y18 loc-x13-y17)
 	(connected loc-x13-y18 loc-x13-y19)
 	(connected loc-x13-y19 loc-x12-y19)
 	(connected loc-x13-y19 loc-x14-y19)
 	(connected loc-x13-y19 loc-x13-y18)
 	(connected loc-x13-y19 loc-x13-y20)
 	(connected loc-x13-y20 loc-x12-y20)
 	(connected loc-x13-y20 loc-x14-y20)
 	(connected loc-x13-y20 loc-x13-y19)
 	(connected loc-x13-y20 loc-x13-y21)
 	(connected loc-x13-y21 loc-x12-y21)
 	(connected loc-x13-y21 loc-x14-y21)
 	(connected loc-x13-y21 loc-x13-y20)
 	(connected loc-x13-y21 loc-x13-y22)
 	(connected loc-x13-y22 loc-x12-y22)
 	(connected loc-x13-y22 loc-x14-y22)
 	(connected loc-x13-y22 loc-x13-y21)
 	(connected loc-x13-y22 loc-x13-y23)
 	(connected loc-x13-y23 loc-x12-y23)
 	(connected loc-x13-y23 loc-x14-y23)
 	(connected loc-x13-y23 loc-x13-y22)
 	(connected loc-x13-y23 loc-x13-y24)
 	(connected loc-x13-y24 loc-x12-y24)
 	(connected loc-x13-y24 loc-x14-y24)
 	(connected loc-x13-y24 loc-x13-y23)
 	(connected loc-x13-y24 loc-x13-y25)
 	(connected loc-x13-y25 loc-x12-y25)
 	(connected loc-x13-y25 loc-x14-y25)
 	(connected loc-x13-y25 loc-x13-y24)
 	(connected loc-x13-y25 loc-x13-y26)
 	(connected loc-x13-y26 loc-x12-y26)
 	(connected loc-x13-y26 loc-x14-y26)
 	(connected loc-x13-y26 loc-x13-y25)
 	(connected loc-x13-y26 loc-x13-y27)
 	(connected loc-x13-y27 loc-x12-y27)
 	(connected loc-x13-y27 loc-x14-y27)
 	(connected loc-x13-y27 loc-x13-y26)
 	(connected loc-x13-y27 loc-x13-y28)
 	(connected loc-x13-y28 loc-x12-y28)
 	(connected loc-x13-y28 loc-x14-y28)
 	(connected loc-x13-y28 loc-x13-y27)
 	(connected loc-x13-y28 loc-x13-y29)
 	(connected loc-x13-y29 loc-x12-y29)
 	(connected loc-x13-y29 loc-x14-y29)
 	(connected loc-x13-y29 loc-x13-y28)
 	(connected loc-x13-y29 loc-x13-y30)
 	(connected loc-x13-y30 loc-x12-y30)
 	(connected loc-x13-y30 loc-x14-y30)
 	(connected loc-x13-y30 loc-x13-y29)
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
 	(connected loc-x14-y15 loc-x14-y16)
 	(connected loc-x14-y16 loc-x13-y16)
 	(connected loc-x14-y16 loc-x15-y16)
 	(connected loc-x14-y16 loc-x14-y15)
 	(connected loc-x14-y16 loc-x14-y17)
 	(connected loc-x14-y17 loc-x13-y17)
 	(connected loc-x14-y17 loc-x15-y17)
 	(connected loc-x14-y17 loc-x14-y16)
 	(connected loc-x14-y17 loc-x14-y18)
 	(connected loc-x14-y18 loc-x13-y18)
 	(connected loc-x14-y18 loc-x15-y18)
 	(connected loc-x14-y18 loc-x14-y17)
 	(connected loc-x14-y18 loc-x14-y19)
 	(connected loc-x14-y19 loc-x13-y19)
 	(connected loc-x14-y19 loc-x15-y19)
 	(connected loc-x14-y19 loc-x14-y18)
 	(connected loc-x14-y19 loc-x14-y20)
 	(connected loc-x14-y20 loc-x13-y20)
 	(connected loc-x14-y20 loc-x15-y20)
 	(connected loc-x14-y20 loc-x14-y19)
 	(connected loc-x14-y20 loc-x14-y21)
 	(connected loc-x14-y21 loc-x13-y21)
 	(connected loc-x14-y21 loc-x15-y21)
 	(connected loc-x14-y21 loc-x14-y20)
 	(connected loc-x14-y21 loc-x14-y22)
 	(connected loc-x14-y22 loc-x13-y22)
 	(connected loc-x14-y22 loc-x15-y22)
 	(connected loc-x14-y22 loc-x14-y21)
 	(connected loc-x14-y22 loc-x14-y23)
 	(connected loc-x14-y23 loc-x13-y23)
 	(connected loc-x14-y23 loc-x15-y23)
 	(connected loc-x14-y23 loc-x14-y22)
 	(connected loc-x14-y23 loc-x14-y24)
 	(connected loc-x14-y24 loc-x13-y24)
 	(connected loc-x14-y24 loc-x15-y24)
 	(connected loc-x14-y24 loc-x14-y23)
 	(connected loc-x14-y24 loc-x14-y25)
 	(connected loc-x14-y25 loc-x13-y25)
 	(connected loc-x14-y25 loc-x15-y25)
 	(connected loc-x14-y25 loc-x14-y24)
 	(connected loc-x14-y25 loc-x14-y26)
 	(connected loc-x14-y26 loc-x13-y26)
 	(connected loc-x14-y26 loc-x15-y26)
 	(connected loc-x14-y26 loc-x14-y25)
 	(connected loc-x14-y26 loc-x14-y27)
 	(connected loc-x14-y27 loc-x13-y27)
 	(connected loc-x14-y27 loc-x15-y27)
 	(connected loc-x14-y27 loc-x14-y26)
 	(connected loc-x14-y27 loc-x14-y28)
 	(connected loc-x14-y28 loc-x13-y28)
 	(connected loc-x14-y28 loc-x15-y28)
 	(connected loc-x14-y28 loc-x14-y27)
 	(connected loc-x14-y28 loc-x14-y29)
 	(connected loc-x14-y29 loc-x13-y29)
 	(connected loc-x14-y29 loc-x15-y29)
 	(connected loc-x14-y29 loc-x14-y28)
 	(connected loc-x14-y29 loc-x14-y30)
 	(connected loc-x14-y30 loc-x13-y30)
 	(connected loc-x14-y30 loc-x15-y30)
 	(connected loc-x14-y30 loc-x14-y29)
 	(connected loc-x15-y0 loc-x14-y0)
 	(connected loc-x15-y0 loc-x16-y0)
 	(connected loc-x15-y0 loc-x15-y1)
 	(connected loc-x15-y1 loc-x14-y1)
 	(connected loc-x15-y1 loc-x16-y1)
 	(connected loc-x15-y1 loc-x15-y0)
 	(connected loc-x15-y1 loc-x15-y2)
 	(connected loc-x15-y2 loc-x14-y2)
 	(connected loc-x15-y2 loc-x16-y2)
 	(connected loc-x15-y2 loc-x15-y1)
 	(connected loc-x15-y2 loc-x15-y3)
 	(connected loc-x15-y3 loc-x14-y3)
 	(connected loc-x15-y3 loc-x16-y3)
 	(connected loc-x15-y3 loc-x15-y2)
 	(connected loc-x15-y3 loc-x15-y4)
 	(connected loc-x15-y4 loc-x14-y4)
 	(connected loc-x15-y4 loc-x16-y4)
 	(connected loc-x15-y4 loc-x15-y3)
 	(connected loc-x15-y4 loc-x15-y5)
 	(connected loc-x15-y5 loc-x14-y5)
 	(connected loc-x15-y5 loc-x16-y5)
 	(connected loc-x15-y5 loc-x15-y4)
 	(connected loc-x15-y5 loc-x15-y6)
 	(connected loc-x15-y6 loc-x14-y6)
 	(connected loc-x15-y6 loc-x16-y6)
 	(connected loc-x15-y6 loc-x15-y5)
 	(connected loc-x15-y6 loc-x15-y7)
 	(connected loc-x15-y7 loc-x14-y7)
 	(connected loc-x15-y7 loc-x16-y7)
 	(connected loc-x15-y7 loc-x15-y6)
 	(connected loc-x15-y7 loc-x15-y8)
 	(connected loc-x15-y8 loc-x14-y8)
 	(connected loc-x15-y8 loc-x16-y8)
 	(connected loc-x15-y8 loc-x15-y7)
 	(connected loc-x15-y8 loc-x15-y9)
 	(connected loc-x15-y9 loc-x14-y9)
 	(connected loc-x15-y9 loc-x16-y9)
 	(connected loc-x15-y9 loc-x15-y8)
 	(connected loc-x15-y9 loc-x15-y10)
 	(connected loc-x15-y10 loc-x14-y10)
 	(connected loc-x15-y10 loc-x16-y10)
 	(connected loc-x15-y10 loc-x15-y9)
 	(connected loc-x15-y10 loc-x15-y11)
 	(connected loc-x15-y11 loc-x14-y11)
 	(connected loc-x15-y11 loc-x16-y11)
 	(connected loc-x15-y11 loc-x15-y10)
 	(connected loc-x15-y11 loc-x15-y12)
 	(connected loc-x15-y12 loc-x14-y12)
 	(connected loc-x15-y12 loc-x16-y12)
 	(connected loc-x15-y12 loc-x15-y11)
 	(connected loc-x15-y12 loc-x15-y13)
 	(connected loc-x15-y13 loc-x14-y13)
 	(connected loc-x15-y13 loc-x16-y13)
 	(connected loc-x15-y13 loc-x15-y12)
 	(connected loc-x15-y13 loc-x15-y14)
 	(connected loc-x15-y14 loc-x14-y14)
 	(connected loc-x15-y14 loc-x16-y14)
 	(connected loc-x15-y14 loc-x15-y13)
 	(connected loc-x15-y14 loc-x15-y15)
 	(connected loc-x15-y15 loc-x14-y15)
 	(connected loc-x15-y15 loc-x16-y15)
 	(connected loc-x15-y15 loc-x15-y14)
 	(connected loc-x15-y15 loc-x15-y16)
 	(connected loc-x15-y16 loc-x14-y16)
 	(connected loc-x15-y16 loc-x16-y16)
 	(connected loc-x15-y16 loc-x15-y15)
 	(connected loc-x15-y16 loc-x15-y17)
 	(connected loc-x15-y17 loc-x14-y17)
 	(connected loc-x15-y17 loc-x16-y17)
 	(connected loc-x15-y17 loc-x15-y16)
 	(connected loc-x15-y17 loc-x15-y18)
 	(connected loc-x15-y18 loc-x14-y18)
 	(connected loc-x15-y18 loc-x16-y18)
 	(connected loc-x15-y18 loc-x15-y17)
 	(connected loc-x15-y18 loc-x15-y19)
 	(connected loc-x15-y19 loc-x14-y19)
 	(connected loc-x15-y19 loc-x16-y19)
 	(connected loc-x15-y19 loc-x15-y18)
 	(connected loc-x15-y19 loc-x15-y20)
 	(connected loc-x15-y20 loc-x14-y20)
 	(connected loc-x15-y20 loc-x16-y20)
 	(connected loc-x15-y20 loc-x15-y19)
 	(connected loc-x15-y20 loc-x15-y21)
 	(connected loc-x15-y21 loc-x14-y21)
 	(connected loc-x15-y21 loc-x16-y21)
 	(connected loc-x15-y21 loc-x15-y20)
 	(connected loc-x15-y21 loc-x15-y22)
 	(connected loc-x15-y22 loc-x14-y22)
 	(connected loc-x15-y22 loc-x16-y22)
 	(connected loc-x15-y22 loc-x15-y21)
 	(connected loc-x15-y22 loc-x15-y23)
 	(connected loc-x15-y23 loc-x14-y23)
 	(connected loc-x15-y23 loc-x16-y23)
 	(connected loc-x15-y23 loc-x15-y22)
 	(connected loc-x15-y23 loc-x15-y24)
 	(connected loc-x15-y24 loc-x14-y24)
 	(connected loc-x15-y24 loc-x16-y24)
 	(connected loc-x15-y24 loc-x15-y23)
 	(connected loc-x15-y24 loc-x15-y25)
 	(connected loc-x15-y25 loc-x14-y25)
 	(connected loc-x15-y25 loc-x16-y25)
 	(connected loc-x15-y25 loc-x15-y24)
 	(connected loc-x15-y25 loc-x15-y26)
 	(connected loc-x15-y26 loc-x14-y26)
 	(connected loc-x15-y26 loc-x16-y26)
 	(connected loc-x15-y26 loc-x15-y25)
 	(connected loc-x15-y26 loc-x15-y27)
 	(connected loc-x15-y27 loc-x14-y27)
 	(connected loc-x15-y27 loc-x16-y27)
 	(connected loc-x15-y27 loc-x15-y26)
 	(connected loc-x15-y27 loc-x15-y28)
 	(connected loc-x15-y28 loc-x14-y28)
 	(connected loc-x15-y28 loc-x16-y28)
 	(connected loc-x15-y28 loc-x15-y27)
 	(connected loc-x15-y28 loc-x15-y29)
 	(connected loc-x15-y29 loc-x14-y29)
 	(connected loc-x15-y29 loc-x16-y29)
 	(connected loc-x15-y29 loc-x15-y28)
 	(connected loc-x15-y29 loc-x15-y30)
 	(connected loc-x15-y30 loc-x14-y30)
 	(connected loc-x15-y30 loc-x16-y30)
 	(connected loc-x15-y30 loc-x15-y29)
 	(connected loc-x16-y0 loc-x15-y0)
 	(connected loc-x16-y0 loc-x17-y0)
 	(connected loc-x16-y0 loc-x16-y1)
 	(connected loc-x16-y1 loc-x15-y1)
 	(connected loc-x16-y1 loc-x17-y1)
 	(connected loc-x16-y1 loc-x16-y0)
 	(connected loc-x16-y1 loc-x16-y2)
 	(connected loc-x16-y2 loc-x15-y2)
 	(connected loc-x16-y2 loc-x17-y2)
 	(connected loc-x16-y2 loc-x16-y1)
 	(connected loc-x16-y2 loc-x16-y3)
 	(connected loc-x16-y3 loc-x15-y3)
 	(connected loc-x16-y3 loc-x17-y3)
 	(connected loc-x16-y3 loc-x16-y2)
 	(connected loc-x16-y3 loc-x16-y4)
 	(connected loc-x16-y4 loc-x15-y4)
 	(connected loc-x16-y4 loc-x17-y4)
 	(connected loc-x16-y4 loc-x16-y3)
 	(connected loc-x16-y4 loc-x16-y5)
 	(connected loc-x16-y5 loc-x15-y5)
 	(connected loc-x16-y5 loc-x17-y5)
 	(connected loc-x16-y5 loc-x16-y4)
 	(connected loc-x16-y5 loc-x16-y6)
 	(connected loc-x16-y6 loc-x15-y6)
 	(connected loc-x16-y6 loc-x17-y6)
 	(connected loc-x16-y6 loc-x16-y5)
 	(connected loc-x16-y6 loc-x16-y7)
 	(connected loc-x16-y7 loc-x15-y7)
 	(connected loc-x16-y7 loc-x17-y7)
 	(connected loc-x16-y7 loc-x16-y6)
 	(connected loc-x16-y7 loc-x16-y8)
 	(connected loc-x16-y8 loc-x15-y8)
 	(connected loc-x16-y8 loc-x17-y8)
 	(connected loc-x16-y8 loc-x16-y7)
 	(connected loc-x16-y8 loc-x16-y9)
 	(connected loc-x16-y9 loc-x15-y9)
 	(connected loc-x16-y9 loc-x17-y9)
 	(connected loc-x16-y9 loc-x16-y8)
 	(connected loc-x16-y9 loc-x16-y10)
 	(connected loc-x16-y10 loc-x15-y10)
 	(connected loc-x16-y10 loc-x17-y10)
 	(connected loc-x16-y10 loc-x16-y9)
 	(connected loc-x16-y10 loc-x16-y11)
 	(connected loc-x16-y11 loc-x15-y11)
 	(connected loc-x16-y11 loc-x17-y11)
 	(connected loc-x16-y11 loc-x16-y10)
 	(connected loc-x16-y11 loc-x16-y12)
 	(connected loc-x16-y12 loc-x15-y12)
 	(connected loc-x16-y12 loc-x17-y12)
 	(connected loc-x16-y12 loc-x16-y11)
 	(connected loc-x16-y12 loc-x16-y13)
 	(connected loc-x16-y13 loc-x15-y13)
 	(connected loc-x16-y13 loc-x17-y13)
 	(connected loc-x16-y13 loc-x16-y12)
 	(connected loc-x16-y13 loc-x16-y14)
 	(connected loc-x16-y14 loc-x15-y14)
 	(connected loc-x16-y14 loc-x17-y14)
 	(connected loc-x16-y14 loc-x16-y13)
 	(connected loc-x16-y14 loc-x16-y15)
 	(connected loc-x16-y15 loc-x15-y15)
 	(connected loc-x16-y15 loc-x17-y15)
 	(connected loc-x16-y15 loc-x16-y14)
 	(connected loc-x16-y15 loc-x16-y16)
 	(connected loc-x16-y16 loc-x15-y16)
 	(connected loc-x16-y16 loc-x17-y16)
 	(connected loc-x16-y16 loc-x16-y15)
 	(connected loc-x16-y16 loc-x16-y17)
 	(connected loc-x16-y17 loc-x15-y17)
 	(connected loc-x16-y17 loc-x17-y17)
 	(connected loc-x16-y17 loc-x16-y16)
 	(connected loc-x16-y17 loc-x16-y18)
 	(connected loc-x16-y18 loc-x15-y18)
 	(connected loc-x16-y18 loc-x17-y18)
 	(connected loc-x16-y18 loc-x16-y17)
 	(connected loc-x16-y18 loc-x16-y19)
 	(connected loc-x16-y19 loc-x15-y19)
 	(connected loc-x16-y19 loc-x17-y19)
 	(connected loc-x16-y19 loc-x16-y18)
 	(connected loc-x16-y19 loc-x16-y20)
 	(connected loc-x16-y20 loc-x15-y20)
 	(connected loc-x16-y20 loc-x17-y20)
 	(connected loc-x16-y20 loc-x16-y19)
 	(connected loc-x16-y20 loc-x16-y21)
 	(connected loc-x16-y21 loc-x15-y21)
 	(connected loc-x16-y21 loc-x17-y21)
 	(connected loc-x16-y21 loc-x16-y20)
 	(connected loc-x16-y21 loc-x16-y22)
 	(connected loc-x16-y22 loc-x15-y22)
 	(connected loc-x16-y22 loc-x17-y22)
 	(connected loc-x16-y22 loc-x16-y21)
 	(connected loc-x16-y22 loc-x16-y23)
 	(connected loc-x16-y23 loc-x15-y23)
 	(connected loc-x16-y23 loc-x17-y23)
 	(connected loc-x16-y23 loc-x16-y22)
 	(connected loc-x16-y23 loc-x16-y24)
 	(connected loc-x16-y24 loc-x15-y24)
 	(connected loc-x16-y24 loc-x17-y24)
 	(connected loc-x16-y24 loc-x16-y23)
 	(connected loc-x16-y24 loc-x16-y25)
 	(connected loc-x16-y25 loc-x15-y25)
 	(connected loc-x16-y25 loc-x17-y25)
 	(connected loc-x16-y25 loc-x16-y24)
 	(connected loc-x16-y25 loc-x16-y26)
 	(connected loc-x16-y26 loc-x15-y26)
 	(connected loc-x16-y26 loc-x17-y26)
 	(connected loc-x16-y26 loc-x16-y25)
 	(connected loc-x16-y26 loc-x16-y27)
 	(connected loc-x16-y27 loc-x15-y27)
 	(connected loc-x16-y27 loc-x17-y27)
 	(connected loc-x16-y27 loc-x16-y26)
 	(connected loc-x16-y27 loc-x16-y28)
 	(connected loc-x16-y28 loc-x15-y28)
 	(connected loc-x16-y28 loc-x17-y28)
 	(connected loc-x16-y28 loc-x16-y27)
 	(connected loc-x16-y28 loc-x16-y29)
 	(connected loc-x16-y29 loc-x15-y29)
 	(connected loc-x16-y29 loc-x17-y29)
 	(connected loc-x16-y29 loc-x16-y28)
 	(connected loc-x16-y29 loc-x16-y30)
 	(connected loc-x16-y30 loc-x15-y30)
 	(connected loc-x16-y30 loc-x17-y30)
 	(connected loc-x16-y30 loc-x16-y29)
 	(connected loc-x17-y0 loc-x16-y0)
 	(connected loc-x17-y0 loc-x18-y0)
 	(connected loc-x17-y0 loc-x17-y1)
 	(connected loc-x17-y1 loc-x16-y1)
 	(connected loc-x17-y1 loc-x18-y1)
 	(connected loc-x17-y1 loc-x17-y0)
 	(connected loc-x17-y1 loc-x17-y2)
 	(connected loc-x17-y2 loc-x16-y2)
 	(connected loc-x17-y2 loc-x18-y2)
 	(connected loc-x17-y2 loc-x17-y1)
 	(connected loc-x17-y2 loc-x17-y3)
 	(connected loc-x17-y3 loc-x16-y3)
 	(connected loc-x17-y3 loc-x18-y3)
 	(connected loc-x17-y3 loc-x17-y2)
 	(connected loc-x17-y3 loc-x17-y4)
 	(connected loc-x17-y4 loc-x16-y4)
 	(connected loc-x17-y4 loc-x18-y4)
 	(connected loc-x17-y4 loc-x17-y3)
 	(connected loc-x17-y4 loc-x17-y5)
 	(connected loc-x17-y5 loc-x16-y5)
 	(connected loc-x17-y5 loc-x18-y5)
 	(connected loc-x17-y5 loc-x17-y4)
 	(connected loc-x17-y5 loc-x17-y6)
 	(connected loc-x17-y6 loc-x16-y6)
 	(connected loc-x17-y6 loc-x18-y6)
 	(connected loc-x17-y6 loc-x17-y5)
 	(connected loc-x17-y6 loc-x17-y7)
 	(connected loc-x17-y7 loc-x16-y7)
 	(connected loc-x17-y7 loc-x18-y7)
 	(connected loc-x17-y7 loc-x17-y6)
 	(connected loc-x17-y7 loc-x17-y8)
 	(connected loc-x17-y8 loc-x16-y8)
 	(connected loc-x17-y8 loc-x18-y8)
 	(connected loc-x17-y8 loc-x17-y7)
 	(connected loc-x17-y8 loc-x17-y9)
 	(connected loc-x17-y9 loc-x16-y9)
 	(connected loc-x17-y9 loc-x18-y9)
 	(connected loc-x17-y9 loc-x17-y8)
 	(connected loc-x17-y9 loc-x17-y10)
 	(connected loc-x17-y10 loc-x16-y10)
 	(connected loc-x17-y10 loc-x18-y10)
 	(connected loc-x17-y10 loc-x17-y9)
 	(connected loc-x17-y10 loc-x17-y11)
 	(connected loc-x17-y11 loc-x16-y11)
 	(connected loc-x17-y11 loc-x18-y11)
 	(connected loc-x17-y11 loc-x17-y10)
 	(connected loc-x17-y11 loc-x17-y12)
 	(connected loc-x17-y12 loc-x16-y12)
 	(connected loc-x17-y12 loc-x18-y12)
 	(connected loc-x17-y12 loc-x17-y11)
 	(connected loc-x17-y12 loc-x17-y13)
 	(connected loc-x17-y13 loc-x16-y13)
 	(connected loc-x17-y13 loc-x18-y13)
 	(connected loc-x17-y13 loc-x17-y12)
 	(connected loc-x17-y13 loc-x17-y14)
 	(connected loc-x17-y14 loc-x16-y14)
 	(connected loc-x17-y14 loc-x18-y14)
 	(connected loc-x17-y14 loc-x17-y13)
 	(connected loc-x17-y14 loc-x17-y15)
 	(connected loc-x17-y15 loc-x16-y15)
 	(connected loc-x17-y15 loc-x18-y15)
 	(connected loc-x17-y15 loc-x17-y14)
 	(connected loc-x17-y15 loc-x17-y16)
 	(connected loc-x17-y16 loc-x16-y16)
 	(connected loc-x17-y16 loc-x18-y16)
 	(connected loc-x17-y16 loc-x17-y15)
 	(connected loc-x17-y16 loc-x17-y17)
 	(connected loc-x17-y17 loc-x16-y17)
 	(connected loc-x17-y17 loc-x18-y17)
 	(connected loc-x17-y17 loc-x17-y16)
 	(connected loc-x17-y17 loc-x17-y18)
 	(connected loc-x17-y18 loc-x16-y18)
 	(connected loc-x17-y18 loc-x18-y18)
 	(connected loc-x17-y18 loc-x17-y17)
 	(connected loc-x17-y18 loc-x17-y19)
 	(connected loc-x17-y19 loc-x16-y19)
 	(connected loc-x17-y19 loc-x18-y19)
 	(connected loc-x17-y19 loc-x17-y18)
 	(connected loc-x17-y19 loc-x17-y20)
 	(connected loc-x17-y20 loc-x16-y20)
 	(connected loc-x17-y20 loc-x18-y20)
 	(connected loc-x17-y20 loc-x17-y19)
 	(connected loc-x17-y20 loc-x17-y21)
 	(connected loc-x17-y21 loc-x16-y21)
 	(connected loc-x17-y21 loc-x18-y21)
 	(connected loc-x17-y21 loc-x17-y20)
 	(connected loc-x17-y21 loc-x17-y22)
 	(connected loc-x17-y22 loc-x16-y22)
 	(connected loc-x17-y22 loc-x18-y22)
 	(connected loc-x17-y22 loc-x17-y21)
 	(connected loc-x17-y22 loc-x17-y23)
 	(connected loc-x17-y23 loc-x16-y23)
 	(connected loc-x17-y23 loc-x18-y23)
 	(connected loc-x17-y23 loc-x17-y22)
 	(connected loc-x17-y23 loc-x17-y24)
 	(connected loc-x17-y24 loc-x16-y24)
 	(connected loc-x17-y24 loc-x18-y24)
 	(connected loc-x17-y24 loc-x17-y23)
 	(connected loc-x17-y24 loc-x17-y25)
 	(connected loc-x17-y25 loc-x16-y25)
 	(connected loc-x17-y25 loc-x18-y25)
 	(connected loc-x17-y25 loc-x17-y24)
 	(connected loc-x17-y25 loc-x17-y26)
 	(connected loc-x17-y26 loc-x16-y26)
 	(connected loc-x17-y26 loc-x18-y26)
 	(connected loc-x17-y26 loc-x17-y25)
 	(connected loc-x17-y26 loc-x17-y27)
 	(connected loc-x17-y27 loc-x16-y27)
 	(connected loc-x17-y27 loc-x18-y27)
 	(connected loc-x17-y27 loc-x17-y26)
 	(connected loc-x17-y27 loc-x17-y28)
 	(connected loc-x17-y28 loc-x16-y28)
 	(connected loc-x17-y28 loc-x18-y28)
 	(connected loc-x17-y28 loc-x17-y27)
 	(connected loc-x17-y28 loc-x17-y29)
 	(connected loc-x17-y29 loc-x16-y29)
 	(connected loc-x17-y29 loc-x18-y29)
 	(connected loc-x17-y29 loc-x17-y28)
 	(connected loc-x17-y29 loc-x17-y30)
 	(connected loc-x17-y30 loc-x16-y30)
 	(connected loc-x17-y30 loc-x18-y30)
 	(connected loc-x17-y30 loc-x17-y29)
 	(connected loc-x18-y0 loc-x17-y0)
 	(connected loc-x18-y0 loc-x19-y0)
 	(connected loc-x18-y0 loc-x18-y1)
 	(connected loc-x18-y1 loc-x17-y1)
 	(connected loc-x18-y1 loc-x19-y1)
 	(connected loc-x18-y1 loc-x18-y0)
 	(connected loc-x18-y1 loc-x18-y2)
 	(connected loc-x18-y2 loc-x17-y2)
 	(connected loc-x18-y2 loc-x19-y2)
 	(connected loc-x18-y2 loc-x18-y1)
 	(connected loc-x18-y2 loc-x18-y3)
 	(connected loc-x18-y3 loc-x17-y3)
 	(connected loc-x18-y3 loc-x19-y3)
 	(connected loc-x18-y3 loc-x18-y2)
 	(connected loc-x18-y3 loc-x18-y4)
 	(connected loc-x18-y4 loc-x17-y4)
 	(connected loc-x18-y4 loc-x19-y4)
 	(connected loc-x18-y4 loc-x18-y3)
 	(connected loc-x18-y4 loc-x18-y5)
 	(connected loc-x18-y5 loc-x17-y5)
 	(connected loc-x18-y5 loc-x19-y5)
 	(connected loc-x18-y5 loc-x18-y4)
 	(connected loc-x18-y5 loc-x18-y6)
 	(connected loc-x18-y6 loc-x17-y6)
 	(connected loc-x18-y6 loc-x19-y6)
 	(connected loc-x18-y6 loc-x18-y5)
 	(connected loc-x18-y6 loc-x18-y7)
 	(connected loc-x18-y7 loc-x17-y7)
 	(connected loc-x18-y7 loc-x19-y7)
 	(connected loc-x18-y7 loc-x18-y6)
 	(connected loc-x18-y7 loc-x18-y8)
 	(connected loc-x18-y8 loc-x17-y8)
 	(connected loc-x18-y8 loc-x19-y8)
 	(connected loc-x18-y8 loc-x18-y7)
 	(connected loc-x18-y8 loc-x18-y9)
 	(connected loc-x18-y9 loc-x17-y9)
 	(connected loc-x18-y9 loc-x19-y9)
 	(connected loc-x18-y9 loc-x18-y8)
 	(connected loc-x18-y9 loc-x18-y10)
 	(connected loc-x18-y10 loc-x17-y10)
 	(connected loc-x18-y10 loc-x19-y10)
 	(connected loc-x18-y10 loc-x18-y9)
 	(connected loc-x18-y10 loc-x18-y11)
 	(connected loc-x18-y11 loc-x17-y11)
 	(connected loc-x18-y11 loc-x19-y11)
 	(connected loc-x18-y11 loc-x18-y10)
 	(connected loc-x18-y11 loc-x18-y12)
 	(connected loc-x18-y12 loc-x17-y12)
 	(connected loc-x18-y12 loc-x19-y12)
 	(connected loc-x18-y12 loc-x18-y11)
 	(connected loc-x18-y12 loc-x18-y13)
 	(connected loc-x18-y13 loc-x17-y13)
 	(connected loc-x18-y13 loc-x19-y13)
 	(connected loc-x18-y13 loc-x18-y12)
 	(connected loc-x18-y13 loc-x18-y14)
 	(connected loc-x18-y14 loc-x17-y14)
 	(connected loc-x18-y14 loc-x19-y14)
 	(connected loc-x18-y14 loc-x18-y13)
 	(connected loc-x18-y14 loc-x18-y15)
 	(connected loc-x18-y15 loc-x17-y15)
 	(connected loc-x18-y15 loc-x19-y15)
 	(connected loc-x18-y15 loc-x18-y14)
 	(connected loc-x18-y15 loc-x18-y16)
 	(connected loc-x18-y16 loc-x17-y16)
 	(connected loc-x18-y16 loc-x19-y16)
 	(connected loc-x18-y16 loc-x18-y15)
 	(connected loc-x18-y16 loc-x18-y17)
 	(connected loc-x18-y17 loc-x17-y17)
 	(connected loc-x18-y17 loc-x19-y17)
 	(connected loc-x18-y17 loc-x18-y16)
 	(connected loc-x18-y17 loc-x18-y18)
 	(connected loc-x18-y18 loc-x17-y18)
 	(connected loc-x18-y18 loc-x19-y18)
 	(connected loc-x18-y18 loc-x18-y17)
 	(connected loc-x18-y18 loc-x18-y19)
 	(connected loc-x18-y19 loc-x17-y19)
 	(connected loc-x18-y19 loc-x19-y19)
 	(connected loc-x18-y19 loc-x18-y18)
 	(connected loc-x18-y19 loc-x18-y20)
 	(connected loc-x18-y20 loc-x17-y20)
 	(connected loc-x18-y20 loc-x19-y20)
 	(connected loc-x18-y20 loc-x18-y19)
 	(connected loc-x18-y20 loc-x18-y21)
 	(connected loc-x18-y21 loc-x17-y21)
 	(connected loc-x18-y21 loc-x19-y21)
 	(connected loc-x18-y21 loc-x18-y20)
 	(connected loc-x18-y21 loc-x18-y22)
 	(connected loc-x18-y22 loc-x17-y22)
 	(connected loc-x18-y22 loc-x19-y22)
 	(connected loc-x18-y22 loc-x18-y21)
 	(connected loc-x18-y22 loc-x18-y23)
 	(connected loc-x18-y23 loc-x17-y23)
 	(connected loc-x18-y23 loc-x19-y23)
 	(connected loc-x18-y23 loc-x18-y22)
 	(connected loc-x18-y23 loc-x18-y24)
 	(connected loc-x18-y24 loc-x17-y24)
 	(connected loc-x18-y24 loc-x19-y24)
 	(connected loc-x18-y24 loc-x18-y23)
 	(connected loc-x18-y24 loc-x18-y25)
 	(connected loc-x18-y25 loc-x17-y25)
 	(connected loc-x18-y25 loc-x19-y25)
 	(connected loc-x18-y25 loc-x18-y24)
 	(connected loc-x18-y25 loc-x18-y26)
 	(connected loc-x18-y26 loc-x17-y26)
 	(connected loc-x18-y26 loc-x19-y26)
 	(connected loc-x18-y26 loc-x18-y25)
 	(connected loc-x18-y26 loc-x18-y27)
 	(connected loc-x18-y27 loc-x17-y27)
 	(connected loc-x18-y27 loc-x19-y27)
 	(connected loc-x18-y27 loc-x18-y26)
 	(connected loc-x18-y27 loc-x18-y28)
 	(connected loc-x18-y28 loc-x17-y28)
 	(connected loc-x18-y28 loc-x19-y28)
 	(connected loc-x18-y28 loc-x18-y27)
 	(connected loc-x18-y28 loc-x18-y29)
 	(connected loc-x18-y29 loc-x17-y29)
 	(connected loc-x18-y29 loc-x19-y29)
 	(connected loc-x18-y29 loc-x18-y28)
 	(connected loc-x18-y29 loc-x18-y30)
 	(connected loc-x18-y30 loc-x17-y30)
 	(connected loc-x18-y30 loc-x19-y30)
 	(connected loc-x18-y30 loc-x18-y29)
 	(connected loc-x19-y0 loc-x18-y0)
 	(connected loc-x19-y0 loc-x20-y0)
 	(connected loc-x19-y0 loc-x19-y1)
 	(connected loc-x19-y1 loc-x18-y1)
 	(connected loc-x19-y1 loc-x20-y1)
 	(connected loc-x19-y1 loc-x19-y0)
 	(connected loc-x19-y1 loc-x19-y2)
 	(connected loc-x19-y2 loc-x18-y2)
 	(connected loc-x19-y2 loc-x20-y2)
 	(connected loc-x19-y2 loc-x19-y1)
 	(connected loc-x19-y2 loc-x19-y3)
 	(connected loc-x19-y3 loc-x18-y3)
 	(connected loc-x19-y3 loc-x20-y3)
 	(connected loc-x19-y3 loc-x19-y2)
 	(connected loc-x19-y3 loc-x19-y4)
 	(connected loc-x19-y4 loc-x18-y4)
 	(connected loc-x19-y4 loc-x20-y4)
 	(connected loc-x19-y4 loc-x19-y3)
 	(connected loc-x19-y4 loc-x19-y5)
 	(connected loc-x19-y5 loc-x18-y5)
 	(connected loc-x19-y5 loc-x20-y5)
 	(connected loc-x19-y5 loc-x19-y4)
 	(connected loc-x19-y5 loc-x19-y6)
 	(connected loc-x19-y6 loc-x18-y6)
 	(connected loc-x19-y6 loc-x20-y6)
 	(connected loc-x19-y6 loc-x19-y5)
 	(connected loc-x19-y6 loc-x19-y7)
 	(connected loc-x19-y7 loc-x18-y7)
 	(connected loc-x19-y7 loc-x20-y7)
 	(connected loc-x19-y7 loc-x19-y6)
 	(connected loc-x19-y7 loc-x19-y8)
 	(connected loc-x19-y8 loc-x18-y8)
 	(connected loc-x19-y8 loc-x20-y8)
 	(connected loc-x19-y8 loc-x19-y7)
 	(connected loc-x19-y8 loc-x19-y9)
 	(connected loc-x19-y9 loc-x18-y9)
 	(connected loc-x19-y9 loc-x20-y9)
 	(connected loc-x19-y9 loc-x19-y8)
 	(connected loc-x19-y9 loc-x19-y10)
 	(connected loc-x19-y10 loc-x18-y10)
 	(connected loc-x19-y10 loc-x20-y10)
 	(connected loc-x19-y10 loc-x19-y9)
 	(connected loc-x19-y10 loc-x19-y11)
 	(connected loc-x19-y11 loc-x18-y11)
 	(connected loc-x19-y11 loc-x20-y11)
 	(connected loc-x19-y11 loc-x19-y10)
 	(connected loc-x19-y11 loc-x19-y12)
 	(connected loc-x19-y12 loc-x18-y12)
 	(connected loc-x19-y12 loc-x20-y12)
 	(connected loc-x19-y12 loc-x19-y11)
 	(connected loc-x19-y12 loc-x19-y13)
 	(connected loc-x19-y13 loc-x18-y13)
 	(connected loc-x19-y13 loc-x20-y13)
 	(connected loc-x19-y13 loc-x19-y12)
 	(connected loc-x19-y13 loc-x19-y14)
 	(connected loc-x19-y14 loc-x18-y14)
 	(connected loc-x19-y14 loc-x20-y14)
 	(connected loc-x19-y14 loc-x19-y13)
 	(connected loc-x19-y14 loc-x19-y15)
 	(connected loc-x19-y15 loc-x18-y15)
 	(connected loc-x19-y15 loc-x20-y15)
 	(connected loc-x19-y15 loc-x19-y14)
 	(connected loc-x19-y15 loc-x19-y16)
 	(connected loc-x19-y16 loc-x18-y16)
 	(connected loc-x19-y16 loc-x20-y16)
 	(connected loc-x19-y16 loc-x19-y15)
 	(connected loc-x19-y16 loc-x19-y17)
 	(connected loc-x19-y17 loc-x18-y17)
 	(connected loc-x19-y17 loc-x20-y17)
 	(connected loc-x19-y17 loc-x19-y16)
 	(connected loc-x19-y17 loc-x19-y18)
 	(connected loc-x19-y18 loc-x18-y18)
 	(connected loc-x19-y18 loc-x20-y18)
 	(connected loc-x19-y18 loc-x19-y17)
 	(connected loc-x19-y18 loc-x19-y19)
 	(connected loc-x19-y19 loc-x18-y19)
 	(connected loc-x19-y19 loc-x20-y19)
 	(connected loc-x19-y19 loc-x19-y18)
 	(connected loc-x19-y19 loc-x19-y20)
 	(connected loc-x19-y20 loc-x18-y20)
 	(connected loc-x19-y20 loc-x20-y20)
 	(connected loc-x19-y20 loc-x19-y19)
 	(connected loc-x19-y20 loc-x19-y21)
 	(connected loc-x19-y21 loc-x18-y21)
 	(connected loc-x19-y21 loc-x20-y21)
 	(connected loc-x19-y21 loc-x19-y20)
 	(connected loc-x19-y21 loc-x19-y22)
 	(connected loc-x19-y22 loc-x18-y22)
 	(connected loc-x19-y22 loc-x20-y22)
 	(connected loc-x19-y22 loc-x19-y21)
 	(connected loc-x19-y22 loc-x19-y23)
 	(connected loc-x19-y23 loc-x18-y23)
 	(connected loc-x19-y23 loc-x20-y23)
 	(connected loc-x19-y23 loc-x19-y22)
 	(connected loc-x19-y23 loc-x19-y24)
 	(connected loc-x19-y24 loc-x18-y24)
 	(connected loc-x19-y24 loc-x20-y24)
 	(connected loc-x19-y24 loc-x19-y23)
 	(connected loc-x19-y24 loc-x19-y25)
 	(connected loc-x19-y25 loc-x18-y25)
 	(connected loc-x19-y25 loc-x20-y25)
 	(connected loc-x19-y25 loc-x19-y24)
 	(connected loc-x19-y25 loc-x19-y26)
 	(connected loc-x19-y26 loc-x18-y26)
 	(connected loc-x19-y26 loc-x20-y26)
 	(connected loc-x19-y26 loc-x19-y25)
 	(connected loc-x19-y26 loc-x19-y27)
 	(connected loc-x19-y27 loc-x18-y27)
 	(connected loc-x19-y27 loc-x20-y27)
 	(connected loc-x19-y27 loc-x19-y26)
 	(connected loc-x19-y27 loc-x19-y28)
 	(connected loc-x19-y28 loc-x18-y28)
 	(connected loc-x19-y28 loc-x20-y28)
 	(connected loc-x19-y28 loc-x19-y27)
 	(connected loc-x19-y28 loc-x19-y29)
 	(connected loc-x19-y29 loc-x18-y29)
 	(connected loc-x19-y29 loc-x20-y29)
 	(connected loc-x19-y29 loc-x19-y28)
 	(connected loc-x19-y29 loc-x19-y30)
 	(connected loc-x19-y30 loc-x18-y30)
 	(connected loc-x19-y30 loc-x20-y30)
 	(connected loc-x19-y30 loc-x19-y29)
 	(connected loc-x20-y0 loc-x19-y0)
 	(connected loc-x20-y0 loc-x21-y0)
 	(connected loc-x20-y0 loc-x20-y1)
 	(connected loc-x20-y1 loc-x19-y1)
 	(connected loc-x20-y1 loc-x21-y1)
 	(connected loc-x20-y1 loc-x20-y0)
 	(connected loc-x20-y1 loc-x20-y2)
 	(connected loc-x20-y2 loc-x19-y2)
 	(connected loc-x20-y2 loc-x21-y2)
 	(connected loc-x20-y2 loc-x20-y1)
 	(connected loc-x20-y2 loc-x20-y3)
 	(connected loc-x20-y3 loc-x19-y3)
 	(connected loc-x20-y3 loc-x21-y3)
 	(connected loc-x20-y3 loc-x20-y2)
 	(connected loc-x20-y3 loc-x20-y4)
 	(connected loc-x20-y4 loc-x19-y4)
 	(connected loc-x20-y4 loc-x21-y4)
 	(connected loc-x20-y4 loc-x20-y3)
 	(connected loc-x20-y4 loc-x20-y5)
 	(connected loc-x20-y5 loc-x19-y5)
 	(connected loc-x20-y5 loc-x21-y5)
 	(connected loc-x20-y5 loc-x20-y4)
 	(connected loc-x20-y5 loc-x20-y6)
 	(connected loc-x20-y6 loc-x19-y6)
 	(connected loc-x20-y6 loc-x21-y6)
 	(connected loc-x20-y6 loc-x20-y5)
 	(connected loc-x20-y6 loc-x20-y7)
 	(connected loc-x20-y7 loc-x19-y7)
 	(connected loc-x20-y7 loc-x21-y7)
 	(connected loc-x20-y7 loc-x20-y6)
 	(connected loc-x20-y7 loc-x20-y8)
 	(connected loc-x20-y8 loc-x19-y8)
 	(connected loc-x20-y8 loc-x21-y8)
 	(connected loc-x20-y8 loc-x20-y7)
 	(connected loc-x20-y8 loc-x20-y9)
 	(connected loc-x20-y9 loc-x19-y9)
 	(connected loc-x20-y9 loc-x21-y9)
 	(connected loc-x20-y9 loc-x20-y8)
 	(connected loc-x20-y9 loc-x20-y10)
 	(connected loc-x20-y10 loc-x19-y10)
 	(connected loc-x20-y10 loc-x21-y10)
 	(connected loc-x20-y10 loc-x20-y9)
 	(connected loc-x20-y10 loc-x20-y11)
 	(connected loc-x20-y11 loc-x19-y11)
 	(connected loc-x20-y11 loc-x21-y11)
 	(connected loc-x20-y11 loc-x20-y10)
 	(connected loc-x20-y11 loc-x20-y12)
 	(connected loc-x20-y12 loc-x19-y12)
 	(connected loc-x20-y12 loc-x21-y12)
 	(connected loc-x20-y12 loc-x20-y11)
 	(connected loc-x20-y12 loc-x20-y13)
 	(connected loc-x20-y13 loc-x19-y13)
 	(connected loc-x20-y13 loc-x21-y13)
 	(connected loc-x20-y13 loc-x20-y12)
 	(connected loc-x20-y13 loc-x20-y14)
 	(connected loc-x20-y14 loc-x19-y14)
 	(connected loc-x20-y14 loc-x21-y14)
 	(connected loc-x20-y14 loc-x20-y13)
 	(connected loc-x20-y14 loc-x20-y15)
 	(connected loc-x20-y15 loc-x19-y15)
 	(connected loc-x20-y15 loc-x21-y15)
 	(connected loc-x20-y15 loc-x20-y14)
 	(connected loc-x20-y15 loc-x20-y16)
 	(connected loc-x20-y16 loc-x19-y16)
 	(connected loc-x20-y16 loc-x21-y16)
 	(connected loc-x20-y16 loc-x20-y15)
 	(connected loc-x20-y16 loc-x20-y17)
 	(connected loc-x20-y17 loc-x19-y17)
 	(connected loc-x20-y17 loc-x21-y17)
 	(connected loc-x20-y17 loc-x20-y16)
 	(connected loc-x20-y17 loc-x20-y18)
 	(connected loc-x20-y18 loc-x19-y18)
 	(connected loc-x20-y18 loc-x21-y18)
 	(connected loc-x20-y18 loc-x20-y17)
 	(connected loc-x20-y18 loc-x20-y19)
 	(connected loc-x20-y19 loc-x19-y19)
 	(connected loc-x20-y19 loc-x21-y19)
 	(connected loc-x20-y19 loc-x20-y18)
 	(connected loc-x20-y19 loc-x20-y20)
 	(connected loc-x20-y20 loc-x19-y20)
 	(connected loc-x20-y20 loc-x21-y20)
 	(connected loc-x20-y20 loc-x20-y19)
 	(connected loc-x20-y20 loc-x20-y21)
 	(connected loc-x20-y21 loc-x19-y21)
 	(connected loc-x20-y21 loc-x21-y21)
 	(connected loc-x20-y21 loc-x20-y20)
 	(connected loc-x20-y21 loc-x20-y22)
 	(connected loc-x20-y22 loc-x19-y22)
 	(connected loc-x20-y22 loc-x21-y22)
 	(connected loc-x20-y22 loc-x20-y21)
 	(connected loc-x20-y22 loc-x20-y23)
 	(connected loc-x20-y23 loc-x19-y23)
 	(connected loc-x20-y23 loc-x21-y23)
 	(connected loc-x20-y23 loc-x20-y22)
 	(connected loc-x20-y23 loc-x20-y24)
 	(connected loc-x20-y24 loc-x19-y24)
 	(connected loc-x20-y24 loc-x21-y24)
 	(connected loc-x20-y24 loc-x20-y23)
 	(connected loc-x20-y24 loc-x20-y25)
 	(connected loc-x20-y25 loc-x19-y25)
 	(connected loc-x20-y25 loc-x21-y25)
 	(connected loc-x20-y25 loc-x20-y24)
 	(connected loc-x20-y25 loc-x20-y26)
 	(connected loc-x20-y26 loc-x19-y26)
 	(connected loc-x20-y26 loc-x21-y26)
 	(connected loc-x20-y26 loc-x20-y25)
 	(connected loc-x20-y26 loc-x20-y27)
 	(connected loc-x20-y27 loc-x19-y27)
 	(connected loc-x20-y27 loc-x21-y27)
 	(connected loc-x20-y27 loc-x20-y26)
 	(connected loc-x20-y27 loc-x20-y28)
 	(connected loc-x20-y28 loc-x19-y28)
 	(connected loc-x20-y28 loc-x21-y28)
 	(connected loc-x20-y28 loc-x20-y27)
 	(connected loc-x20-y28 loc-x20-y29)
 	(connected loc-x20-y29 loc-x19-y29)
 	(connected loc-x20-y29 loc-x21-y29)
 	(connected loc-x20-y29 loc-x20-y28)
 	(connected loc-x20-y29 loc-x20-y30)
 	(connected loc-x20-y30 loc-x19-y30)
 	(connected loc-x20-y30 loc-x21-y30)
 	(connected loc-x20-y30 loc-x20-y29)
 	(connected loc-x21-y0 loc-x20-y0)
 	(connected loc-x21-y0 loc-x22-y0)
 	(connected loc-x21-y0 loc-x21-y1)
 	(connected loc-x21-y1 loc-x20-y1)
 	(connected loc-x21-y1 loc-x22-y1)
 	(connected loc-x21-y1 loc-x21-y0)
 	(connected loc-x21-y1 loc-x21-y2)
 	(connected loc-x21-y2 loc-x20-y2)
 	(connected loc-x21-y2 loc-x22-y2)
 	(connected loc-x21-y2 loc-x21-y1)
 	(connected loc-x21-y2 loc-x21-y3)
 	(connected loc-x21-y3 loc-x20-y3)
 	(connected loc-x21-y3 loc-x22-y3)
 	(connected loc-x21-y3 loc-x21-y2)
 	(connected loc-x21-y3 loc-x21-y4)
 	(connected loc-x21-y4 loc-x20-y4)
 	(connected loc-x21-y4 loc-x22-y4)
 	(connected loc-x21-y4 loc-x21-y3)
 	(connected loc-x21-y4 loc-x21-y5)
 	(connected loc-x21-y5 loc-x20-y5)
 	(connected loc-x21-y5 loc-x22-y5)
 	(connected loc-x21-y5 loc-x21-y4)
 	(connected loc-x21-y5 loc-x21-y6)
 	(connected loc-x21-y6 loc-x20-y6)
 	(connected loc-x21-y6 loc-x22-y6)
 	(connected loc-x21-y6 loc-x21-y5)
 	(connected loc-x21-y6 loc-x21-y7)
 	(connected loc-x21-y7 loc-x20-y7)
 	(connected loc-x21-y7 loc-x22-y7)
 	(connected loc-x21-y7 loc-x21-y6)
 	(connected loc-x21-y7 loc-x21-y8)
 	(connected loc-x21-y8 loc-x20-y8)
 	(connected loc-x21-y8 loc-x22-y8)
 	(connected loc-x21-y8 loc-x21-y7)
 	(connected loc-x21-y8 loc-x21-y9)
 	(connected loc-x21-y9 loc-x20-y9)
 	(connected loc-x21-y9 loc-x22-y9)
 	(connected loc-x21-y9 loc-x21-y8)
 	(connected loc-x21-y9 loc-x21-y10)
 	(connected loc-x21-y10 loc-x20-y10)
 	(connected loc-x21-y10 loc-x22-y10)
 	(connected loc-x21-y10 loc-x21-y9)
 	(connected loc-x21-y10 loc-x21-y11)
 	(connected loc-x21-y11 loc-x20-y11)
 	(connected loc-x21-y11 loc-x22-y11)
 	(connected loc-x21-y11 loc-x21-y10)
 	(connected loc-x21-y11 loc-x21-y12)
 	(connected loc-x21-y12 loc-x20-y12)
 	(connected loc-x21-y12 loc-x22-y12)
 	(connected loc-x21-y12 loc-x21-y11)
 	(connected loc-x21-y12 loc-x21-y13)
 	(connected loc-x21-y13 loc-x20-y13)
 	(connected loc-x21-y13 loc-x22-y13)
 	(connected loc-x21-y13 loc-x21-y12)
 	(connected loc-x21-y13 loc-x21-y14)
 	(connected loc-x21-y14 loc-x20-y14)
 	(connected loc-x21-y14 loc-x22-y14)
 	(connected loc-x21-y14 loc-x21-y13)
 	(connected loc-x21-y14 loc-x21-y15)
 	(connected loc-x21-y15 loc-x20-y15)
 	(connected loc-x21-y15 loc-x22-y15)
 	(connected loc-x21-y15 loc-x21-y14)
 	(connected loc-x21-y15 loc-x21-y16)
 	(connected loc-x21-y16 loc-x20-y16)
 	(connected loc-x21-y16 loc-x22-y16)
 	(connected loc-x21-y16 loc-x21-y15)
 	(connected loc-x21-y16 loc-x21-y17)
 	(connected loc-x21-y17 loc-x20-y17)
 	(connected loc-x21-y17 loc-x22-y17)
 	(connected loc-x21-y17 loc-x21-y16)
 	(connected loc-x21-y17 loc-x21-y18)
 	(connected loc-x21-y18 loc-x20-y18)
 	(connected loc-x21-y18 loc-x22-y18)
 	(connected loc-x21-y18 loc-x21-y17)
 	(connected loc-x21-y18 loc-x21-y19)
 	(connected loc-x21-y19 loc-x20-y19)
 	(connected loc-x21-y19 loc-x22-y19)
 	(connected loc-x21-y19 loc-x21-y18)
 	(connected loc-x21-y19 loc-x21-y20)
 	(connected loc-x21-y20 loc-x20-y20)
 	(connected loc-x21-y20 loc-x22-y20)
 	(connected loc-x21-y20 loc-x21-y19)
 	(connected loc-x21-y20 loc-x21-y21)
 	(connected loc-x21-y21 loc-x20-y21)
 	(connected loc-x21-y21 loc-x22-y21)
 	(connected loc-x21-y21 loc-x21-y20)
 	(connected loc-x21-y21 loc-x21-y22)
 	(connected loc-x21-y22 loc-x20-y22)
 	(connected loc-x21-y22 loc-x22-y22)
 	(connected loc-x21-y22 loc-x21-y21)
 	(connected loc-x21-y22 loc-x21-y23)
 	(connected loc-x21-y23 loc-x20-y23)
 	(connected loc-x21-y23 loc-x22-y23)
 	(connected loc-x21-y23 loc-x21-y22)
 	(connected loc-x21-y23 loc-x21-y24)
 	(connected loc-x21-y24 loc-x20-y24)
 	(connected loc-x21-y24 loc-x22-y24)
 	(connected loc-x21-y24 loc-x21-y23)
 	(connected loc-x21-y24 loc-x21-y25)
 	(connected loc-x21-y25 loc-x20-y25)
 	(connected loc-x21-y25 loc-x22-y25)
 	(connected loc-x21-y25 loc-x21-y24)
 	(connected loc-x21-y25 loc-x21-y26)
 	(connected loc-x21-y26 loc-x20-y26)
 	(connected loc-x21-y26 loc-x22-y26)
 	(connected loc-x21-y26 loc-x21-y25)
 	(connected loc-x21-y26 loc-x21-y27)
 	(connected loc-x21-y27 loc-x20-y27)
 	(connected loc-x21-y27 loc-x22-y27)
 	(connected loc-x21-y27 loc-x21-y26)
 	(connected loc-x21-y27 loc-x21-y28)
 	(connected loc-x21-y28 loc-x20-y28)
 	(connected loc-x21-y28 loc-x22-y28)
 	(connected loc-x21-y28 loc-x21-y27)
 	(connected loc-x21-y28 loc-x21-y29)
 	(connected loc-x21-y29 loc-x20-y29)
 	(connected loc-x21-y29 loc-x22-y29)
 	(connected loc-x21-y29 loc-x21-y28)
 	(connected loc-x21-y29 loc-x21-y30)
 	(connected loc-x21-y30 loc-x20-y30)
 	(connected loc-x21-y30 loc-x22-y30)
 	(connected loc-x21-y30 loc-x21-y29)
 	(connected loc-x22-y0 loc-x21-y0)
 	(connected loc-x22-y0 loc-x23-y0)
 	(connected loc-x22-y0 loc-x22-y1)
 	(connected loc-x22-y1 loc-x21-y1)
 	(connected loc-x22-y1 loc-x23-y1)
 	(connected loc-x22-y1 loc-x22-y0)
 	(connected loc-x22-y1 loc-x22-y2)
 	(connected loc-x22-y2 loc-x21-y2)
 	(connected loc-x22-y2 loc-x23-y2)
 	(connected loc-x22-y2 loc-x22-y1)
 	(connected loc-x22-y2 loc-x22-y3)
 	(connected loc-x22-y3 loc-x21-y3)
 	(connected loc-x22-y3 loc-x23-y3)
 	(connected loc-x22-y3 loc-x22-y2)
 	(connected loc-x22-y3 loc-x22-y4)
 	(connected loc-x22-y4 loc-x21-y4)
 	(connected loc-x22-y4 loc-x23-y4)
 	(connected loc-x22-y4 loc-x22-y3)
 	(connected loc-x22-y4 loc-x22-y5)
 	(connected loc-x22-y5 loc-x21-y5)
 	(connected loc-x22-y5 loc-x23-y5)
 	(connected loc-x22-y5 loc-x22-y4)
 	(connected loc-x22-y5 loc-x22-y6)
 	(connected loc-x22-y6 loc-x21-y6)
 	(connected loc-x22-y6 loc-x23-y6)
 	(connected loc-x22-y6 loc-x22-y5)
 	(connected loc-x22-y6 loc-x22-y7)
 	(connected loc-x22-y7 loc-x21-y7)
 	(connected loc-x22-y7 loc-x23-y7)
 	(connected loc-x22-y7 loc-x22-y6)
 	(connected loc-x22-y7 loc-x22-y8)
 	(connected loc-x22-y8 loc-x21-y8)
 	(connected loc-x22-y8 loc-x23-y8)
 	(connected loc-x22-y8 loc-x22-y7)
 	(connected loc-x22-y8 loc-x22-y9)
 	(connected loc-x22-y9 loc-x21-y9)
 	(connected loc-x22-y9 loc-x23-y9)
 	(connected loc-x22-y9 loc-x22-y8)
 	(connected loc-x22-y9 loc-x22-y10)
 	(connected loc-x22-y10 loc-x21-y10)
 	(connected loc-x22-y10 loc-x23-y10)
 	(connected loc-x22-y10 loc-x22-y9)
 	(connected loc-x22-y10 loc-x22-y11)
 	(connected loc-x22-y11 loc-x21-y11)
 	(connected loc-x22-y11 loc-x23-y11)
 	(connected loc-x22-y11 loc-x22-y10)
 	(connected loc-x22-y11 loc-x22-y12)
 	(connected loc-x22-y12 loc-x21-y12)
 	(connected loc-x22-y12 loc-x23-y12)
 	(connected loc-x22-y12 loc-x22-y11)
 	(connected loc-x22-y12 loc-x22-y13)
 	(connected loc-x22-y13 loc-x21-y13)
 	(connected loc-x22-y13 loc-x23-y13)
 	(connected loc-x22-y13 loc-x22-y12)
 	(connected loc-x22-y13 loc-x22-y14)
 	(connected loc-x22-y14 loc-x21-y14)
 	(connected loc-x22-y14 loc-x23-y14)
 	(connected loc-x22-y14 loc-x22-y13)
 	(connected loc-x22-y14 loc-x22-y15)
 	(connected loc-x22-y15 loc-x21-y15)
 	(connected loc-x22-y15 loc-x23-y15)
 	(connected loc-x22-y15 loc-x22-y14)
 	(connected loc-x22-y15 loc-x22-y16)
 	(connected loc-x22-y16 loc-x21-y16)
 	(connected loc-x22-y16 loc-x23-y16)
 	(connected loc-x22-y16 loc-x22-y15)
 	(connected loc-x22-y16 loc-x22-y17)
 	(connected loc-x22-y17 loc-x21-y17)
 	(connected loc-x22-y17 loc-x23-y17)
 	(connected loc-x22-y17 loc-x22-y16)
 	(connected loc-x22-y17 loc-x22-y18)
 	(connected loc-x22-y18 loc-x21-y18)
 	(connected loc-x22-y18 loc-x23-y18)
 	(connected loc-x22-y18 loc-x22-y17)
 	(connected loc-x22-y18 loc-x22-y19)
 	(connected loc-x22-y19 loc-x21-y19)
 	(connected loc-x22-y19 loc-x23-y19)
 	(connected loc-x22-y19 loc-x22-y18)
 	(connected loc-x22-y19 loc-x22-y20)
 	(connected loc-x22-y20 loc-x21-y20)
 	(connected loc-x22-y20 loc-x23-y20)
 	(connected loc-x22-y20 loc-x22-y19)
 	(connected loc-x22-y20 loc-x22-y21)
 	(connected loc-x22-y21 loc-x21-y21)
 	(connected loc-x22-y21 loc-x23-y21)
 	(connected loc-x22-y21 loc-x22-y20)
 	(connected loc-x22-y21 loc-x22-y22)
 	(connected loc-x22-y22 loc-x21-y22)
 	(connected loc-x22-y22 loc-x23-y22)
 	(connected loc-x22-y22 loc-x22-y21)
 	(connected loc-x22-y22 loc-x22-y23)
 	(connected loc-x22-y23 loc-x21-y23)
 	(connected loc-x22-y23 loc-x23-y23)
 	(connected loc-x22-y23 loc-x22-y22)
 	(connected loc-x22-y23 loc-x22-y24)
 	(connected loc-x22-y24 loc-x21-y24)
 	(connected loc-x22-y24 loc-x23-y24)
 	(connected loc-x22-y24 loc-x22-y23)
 	(connected loc-x22-y24 loc-x22-y25)
 	(connected loc-x22-y25 loc-x21-y25)
 	(connected loc-x22-y25 loc-x23-y25)
 	(connected loc-x22-y25 loc-x22-y24)
 	(connected loc-x22-y25 loc-x22-y26)
 	(connected loc-x22-y26 loc-x21-y26)
 	(connected loc-x22-y26 loc-x23-y26)
 	(connected loc-x22-y26 loc-x22-y25)
 	(connected loc-x22-y26 loc-x22-y27)
 	(connected loc-x22-y27 loc-x21-y27)
 	(connected loc-x22-y27 loc-x23-y27)
 	(connected loc-x22-y27 loc-x22-y26)
 	(connected loc-x22-y27 loc-x22-y28)
 	(connected loc-x22-y28 loc-x21-y28)
 	(connected loc-x22-y28 loc-x23-y28)
 	(connected loc-x22-y28 loc-x22-y27)
 	(connected loc-x22-y28 loc-x22-y29)
 	(connected loc-x22-y29 loc-x21-y29)
 	(connected loc-x22-y29 loc-x23-y29)
 	(connected loc-x22-y29 loc-x22-y28)
 	(connected loc-x22-y29 loc-x22-y30)
 	(connected loc-x22-y30 loc-x21-y30)
 	(connected loc-x22-y30 loc-x23-y30)
 	(connected loc-x22-y30 loc-x22-y29)
 	(connected loc-x23-y0 loc-x22-y0)
 	(connected loc-x23-y0 loc-x24-y0)
 	(connected loc-x23-y0 loc-x23-y1)
 	(connected loc-x23-y1 loc-x22-y1)
 	(connected loc-x23-y1 loc-x24-y1)
 	(connected loc-x23-y1 loc-x23-y0)
 	(connected loc-x23-y1 loc-x23-y2)
 	(connected loc-x23-y2 loc-x22-y2)
 	(connected loc-x23-y2 loc-x24-y2)
 	(connected loc-x23-y2 loc-x23-y1)
 	(connected loc-x23-y2 loc-x23-y3)
 	(connected loc-x23-y3 loc-x22-y3)
 	(connected loc-x23-y3 loc-x24-y3)
 	(connected loc-x23-y3 loc-x23-y2)
 	(connected loc-x23-y3 loc-x23-y4)
 	(connected loc-x23-y4 loc-x22-y4)
 	(connected loc-x23-y4 loc-x24-y4)
 	(connected loc-x23-y4 loc-x23-y3)
 	(connected loc-x23-y4 loc-x23-y5)
 	(connected loc-x23-y5 loc-x22-y5)
 	(connected loc-x23-y5 loc-x24-y5)
 	(connected loc-x23-y5 loc-x23-y4)
 	(connected loc-x23-y5 loc-x23-y6)
 	(connected loc-x23-y6 loc-x22-y6)
 	(connected loc-x23-y6 loc-x24-y6)
 	(connected loc-x23-y6 loc-x23-y5)
 	(connected loc-x23-y6 loc-x23-y7)
 	(connected loc-x23-y7 loc-x22-y7)
 	(connected loc-x23-y7 loc-x24-y7)
 	(connected loc-x23-y7 loc-x23-y6)
 	(connected loc-x23-y7 loc-x23-y8)
 	(connected loc-x23-y8 loc-x22-y8)
 	(connected loc-x23-y8 loc-x24-y8)
 	(connected loc-x23-y8 loc-x23-y7)
 	(connected loc-x23-y8 loc-x23-y9)
 	(connected loc-x23-y9 loc-x22-y9)
 	(connected loc-x23-y9 loc-x24-y9)
 	(connected loc-x23-y9 loc-x23-y8)
 	(connected loc-x23-y9 loc-x23-y10)
 	(connected loc-x23-y10 loc-x22-y10)
 	(connected loc-x23-y10 loc-x24-y10)
 	(connected loc-x23-y10 loc-x23-y9)
 	(connected loc-x23-y10 loc-x23-y11)
 	(connected loc-x23-y11 loc-x22-y11)
 	(connected loc-x23-y11 loc-x24-y11)
 	(connected loc-x23-y11 loc-x23-y10)
 	(connected loc-x23-y11 loc-x23-y12)
 	(connected loc-x23-y12 loc-x22-y12)
 	(connected loc-x23-y12 loc-x24-y12)
 	(connected loc-x23-y12 loc-x23-y11)
 	(connected loc-x23-y12 loc-x23-y13)
 	(connected loc-x23-y13 loc-x22-y13)
 	(connected loc-x23-y13 loc-x24-y13)
 	(connected loc-x23-y13 loc-x23-y12)
 	(connected loc-x23-y13 loc-x23-y14)
 	(connected loc-x23-y14 loc-x22-y14)
 	(connected loc-x23-y14 loc-x24-y14)
 	(connected loc-x23-y14 loc-x23-y13)
 	(connected loc-x23-y14 loc-x23-y15)
 	(connected loc-x23-y15 loc-x22-y15)
 	(connected loc-x23-y15 loc-x24-y15)
 	(connected loc-x23-y15 loc-x23-y14)
 	(connected loc-x23-y15 loc-x23-y16)
 	(connected loc-x23-y16 loc-x22-y16)
 	(connected loc-x23-y16 loc-x24-y16)
 	(connected loc-x23-y16 loc-x23-y15)
 	(connected loc-x23-y16 loc-x23-y17)
 	(connected loc-x23-y17 loc-x22-y17)
 	(connected loc-x23-y17 loc-x24-y17)
 	(connected loc-x23-y17 loc-x23-y16)
 	(connected loc-x23-y17 loc-x23-y18)
 	(connected loc-x23-y18 loc-x22-y18)
 	(connected loc-x23-y18 loc-x24-y18)
 	(connected loc-x23-y18 loc-x23-y17)
 	(connected loc-x23-y18 loc-x23-y19)
 	(connected loc-x23-y19 loc-x22-y19)
 	(connected loc-x23-y19 loc-x24-y19)
 	(connected loc-x23-y19 loc-x23-y18)
 	(connected loc-x23-y19 loc-x23-y20)
 	(connected loc-x23-y20 loc-x22-y20)
 	(connected loc-x23-y20 loc-x24-y20)
 	(connected loc-x23-y20 loc-x23-y19)
 	(connected loc-x23-y20 loc-x23-y21)
 	(connected loc-x23-y21 loc-x22-y21)
 	(connected loc-x23-y21 loc-x24-y21)
 	(connected loc-x23-y21 loc-x23-y20)
 	(connected loc-x23-y21 loc-x23-y22)
 	(connected loc-x23-y22 loc-x22-y22)
 	(connected loc-x23-y22 loc-x24-y22)
 	(connected loc-x23-y22 loc-x23-y21)
 	(connected loc-x23-y22 loc-x23-y23)
 	(connected loc-x23-y23 loc-x22-y23)
 	(connected loc-x23-y23 loc-x24-y23)
 	(connected loc-x23-y23 loc-x23-y22)
 	(connected loc-x23-y23 loc-x23-y24)
 	(connected loc-x23-y24 loc-x22-y24)
 	(connected loc-x23-y24 loc-x24-y24)
 	(connected loc-x23-y24 loc-x23-y23)
 	(connected loc-x23-y24 loc-x23-y25)
 	(connected loc-x23-y25 loc-x22-y25)
 	(connected loc-x23-y25 loc-x24-y25)
 	(connected loc-x23-y25 loc-x23-y24)
 	(connected loc-x23-y25 loc-x23-y26)
 	(connected loc-x23-y26 loc-x22-y26)
 	(connected loc-x23-y26 loc-x24-y26)
 	(connected loc-x23-y26 loc-x23-y25)
 	(connected loc-x23-y26 loc-x23-y27)
 	(connected loc-x23-y27 loc-x22-y27)
 	(connected loc-x23-y27 loc-x24-y27)
 	(connected loc-x23-y27 loc-x23-y26)
 	(connected loc-x23-y27 loc-x23-y28)
 	(connected loc-x23-y28 loc-x22-y28)
 	(connected loc-x23-y28 loc-x24-y28)
 	(connected loc-x23-y28 loc-x23-y27)
 	(connected loc-x23-y28 loc-x23-y29)
 	(connected loc-x23-y29 loc-x22-y29)
 	(connected loc-x23-y29 loc-x24-y29)
 	(connected loc-x23-y29 loc-x23-y28)
 	(connected loc-x23-y29 loc-x23-y30)
 	(connected loc-x23-y30 loc-x22-y30)
 	(connected loc-x23-y30 loc-x24-y30)
 	(connected loc-x23-y30 loc-x23-y29)
 	(connected loc-x24-y0 loc-x23-y0)
 	(connected loc-x24-y0 loc-x25-y0)
 	(connected loc-x24-y0 loc-x24-y1)
 	(connected loc-x24-y1 loc-x23-y1)
 	(connected loc-x24-y1 loc-x25-y1)
 	(connected loc-x24-y1 loc-x24-y0)
 	(connected loc-x24-y1 loc-x24-y2)
 	(connected loc-x24-y2 loc-x23-y2)
 	(connected loc-x24-y2 loc-x25-y2)
 	(connected loc-x24-y2 loc-x24-y1)
 	(connected loc-x24-y2 loc-x24-y3)
 	(connected loc-x24-y3 loc-x23-y3)
 	(connected loc-x24-y3 loc-x25-y3)
 	(connected loc-x24-y3 loc-x24-y2)
 	(connected loc-x24-y3 loc-x24-y4)
 	(connected loc-x24-y4 loc-x23-y4)
 	(connected loc-x24-y4 loc-x25-y4)
 	(connected loc-x24-y4 loc-x24-y3)
 	(connected loc-x24-y4 loc-x24-y5)
 	(connected loc-x24-y5 loc-x23-y5)
 	(connected loc-x24-y5 loc-x25-y5)
 	(connected loc-x24-y5 loc-x24-y4)
 	(connected loc-x24-y5 loc-x24-y6)
 	(connected loc-x24-y6 loc-x23-y6)
 	(connected loc-x24-y6 loc-x25-y6)
 	(connected loc-x24-y6 loc-x24-y5)
 	(connected loc-x24-y6 loc-x24-y7)
 	(connected loc-x24-y7 loc-x23-y7)
 	(connected loc-x24-y7 loc-x25-y7)
 	(connected loc-x24-y7 loc-x24-y6)
 	(connected loc-x24-y7 loc-x24-y8)
 	(connected loc-x24-y8 loc-x23-y8)
 	(connected loc-x24-y8 loc-x25-y8)
 	(connected loc-x24-y8 loc-x24-y7)
 	(connected loc-x24-y8 loc-x24-y9)
 	(connected loc-x24-y9 loc-x23-y9)
 	(connected loc-x24-y9 loc-x25-y9)
 	(connected loc-x24-y9 loc-x24-y8)
 	(connected loc-x24-y9 loc-x24-y10)
 	(connected loc-x24-y10 loc-x23-y10)
 	(connected loc-x24-y10 loc-x25-y10)
 	(connected loc-x24-y10 loc-x24-y9)
 	(connected loc-x24-y10 loc-x24-y11)
 	(connected loc-x24-y11 loc-x23-y11)
 	(connected loc-x24-y11 loc-x25-y11)
 	(connected loc-x24-y11 loc-x24-y10)
 	(connected loc-x24-y11 loc-x24-y12)
 	(connected loc-x24-y12 loc-x23-y12)
 	(connected loc-x24-y12 loc-x25-y12)
 	(connected loc-x24-y12 loc-x24-y11)
 	(connected loc-x24-y12 loc-x24-y13)
 	(connected loc-x24-y13 loc-x23-y13)
 	(connected loc-x24-y13 loc-x25-y13)
 	(connected loc-x24-y13 loc-x24-y12)
 	(connected loc-x24-y13 loc-x24-y14)
 	(connected loc-x24-y14 loc-x23-y14)
 	(connected loc-x24-y14 loc-x25-y14)
 	(connected loc-x24-y14 loc-x24-y13)
 	(connected loc-x24-y14 loc-x24-y15)
 	(connected loc-x24-y15 loc-x23-y15)
 	(connected loc-x24-y15 loc-x25-y15)
 	(connected loc-x24-y15 loc-x24-y14)
 	(connected loc-x24-y15 loc-x24-y16)
 	(connected loc-x24-y16 loc-x23-y16)
 	(connected loc-x24-y16 loc-x25-y16)
 	(connected loc-x24-y16 loc-x24-y15)
 	(connected loc-x24-y16 loc-x24-y17)
 	(connected loc-x24-y17 loc-x23-y17)
 	(connected loc-x24-y17 loc-x25-y17)
 	(connected loc-x24-y17 loc-x24-y16)
 	(connected loc-x24-y17 loc-x24-y18)
 	(connected loc-x24-y18 loc-x23-y18)
 	(connected loc-x24-y18 loc-x25-y18)
 	(connected loc-x24-y18 loc-x24-y17)
 	(connected loc-x24-y18 loc-x24-y19)
 	(connected loc-x24-y19 loc-x23-y19)
 	(connected loc-x24-y19 loc-x25-y19)
 	(connected loc-x24-y19 loc-x24-y18)
 	(connected loc-x24-y19 loc-x24-y20)
 	(connected loc-x24-y20 loc-x23-y20)
 	(connected loc-x24-y20 loc-x25-y20)
 	(connected loc-x24-y20 loc-x24-y19)
 	(connected loc-x24-y20 loc-x24-y21)
 	(connected loc-x24-y21 loc-x23-y21)
 	(connected loc-x24-y21 loc-x25-y21)
 	(connected loc-x24-y21 loc-x24-y20)
 	(connected loc-x24-y21 loc-x24-y22)
 	(connected loc-x24-y22 loc-x23-y22)
 	(connected loc-x24-y22 loc-x25-y22)
 	(connected loc-x24-y22 loc-x24-y21)
 	(connected loc-x24-y22 loc-x24-y23)
 	(connected loc-x24-y23 loc-x23-y23)
 	(connected loc-x24-y23 loc-x25-y23)
 	(connected loc-x24-y23 loc-x24-y22)
 	(connected loc-x24-y23 loc-x24-y24)
 	(connected loc-x24-y24 loc-x23-y24)
 	(connected loc-x24-y24 loc-x25-y24)
 	(connected loc-x24-y24 loc-x24-y23)
 	(connected loc-x24-y24 loc-x24-y25)
 	(connected loc-x24-y25 loc-x23-y25)
 	(connected loc-x24-y25 loc-x25-y25)
 	(connected loc-x24-y25 loc-x24-y24)
 	(connected loc-x24-y25 loc-x24-y26)
 	(connected loc-x24-y26 loc-x23-y26)
 	(connected loc-x24-y26 loc-x25-y26)
 	(connected loc-x24-y26 loc-x24-y25)
 	(connected loc-x24-y26 loc-x24-y27)
 	(connected loc-x24-y27 loc-x23-y27)
 	(connected loc-x24-y27 loc-x25-y27)
 	(connected loc-x24-y27 loc-x24-y26)
 	(connected loc-x24-y27 loc-x24-y28)
 	(connected loc-x24-y28 loc-x23-y28)
 	(connected loc-x24-y28 loc-x25-y28)
 	(connected loc-x24-y28 loc-x24-y27)
 	(connected loc-x24-y28 loc-x24-y29)
 	(connected loc-x24-y29 loc-x23-y29)
 	(connected loc-x24-y29 loc-x25-y29)
 	(connected loc-x24-y29 loc-x24-y28)
 	(connected loc-x24-y29 loc-x24-y30)
 	(connected loc-x24-y30 loc-x23-y30)
 	(connected loc-x24-y30 loc-x25-y30)
 	(connected loc-x24-y30 loc-x24-y29)
 	(connected loc-x25-y0 loc-x24-y0)
 	(connected loc-x25-y0 loc-x26-y0)
 	(connected loc-x25-y0 loc-x25-y1)
 	(connected loc-x25-y1 loc-x24-y1)
 	(connected loc-x25-y1 loc-x26-y1)
 	(connected loc-x25-y1 loc-x25-y0)
 	(connected loc-x25-y1 loc-x25-y2)
 	(connected loc-x25-y2 loc-x24-y2)
 	(connected loc-x25-y2 loc-x26-y2)
 	(connected loc-x25-y2 loc-x25-y1)
 	(connected loc-x25-y2 loc-x25-y3)
 	(connected loc-x25-y3 loc-x24-y3)
 	(connected loc-x25-y3 loc-x26-y3)
 	(connected loc-x25-y3 loc-x25-y2)
 	(connected loc-x25-y3 loc-x25-y4)
 	(connected loc-x25-y4 loc-x24-y4)
 	(connected loc-x25-y4 loc-x26-y4)
 	(connected loc-x25-y4 loc-x25-y3)
 	(connected loc-x25-y4 loc-x25-y5)
 	(connected loc-x25-y5 loc-x24-y5)
 	(connected loc-x25-y5 loc-x26-y5)
 	(connected loc-x25-y5 loc-x25-y4)
 	(connected loc-x25-y5 loc-x25-y6)
 	(connected loc-x25-y6 loc-x24-y6)
 	(connected loc-x25-y6 loc-x26-y6)
 	(connected loc-x25-y6 loc-x25-y5)
 	(connected loc-x25-y6 loc-x25-y7)
 	(connected loc-x25-y7 loc-x24-y7)
 	(connected loc-x25-y7 loc-x26-y7)
 	(connected loc-x25-y7 loc-x25-y6)
 	(connected loc-x25-y7 loc-x25-y8)
 	(connected loc-x25-y8 loc-x24-y8)
 	(connected loc-x25-y8 loc-x26-y8)
 	(connected loc-x25-y8 loc-x25-y7)
 	(connected loc-x25-y8 loc-x25-y9)
 	(connected loc-x25-y9 loc-x24-y9)
 	(connected loc-x25-y9 loc-x26-y9)
 	(connected loc-x25-y9 loc-x25-y8)
 	(connected loc-x25-y9 loc-x25-y10)
 	(connected loc-x25-y10 loc-x24-y10)
 	(connected loc-x25-y10 loc-x26-y10)
 	(connected loc-x25-y10 loc-x25-y9)
 	(connected loc-x25-y10 loc-x25-y11)
 	(connected loc-x25-y11 loc-x24-y11)
 	(connected loc-x25-y11 loc-x26-y11)
 	(connected loc-x25-y11 loc-x25-y10)
 	(connected loc-x25-y11 loc-x25-y12)
 	(connected loc-x25-y12 loc-x24-y12)
 	(connected loc-x25-y12 loc-x26-y12)
 	(connected loc-x25-y12 loc-x25-y11)
 	(connected loc-x25-y12 loc-x25-y13)
 	(connected loc-x25-y13 loc-x24-y13)
 	(connected loc-x25-y13 loc-x26-y13)
 	(connected loc-x25-y13 loc-x25-y12)
 	(connected loc-x25-y13 loc-x25-y14)
 	(connected loc-x25-y14 loc-x24-y14)
 	(connected loc-x25-y14 loc-x26-y14)
 	(connected loc-x25-y14 loc-x25-y13)
 	(connected loc-x25-y14 loc-x25-y15)
 	(connected loc-x25-y15 loc-x24-y15)
 	(connected loc-x25-y15 loc-x26-y15)
 	(connected loc-x25-y15 loc-x25-y14)
 	(connected loc-x25-y15 loc-x25-y16)
 	(connected loc-x25-y16 loc-x24-y16)
 	(connected loc-x25-y16 loc-x26-y16)
 	(connected loc-x25-y16 loc-x25-y15)
 	(connected loc-x25-y16 loc-x25-y17)
 	(connected loc-x25-y17 loc-x24-y17)
 	(connected loc-x25-y17 loc-x26-y17)
 	(connected loc-x25-y17 loc-x25-y16)
 	(connected loc-x25-y17 loc-x25-y18)
 	(connected loc-x25-y18 loc-x24-y18)
 	(connected loc-x25-y18 loc-x26-y18)
 	(connected loc-x25-y18 loc-x25-y17)
 	(connected loc-x25-y18 loc-x25-y19)
 	(connected loc-x25-y19 loc-x24-y19)
 	(connected loc-x25-y19 loc-x26-y19)
 	(connected loc-x25-y19 loc-x25-y18)
 	(connected loc-x25-y19 loc-x25-y20)
 	(connected loc-x25-y20 loc-x24-y20)
 	(connected loc-x25-y20 loc-x26-y20)
 	(connected loc-x25-y20 loc-x25-y19)
 	(connected loc-x25-y20 loc-x25-y21)
 	(connected loc-x25-y21 loc-x24-y21)
 	(connected loc-x25-y21 loc-x26-y21)
 	(connected loc-x25-y21 loc-x25-y20)
 	(connected loc-x25-y21 loc-x25-y22)
 	(connected loc-x25-y22 loc-x24-y22)
 	(connected loc-x25-y22 loc-x26-y22)
 	(connected loc-x25-y22 loc-x25-y21)
 	(connected loc-x25-y22 loc-x25-y23)
 	(connected loc-x25-y23 loc-x24-y23)
 	(connected loc-x25-y23 loc-x26-y23)
 	(connected loc-x25-y23 loc-x25-y22)
 	(connected loc-x25-y23 loc-x25-y24)
 	(connected loc-x25-y24 loc-x24-y24)
 	(connected loc-x25-y24 loc-x26-y24)
 	(connected loc-x25-y24 loc-x25-y23)
 	(connected loc-x25-y24 loc-x25-y25)
 	(connected loc-x25-y25 loc-x24-y25)
 	(connected loc-x25-y25 loc-x26-y25)
 	(connected loc-x25-y25 loc-x25-y24)
 	(connected loc-x25-y25 loc-x25-y26)
 	(connected loc-x25-y26 loc-x24-y26)
 	(connected loc-x25-y26 loc-x26-y26)
 	(connected loc-x25-y26 loc-x25-y25)
 	(connected loc-x25-y26 loc-x25-y27)
 	(connected loc-x25-y27 loc-x24-y27)
 	(connected loc-x25-y27 loc-x26-y27)
 	(connected loc-x25-y27 loc-x25-y26)
 	(connected loc-x25-y27 loc-x25-y28)
 	(connected loc-x25-y28 loc-x24-y28)
 	(connected loc-x25-y28 loc-x26-y28)
 	(connected loc-x25-y28 loc-x25-y27)
 	(connected loc-x25-y28 loc-x25-y29)
 	(connected loc-x25-y29 loc-x24-y29)
 	(connected loc-x25-y29 loc-x26-y29)
 	(connected loc-x25-y29 loc-x25-y28)
 	(connected loc-x25-y29 loc-x25-y30)
 	(connected loc-x25-y30 loc-x24-y30)
 	(connected loc-x25-y30 loc-x26-y30)
 	(connected loc-x25-y30 loc-x25-y29)
 	(connected loc-x26-y0 loc-x25-y0)
 	(connected loc-x26-y0 loc-x27-y0)
 	(connected loc-x26-y0 loc-x26-y1)
 	(connected loc-x26-y1 loc-x25-y1)
 	(connected loc-x26-y1 loc-x27-y1)
 	(connected loc-x26-y1 loc-x26-y0)
 	(connected loc-x26-y1 loc-x26-y2)
 	(connected loc-x26-y2 loc-x25-y2)
 	(connected loc-x26-y2 loc-x27-y2)
 	(connected loc-x26-y2 loc-x26-y1)
 	(connected loc-x26-y2 loc-x26-y3)
 	(connected loc-x26-y3 loc-x25-y3)
 	(connected loc-x26-y3 loc-x27-y3)
 	(connected loc-x26-y3 loc-x26-y2)
 	(connected loc-x26-y3 loc-x26-y4)
 	(connected loc-x26-y4 loc-x25-y4)
 	(connected loc-x26-y4 loc-x27-y4)
 	(connected loc-x26-y4 loc-x26-y3)
 	(connected loc-x26-y4 loc-x26-y5)
 	(connected loc-x26-y5 loc-x25-y5)
 	(connected loc-x26-y5 loc-x27-y5)
 	(connected loc-x26-y5 loc-x26-y4)
 	(connected loc-x26-y5 loc-x26-y6)
 	(connected loc-x26-y6 loc-x25-y6)
 	(connected loc-x26-y6 loc-x27-y6)
 	(connected loc-x26-y6 loc-x26-y5)
 	(connected loc-x26-y6 loc-x26-y7)
 	(connected loc-x26-y7 loc-x25-y7)
 	(connected loc-x26-y7 loc-x27-y7)
 	(connected loc-x26-y7 loc-x26-y6)
 	(connected loc-x26-y7 loc-x26-y8)
 	(connected loc-x26-y8 loc-x25-y8)
 	(connected loc-x26-y8 loc-x27-y8)
 	(connected loc-x26-y8 loc-x26-y7)
 	(connected loc-x26-y8 loc-x26-y9)
 	(connected loc-x26-y9 loc-x25-y9)
 	(connected loc-x26-y9 loc-x27-y9)
 	(connected loc-x26-y9 loc-x26-y8)
 	(connected loc-x26-y9 loc-x26-y10)
 	(connected loc-x26-y10 loc-x25-y10)
 	(connected loc-x26-y10 loc-x27-y10)
 	(connected loc-x26-y10 loc-x26-y9)
 	(connected loc-x26-y10 loc-x26-y11)
 	(connected loc-x26-y11 loc-x25-y11)
 	(connected loc-x26-y11 loc-x27-y11)
 	(connected loc-x26-y11 loc-x26-y10)
 	(connected loc-x26-y11 loc-x26-y12)
 	(connected loc-x26-y12 loc-x25-y12)
 	(connected loc-x26-y12 loc-x27-y12)
 	(connected loc-x26-y12 loc-x26-y11)
 	(connected loc-x26-y12 loc-x26-y13)
 	(connected loc-x26-y13 loc-x25-y13)
 	(connected loc-x26-y13 loc-x27-y13)
 	(connected loc-x26-y13 loc-x26-y12)
 	(connected loc-x26-y13 loc-x26-y14)
 	(connected loc-x26-y14 loc-x25-y14)
 	(connected loc-x26-y14 loc-x27-y14)
 	(connected loc-x26-y14 loc-x26-y13)
 	(connected loc-x26-y14 loc-x26-y15)
 	(connected loc-x26-y15 loc-x25-y15)
 	(connected loc-x26-y15 loc-x27-y15)
 	(connected loc-x26-y15 loc-x26-y14)
 	(connected loc-x26-y15 loc-x26-y16)
 	(connected loc-x26-y16 loc-x25-y16)
 	(connected loc-x26-y16 loc-x27-y16)
 	(connected loc-x26-y16 loc-x26-y15)
 	(connected loc-x26-y16 loc-x26-y17)
 	(connected loc-x26-y17 loc-x25-y17)
 	(connected loc-x26-y17 loc-x27-y17)
 	(connected loc-x26-y17 loc-x26-y16)
 	(connected loc-x26-y17 loc-x26-y18)
 	(connected loc-x26-y18 loc-x25-y18)
 	(connected loc-x26-y18 loc-x27-y18)
 	(connected loc-x26-y18 loc-x26-y17)
 	(connected loc-x26-y18 loc-x26-y19)
 	(connected loc-x26-y19 loc-x25-y19)
 	(connected loc-x26-y19 loc-x27-y19)
 	(connected loc-x26-y19 loc-x26-y18)
 	(connected loc-x26-y19 loc-x26-y20)
 	(connected loc-x26-y20 loc-x25-y20)
 	(connected loc-x26-y20 loc-x27-y20)
 	(connected loc-x26-y20 loc-x26-y19)
 	(connected loc-x26-y20 loc-x26-y21)
 	(connected loc-x26-y21 loc-x25-y21)
 	(connected loc-x26-y21 loc-x27-y21)
 	(connected loc-x26-y21 loc-x26-y20)
 	(connected loc-x26-y21 loc-x26-y22)
 	(connected loc-x26-y22 loc-x25-y22)
 	(connected loc-x26-y22 loc-x27-y22)
 	(connected loc-x26-y22 loc-x26-y21)
 	(connected loc-x26-y22 loc-x26-y23)
 	(connected loc-x26-y23 loc-x25-y23)
 	(connected loc-x26-y23 loc-x27-y23)
 	(connected loc-x26-y23 loc-x26-y22)
 	(connected loc-x26-y23 loc-x26-y24)
 	(connected loc-x26-y24 loc-x25-y24)
 	(connected loc-x26-y24 loc-x27-y24)
 	(connected loc-x26-y24 loc-x26-y23)
 	(connected loc-x26-y24 loc-x26-y25)
 	(connected loc-x26-y25 loc-x25-y25)
 	(connected loc-x26-y25 loc-x27-y25)
 	(connected loc-x26-y25 loc-x26-y24)
 	(connected loc-x26-y25 loc-x26-y26)
 	(connected loc-x26-y26 loc-x25-y26)
 	(connected loc-x26-y26 loc-x27-y26)
 	(connected loc-x26-y26 loc-x26-y25)
 	(connected loc-x26-y26 loc-x26-y27)
 	(connected loc-x26-y27 loc-x25-y27)
 	(connected loc-x26-y27 loc-x27-y27)
 	(connected loc-x26-y27 loc-x26-y26)
 	(connected loc-x26-y27 loc-x26-y28)
 	(connected loc-x26-y28 loc-x25-y28)
 	(connected loc-x26-y28 loc-x27-y28)
 	(connected loc-x26-y28 loc-x26-y27)
 	(connected loc-x26-y28 loc-x26-y29)
 	(connected loc-x26-y29 loc-x25-y29)
 	(connected loc-x26-y29 loc-x27-y29)
 	(connected loc-x26-y29 loc-x26-y28)
 	(connected loc-x26-y29 loc-x26-y30)
 	(connected loc-x26-y30 loc-x25-y30)
 	(connected loc-x26-y30 loc-x27-y30)
 	(connected loc-x26-y30 loc-x26-y29)
 	(connected loc-x27-y0 loc-x26-y0)
 	(connected loc-x27-y0 loc-x28-y0)
 	(connected loc-x27-y0 loc-x27-y1)
 	(connected loc-x27-y1 loc-x26-y1)
 	(connected loc-x27-y1 loc-x28-y1)
 	(connected loc-x27-y1 loc-x27-y0)
 	(connected loc-x27-y1 loc-x27-y2)
 	(connected loc-x27-y2 loc-x26-y2)
 	(connected loc-x27-y2 loc-x28-y2)
 	(connected loc-x27-y2 loc-x27-y1)
 	(connected loc-x27-y2 loc-x27-y3)
 	(connected loc-x27-y3 loc-x26-y3)
 	(connected loc-x27-y3 loc-x28-y3)
 	(connected loc-x27-y3 loc-x27-y2)
 	(connected loc-x27-y3 loc-x27-y4)
 	(connected loc-x27-y4 loc-x26-y4)
 	(connected loc-x27-y4 loc-x28-y4)
 	(connected loc-x27-y4 loc-x27-y3)
 	(connected loc-x27-y4 loc-x27-y5)
 	(connected loc-x27-y5 loc-x26-y5)
 	(connected loc-x27-y5 loc-x28-y5)
 	(connected loc-x27-y5 loc-x27-y4)
 	(connected loc-x27-y5 loc-x27-y6)
 	(connected loc-x27-y6 loc-x26-y6)
 	(connected loc-x27-y6 loc-x28-y6)
 	(connected loc-x27-y6 loc-x27-y5)
 	(connected loc-x27-y6 loc-x27-y7)
 	(connected loc-x27-y7 loc-x26-y7)
 	(connected loc-x27-y7 loc-x28-y7)
 	(connected loc-x27-y7 loc-x27-y6)
 	(connected loc-x27-y7 loc-x27-y8)
 	(connected loc-x27-y8 loc-x26-y8)
 	(connected loc-x27-y8 loc-x28-y8)
 	(connected loc-x27-y8 loc-x27-y7)
 	(connected loc-x27-y8 loc-x27-y9)
 	(connected loc-x27-y9 loc-x26-y9)
 	(connected loc-x27-y9 loc-x28-y9)
 	(connected loc-x27-y9 loc-x27-y8)
 	(connected loc-x27-y9 loc-x27-y10)
 	(connected loc-x27-y10 loc-x26-y10)
 	(connected loc-x27-y10 loc-x28-y10)
 	(connected loc-x27-y10 loc-x27-y9)
 	(connected loc-x27-y10 loc-x27-y11)
 	(connected loc-x27-y11 loc-x26-y11)
 	(connected loc-x27-y11 loc-x28-y11)
 	(connected loc-x27-y11 loc-x27-y10)
 	(connected loc-x27-y11 loc-x27-y12)
 	(connected loc-x27-y12 loc-x26-y12)
 	(connected loc-x27-y12 loc-x28-y12)
 	(connected loc-x27-y12 loc-x27-y11)
 	(connected loc-x27-y12 loc-x27-y13)
 	(connected loc-x27-y13 loc-x26-y13)
 	(connected loc-x27-y13 loc-x28-y13)
 	(connected loc-x27-y13 loc-x27-y12)
 	(connected loc-x27-y13 loc-x27-y14)
 	(connected loc-x27-y14 loc-x26-y14)
 	(connected loc-x27-y14 loc-x28-y14)
 	(connected loc-x27-y14 loc-x27-y13)
 	(connected loc-x27-y14 loc-x27-y15)
 	(connected loc-x27-y15 loc-x26-y15)
 	(connected loc-x27-y15 loc-x28-y15)
 	(connected loc-x27-y15 loc-x27-y14)
 	(connected loc-x27-y15 loc-x27-y16)
 	(connected loc-x27-y16 loc-x26-y16)
 	(connected loc-x27-y16 loc-x28-y16)
 	(connected loc-x27-y16 loc-x27-y15)
 	(connected loc-x27-y16 loc-x27-y17)
 	(connected loc-x27-y17 loc-x26-y17)
 	(connected loc-x27-y17 loc-x28-y17)
 	(connected loc-x27-y17 loc-x27-y16)
 	(connected loc-x27-y17 loc-x27-y18)
 	(connected loc-x27-y18 loc-x26-y18)
 	(connected loc-x27-y18 loc-x28-y18)
 	(connected loc-x27-y18 loc-x27-y17)
 	(connected loc-x27-y18 loc-x27-y19)
 	(connected loc-x27-y19 loc-x26-y19)
 	(connected loc-x27-y19 loc-x28-y19)
 	(connected loc-x27-y19 loc-x27-y18)
 	(connected loc-x27-y19 loc-x27-y20)
 	(connected loc-x27-y20 loc-x26-y20)
 	(connected loc-x27-y20 loc-x28-y20)
 	(connected loc-x27-y20 loc-x27-y19)
 	(connected loc-x27-y20 loc-x27-y21)
 	(connected loc-x27-y21 loc-x26-y21)
 	(connected loc-x27-y21 loc-x28-y21)
 	(connected loc-x27-y21 loc-x27-y20)
 	(connected loc-x27-y21 loc-x27-y22)
 	(connected loc-x27-y22 loc-x26-y22)
 	(connected loc-x27-y22 loc-x28-y22)
 	(connected loc-x27-y22 loc-x27-y21)
 	(connected loc-x27-y22 loc-x27-y23)
 	(connected loc-x27-y23 loc-x26-y23)
 	(connected loc-x27-y23 loc-x28-y23)
 	(connected loc-x27-y23 loc-x27-y22)
 	(connected loc-x27-y23 loc-x27-y24)
 	(connected loc-x27-y24 loc-x26-y24)
 	(connected loc-x27-y24 loc-x28-y24)
 	(connected loc-x27-y24 loc-x27-y23)
 	(connected loc-x27-y24 loc-x27-y25)
 	(connected loc-x27-y25 loc-x26-y25)
 	(connected loc-x27-y25 loc-x28-y25)
 	(connected loc-x27-y25 loc-x27-y24)
 	(connected loc-x27-y25 loc-x27-y26)
 	(connected loc-x27-y26 loc-x26-y26)
 	(connected loc-x27-y26 loc-x28-y26)
 	(connected loc-x27-y26 loc-x27-y25)
 	(connected loc-x27-y26 loc-x27-y27)
 	(connected loc-x27-y27 loc-x26-y27)
 	(connected loc-x27-y27 loc-x28-y27)
 	(connected loc-x27-y27 loc-x27-y26)
 	(connected loc-x27-y27 loc-x27-y28)
 	(connected loc-x27-y28 loc-x26-y28)
 	(connected loc-x27-y28 loc-x28-y28)
 	(connected loc-x27-y28 loc-x27-y27)
 	(connected loc-x27-y28 loc-x27-y29)
 	(connected loc-x27-y29 loc-x26-y29)
 	(connected loc-x27-y29 loc-x28-y29)
 	(connected loc-x27-y29 loc-x27-y28)
 	(connected loc-x27-y29 loc-x27-y30)
 	(connected loc-x27-y30 loc-x26-y30)
 	(connected loc-x27-y30 loc-x28-y30)
 	(connected loc-x27-y30 loc-x27-y29)
 	(connected loc-x28-y0 loc-x27-y0)
 	(connected loc-x28-y0 loc-x29-y0)
 	(connected loc-x28-y0 loc-x28-y1)
 	(connected loc-x28-y1 loc-x27-y1)
 	(connected loc-x28-y1 loc-x29-y1)
 	(connected loc-x28-y1 loc-x28-y0)
 	(connected loc-x28-y1 loc-x28-y2)
 	(connected loc-x28-y2 loc-x27-y2)
 	(connected loc-x28-y2 loc-x29-y2)
 	(connected loc-x28-y2 loc-x28-y1)
 	(connected loc-x28-y2 loc-x28-y3)
 	(connected loc-x28-y3 loc-x27-y3)
 	(connected loc-x28-y3 loc-x29-y3)
 	(connected loc-x28-y3 loc-x28-y2)
 	(connected loc-x28-y3 loc-x28-y4)
 	(connected loc-x28-y4 loc-x27-y4)
 	(connected loc-x28-y4 loc-x29-y4)
 	(connected loc-x28-y4 loc-x28-y3)
 	(connected loc-x28-y4 loc-x28-y5)
 	(connected loc-x28-y5 loc-x27-y5)
 	(connected loc-x28-y5 loc-x29-y5)
 	(connected loc-x28-y5 loc-x28-y4)
 	(connected loc-x28-y5 loc-x28-y6)
 	(connected loc-x28-y6 loc-x27-y6)
 	(connected loc-x28-y6 loc-x29-y6)
 	(connected loc-x28-y6 loc-x28-y5)
 	(connected loc-x28-y6 loc-x28-y7)
 	(connected loc-x28-y7 loc-x27-y7)
 	(connected loc-x28-y7 loc-x29-y7)
 	(connected loc-x28-y7 loc-x28-y6)
 	(connected loc-x28-y7 loc-x28-y8)
 	(connected loc-x28-y8 loc-x27-y8)
 	(connected loc-x28-y8 loc-x29-y8)
 	(connected loc-x28-y8 loc-x28-y7)
 	(connected loc-x28-y8 loc-x28-y9)
 	(connected loc-x28-y9 loc-x27-y9)
 	(connected loc-x28-y9 loc-x29-y9)
 	(connected loc-x28-y9 loc-x28-y8)
 	(connected loc-x28-y9 loc-x28-y10)
 	(connected loc-x28-y10 loc-x27-y10)
 	(connected loc-x28-y10 loc-x29-y10)
 	(connected loc-x28-y10 loc-x28-y9)
 	(connected loc-x28-y10 loc-x28-y11)
 	(connected loc-x28-y11 loc-x27-y11)
 	(connected loc-x28-y11 loc-x29-y11)
 	(connected loc-x28-y11 loc-x28-y10)
 	(connected loc-x28-y11 loc-x28-y12)
 	(connected loc-x28-y12 loc-x27-y12)
 	(connected loc-x28-y12 loc-x29-y12)
 	(connected loc-x28-y12 loc-x28-y11)
 	(connected loc-x28-y12 loc-x28-y13)
 	(connected loc-x28-y13 loc-x27-y13)
 	(connected loc-x28-y13 loc-x29-y13)
 	(connected loc-x28-y13 loc-x28-y12)
 	(connected loc-x28-y13 loc-x28-y14)
 	(connected loc-x28-y14 loc-x27-y14)
 	(connected loc-x28-y14 loc-x29-y14)
 	(connected loc-x28-y14 loc-x28-y13)
 	(connected loc-x28-y14 loc-x28-y15)
 	(connected loc-x28-y15 loc-x27-y15)
 	(connected loc-x28-y15 loc-x29-y15)
 	(connected loc-x28-y15 loc-x28-y14)
 	(connected loc-x28-y15 loc-x28-y16)
 	(connected loc-x28-y16 loc-x27-y16)
 	(connected loc-x28-y16 loc-x29-y16)
 	(connected loc-x28-y16 loc-x28-y15)
 	(connected loc-x28-y16 loc-x28-y17)
 	(connected loc-x28-y17 loc-x27-y17)
 	(connected loc-x28-y17 loc-x29-y17)
 	(connected loc-x28-y17 loc-x28-y16)
 	(connected loc-x28-y17 loc-x28-y18)
 	(connected loc-x28-y18 loc-x27-y18)
 	(connected loc-x28-y18 loc-x29-y18)
 	(connected loc-x28-y18 loc-x28-y17)
 	(connected loc-x28-y18 loc-x28-y19)
 	(connected loc-x28-y19 loc-x27-y19)
 	(connected loc-x28-y19 loc-x29-y19)
 	(connected loc-x28-y19 loc-x28-y18)
 	(connected loc-x28-y19 loc-x28-y20)
 	(connected loc-x28-y20 loc-x27-y20)
 	(connected loc-x28-y20 loc-x29-y20)
 	(connected loc-x28-y20 loc-x28-y19)
 	(connected loc-x28-y20 loc-x28-y21)
 	(connected loc-x28-y21 loc-x27-y21)
 	(connected loc-x28-y21 loc-x29-y21)
 	(connected loc-x28-y21 loc-x28-y20)
 	(connected loc-x28-y21 loc-x28-y22)
 	(connected loc-x28-y22 loc-x27-y22)
 	(connected loc-x28-y22 loc-x29-y22)
 	(connected loc-x28-y22 loc-x28-y21)
 	(connected loc-x28-y22 loc-x28-y23)
 	(connected loc-x28-y23 loc-x27-y23)
 	(connected loc-x28-y23 loc-x29-y23)
 	(connected loc-x28-y23 loc-x28-y22)
 	(connected loc-x28-y23 loc-x28-y24)
 	(connected loc-x28-y24 loc-x27-y24)
 	(connected loc-x28-y24 loc-x29-y24)
 	(connected loc-x28-y24 loc-x28-y23)
 	(connected loc-x28-y24 loc-x28-y25)
 	(connected loc-x28-y25 loc-x27-y25)
 	(connected loc-x28-y25 loc-x29-y25)
 	(connected loc-x28-y25 loc-x28-y24)
 	(connected loc-x28-y25 loc-x28-y26)
 	(connected loc-x28-y26 loc-x27-y26)
 	(connected loc-x28-y26 loc-x29-y26)
 	(connected loc-x28-y26 loc-x28-y25)
 	(connected loc-x28-y26 loc-x28-y27)
 	(connected loc-x28-y27 loc-x27-y27)
 	(connected loc-x28-y27 loc-x29-y27)
 	(connected loc-x28-y27 loc-x28-y26)
 	(connected loc-x28-y27 loc-x28-y28)
 	(connected loc-x28-y28 loc-x27-y28)
 	(connected loc-x28-y28 loc-x29-y28)
 	(connected loc-x28-y28 loc-x28-y27)
 	(connected loc-x28-y28 loc-x28-y29)
 	(connected loc-x28-y29 loc-x27-y29)
 	(connected loc-x28-y29 loc-x29-y29)
 	(connected loc-x28-y29 loc-x28-y28)
 	(connected loc-x28-y29 loc-x28-y30)
 	(connected loc-x28-y30 loc-x27-y30)
 	(connected loc-x28-y30 loc-x29-y30)
 	(connected loc-x28-y30 loc-x28-y29)
 	(connected loc-x29-y0 loc-x28-y0)
 	(connected loc-x29-y0 loc-x30-y0)
 	(connected loc-x29-y0 loc-x29-y1)
 	(connected loc-x29-y1 loc-x28-y1)
 	(connected loc-x29-y1 loc-x30-y1)
 	(connected loc-x29-y1 loc-x29-y0)
 	(connected loc-x29-y1 loc-x29-y2)
 	(connected loc-x29-y2 loc-x28-y2)
 	(connected loc-x29-y2 loc-x30-y2)
 	(connected loc-x29-y2 loc-x29-y1)
 	(connected loc-x29-y2 loc-x29-y3)
 	(connected loc-x29-y3 loc-x28-y3)
 	(connected loc-x29-y3 loc-x30-y3)
 	(connected loc-x29-y3 loc-x29-y2)
 	(connected loc-x29-y3 loc-x29-y4)
 	(connected loc-x29-y4 loc-x28-y4)
 	(connected loc-x29-y4 loc-x30-y4)
 	(connected loc-x29-y4 loc-x29-y3)
 	(connected loc-x29-y4 loc-x29-y5)
 	(connected loc-x29-y5 loc-x28-y5)
 	(connected loc-x29-y5 loc-x30-y5)
 	(connected loc-x29-y5 loc-x29-y4)
 	(connected loc-x29-y5 loc-x29-y6)
 	(connected loc-x29-y6 loc-x28-y6)
 	(connected loc-x29-y6 loc-x30-y6)
 	(connected loc-x29-y6 loc-x29-y5)
 	(connected loc-x29-y6 loc-x29-y7)
 	(connected loc-x29-y7 loc-x28-y7)
 	(connected loc-x29-y7 loc-x30-y7)
 	(connected loc-x29-y7 loc-x29-y6)
 	(connected loc-x29-y7 loc-x29-y8)
 	(connected loc-x29-y8 loc-x28-y8)
 	(connected loc-x29-y8 loc-x30-y8)
 	(connected loc-x29-y8 loc-x29-y7)
 	(connected loc-x29-y8 loc-x29-y9)
 	(connected loc-x29-y9 loc-x28-y9)
 	(connected loc-x29-y9 loc-x30-y9)
 	(connected loc-x29-y9 loc-x29-y8)
 	(connected loc-x29-y9 loc-x29-y10)
 	(connected loc-x29-y10 loc-x28-y10)
 	(connected loc-x29-y10 loc-x30-y10)
 	(connected loc-x29-y10 loc-x29-y9)
 	(connected loc-x29-y10 loc-x29-y11)
 	(connected loc-x29-y11 loc-x28-y11)
 	(connected loc-x29-y11 loc-x30-y11)
 	(connected loc-x29-y11 loc-x29-y10)
 	(connected loc-x29-y11 loc-x29-y12)
 	(connected loc-x29-y12 loc-x28-y12)
 	(connected loc-x29-y12 loc-x30-y12)
 	(connected loc-x29-y12 loc-x29-y11)
 	(connected loc-x29-y12 loc-x29-y13)
 	(connected loc-x29-y13 loc-x28-y13)
 	(connected loc-x29-y13 loc-x30-y13)
 	(connected loc-x29-y13 loc-x29-y12)
 	(connected loc-x29-y13 loc-x29-y14)
 	(connected loc-x29-y14 loc-x28-y14)
 	(connected loc-x29-y14 loc-x30-y14)
 	(connected loc-x29-y14 loc-x29-y13)
 	(connected loc-x29-y14 loc-x29-y15)
 	(connected loc-x29-y15 loc-x28-y15)
 	(connected loc-x29-y15 loc-x30-y15)
 	(connected loc-x29-y15 loc-x29-y14)
 	(connected loc-x29-y15 loc-x29-y16)
 	(connected loc-x29-y16 loc-x28-y16)
 	(connected loc-x29-y16 loc-x30-y16)
 	(connected loc-x29-y16 loc-x29-y15)
 	(connected loc-x29-y16 loc-x29-y17)
 	(connected loc-x29-y17 loc-x28-y17)
 	(connected loc-x29-y17 loc-x30-y17)
 	(connected loc-x29-y17 loc-x29-y16)
 	(connected loc-x29-y17 loc-x29-y18)
 	(connected loc-x29-y18 loc-x28-y18)
 	(connected loc-x29-y18 loc-x30-y18)
 	(connected loc-x29-y18 loc-x29-y17)
 	(connected loc-x29-y18 loc-x29-y19)
 	(connected loc-x29-y19 loc-x28-y19)
 	(connected loc-x29-y19 loc-x30-y19)
 	(connected loc-x29-y19 loc-x29-y18)
 	(connected loc-x29-y19 loc-x29-y20)
 	(connected loc-x29-y20 loc-x28-y20)
 	(connected loc-x29-y20 loc-x30-y20)
 	(connected loc-x29-y20 loc-x29-y19)
 	(connected loc-x29-y20 loc-x29-y21)
 	(connected loc-x29-y21 loc-x28-y21)
 	(connected loc-x29-y21 loc-x30-y21)
 	(connected loc-x29-y21 loc-x29-y20)
 	(connected loc-x29-y21 loc-x29-y22)
 	(connected loc-x29-y22 loc-x28-y22)
 	(connected loc-x29-y22 loc-x30-y22)
 	(connected loc-x29-y22 loc-x29-y21)
 	(connected loc-x29-y22 loc-x29-y23)
 	(connected loc-x29-y23 loc-x28-y23)
 	(connected loc-x29-y23 loc-x30-y23)
 	(connected loc-x29-y23 loc-x29-y22)
 	(connected loc-x29-y23 loc-x29-y24)
 	(connected loc-x29-y24 loc-x28-y24)
 	(connected loc-x29-y24 loc-x30-y24)
 	(connected loc-x29-y24 loc-x29-y23)
 	(connected loc-x29-y24 loc-x29-y25)
 	(connected loc-x29-y25 loc-x28-y25)
 	(connected loc-x29-y25 loc-x30-y25)
 	(connected loc-x29-y25 loc-x29-y24)
 	(connected loc-x29-y25 loc-x29-y26)
 	(connected loc-x29-y26 loc-x28-y26)
 	(connected loc-x29-y26 loc-x30-y26)
 	(connected loc-x29-y26 loc-x29-y25)
 	(connected loc-x29-y26 loc-x29-y27)
 	(connected loc-x29-y27 loc-x28-y27)
 	(connected loc-x29-y27 loc-x30-y27)
 	(connected loc-x29-y27 loc-x29-y26)
 	(connected loc-x29-y27 loc-x29-y28)
 	(connected loc-x29-y28 loc-x28-y28)
 	(connected loc-x29-y28 loc-x30-y28)
 	(connected loc-x29-y28 loc-x29-y27)
 	(connected loc-x29-y28 loc-x29-y29)
 	(connected loc-x29-y29 loc-x28-y29)
 	(connected loc-x29-y29 loc-x30-y29)
 	(connected loc-x29-y29 loc-x29-y28)
 	(connected loc-x29-y29 loc-x29-y30)
 	(connected loc-x29-y30 loc-x28-y30)
 	(connected loc-x29-y30 loc-x30-y30)
 	(connected loc-x29-y30 loc-x29-y29)
 	(connected loc-x30-y0 loc-x29-y0)
 	(connected loc-x30-y0 loc-x30-y1)
 	(connected loc-x30-y1 loc-x29-y1)
 	(connected loc-x30-y1 loc-x30-y0)
 	(connected loc-x30-y1 loc-x30-y2)
 	(connected loc-x30-y2 loc-x29-y2)
 	(connected loc-x30-y2 loc-x30-y1)
 	(connected loc-x30-y2 loc-x30-y3)
 	(connected loc-x30-y3 loc-x29-y3)
 	(connected loc-x30-y3 loc-x30-y2)
 	(connected loc-x30-y3 loc-x30-y4)
 	(connected loc-x30-y4 loc-x29-y4)
 	(connected loc-x30-y4 loc-x30-y3)
 	(connected loc-x30-y4 loc-x30-y5)
 	(connected loc-x30-y5 loc-x29-y5)
 	(connected loc-x30-y5 loc-x30-y4)
 	(connected loc-x30-y5 loc-x30-y6)
 	(connected loc-x30-y6 loc-x29-y6)
 	(connected loc-x30-y6 loc-x30-y5)
 	(connected loc-x30-y6 loc-x30-y7)
 	(connected loc-x30-y7 loc-x29-y7)
 	(connected loc-x30-y7 loc-x30-y6)
 	(connected loc-x30-y7 loc-x30-y8)
 	(connected loc-x30-y8 loc-x29-y8)
 	(connected loc-x30-y8 loc-x30-y7)
 	(connected loc-x30-y8 loc-x30-y9)
 	(connected loc-x30-y9 loc-x29-y9)
 	(connected loc-x30-y9 loc-x30-y8)
 	(connected loc-x30-y9 loc-x30-y10)
 	(connected loc-x30-y10 loc-x29-y10)
 	(connected loc-x30-y10 loc-x30-y9)
 	(connected loc-x30-y10 loc-x30-y11)
 	(connected loc-x30-y11 loc-x29-y11)
 	(connected loc-x30-y11 loc-x30-y10)
 	(connected loc-x30-y11 loc-x30-y12)
 	(connected loc-x30-y12 loc-x29-y12)
 	(connected loc-x30-y12 loc-x30-y11)
 	(connected loc-x30-y12 loc-x30-y13)
 	(connected loc-x30-y13 loc-x29-y13)
 	(connected loc-x30-y13 loc-x30-y12)
 	(connected loc-x30-y13 loc-x30-y14)
 	(connected loc-x30-y14 loc-x29-y14)
 	(connected loc-x30-y14 loc-x30-y13)
 	(connected loc-x30-y14 loc-x30-y15)
 	(connected loc-x30-y15 loc-x29-y15)
 	(connected loc-x30-y15 loc-x30-y14)
 	(connected loc-x30-y15 loc-x30-y16)
 	(connected loc-x30-y16 loc-x29-y16)
 	(connected loc-x30-y16 loc-x30-y15)
 	(connected loc-x30-y16 loc-x30-y17)
 	(connected loc-x30-y17 loc-x29-y17)
 	(connected loc-x30-y17 loc-x30-y16)
 	(connected loc-x30-y17 loc-x30-y18)
 	(connected loc-x30-y18 loc-x29-y18)
 	(connected loc-x30-y18 loc-x30-y17)
 	(connected loc-x30-y18 loc-x30-y19)
 	(connected loc-x30-y19 loc-x29-y19)
 	(connected loc-x30-y19 loc-x30-y18)
 	(connected loc-x30-y19 loc-x30-y20)
 	(connected loc-x30-y20 loc-x29-y20)
 	(connected loc-x30-y20 loc-x30-y19)
 	(connected loc-x30-y20 loc-x30-y21)
 	(connected loc-x30-y21 loc-x29-y21)
 	(connected loc-x30-y21 loc-x30-y20)
 	(connected loc-x30-y21 loc-x30-y22)
 	(connected loc-x30-y22 loc-x29-y22)
 	(connected loc-x30-y22 loc-x30-y21)
 	(connected loc-x30-y22 loc-x30-y23)
 	(connected loc-x30-y23 loc-x29-y23)
 	(connected loc-x30-y23 loc-x30-y22)
 	(connected loc-x30-y23 loc-x30-y24)
 	(connected loc-x30-y24 loc-x29-y24)
 	(connected loc-x30-y24 loc-x30-y23)
 	(connected loc-x30-y24 loc-x30-y25)
 	(connected loc-x30-y25 loc-x29-y25)
 	(connected loc-x30-y25 loc-x30-y24)
 	(connected loc-x30-y25 loc-x30-y26)
 	(connected loc-x30-y26 loc-x29-y26)
 	(connected loc-x30-y26 loc-x30-y25)
 	(connected loc-x30-y26 loc-x30-y27)
 	(connected loc-x30-y27 loc-x29-y27)
 	(connected loc-x30-y27 loc-x30-y26)
 	(connected loc-x30-y27 loc-x30-y28)
 	(connected loc-x30-y28 loc-x29-y28)
 	(connected loc-x30-y28 loc-x30-y27)
 	(connected loc-x30-y28 loc-x30-y29)
 	(connected loc-x30-y29 loc-x29-y29)
 	(connected loc-x30-y29 loc-x30-y28)
 	(connected loc-x30-y29 loc-x30-y30)
 	(connected loc-x30-y30 loc-x29-y30)
 	(connected loc-x30-y30 loc-x30-y29)
 
)
(:goal
(and 
	(visited loc-x0-y23)
	(visited loc-x1-y0)
	(visited loc-x1-y11)
	(visited loc-x1-y16)
	(visited loc-x3-y3)
	(visited loc-x3-y7)
	(visited loc-x3-y9)
	(visited loc-x3-y22)
	(visited loc-x3-y30)
	(visited loc-x4-y0)
	(visited loc-x4-y6)
	(visited loc-x4-y10)
	(visited loc-x4-y25)
	(visited loc-x4-y26)
	(visited loc-x5-y24)
	(visited loc-x5-y27)
	(visited loc-x6-y3)
	(visited loc-x6-y12)
	(visited loc-x7-y12)
	(visited loc-x7-y16)
	(visited loc-x7-y18)
	(visited loc-x8-y9)
	(visited loc-x9-y9)
	(visited loc-x9-y12)
	(visited loc-x9-y20)
	(visited loc-x10-y1)
	(visited loc-x10-y8)
	(visited loc-x10-y19)
	(visited loc-x11-y9)
	(visited loc-x11-y21)
	(visited loc-x11-y23)
	(visited loc-x12-y0)
	(visited loc-x12-y10)
	(visited loc-x13-y0)
	(visited loc-x13-y3)
	(visited loc-x13-y10)
	(visited loc-x14-y11)
	(visited loc-x14-y22)
	(visited loc-x14-y28)
	(visited loc-x15-y0)
	(visited loc-x15-y3)
	(visited loc-x15-y4)
	(visited loc-x15-y30)
	(visited loc-x16-y4)
	(visited loc-x16-y21)
	(visited loc-x17-y0)
	(visited loc-x17-y17)
	(visited loc-x18-y21)
	(visited loc-x18-y27)
	(visited loc-x18-y30)
	(visited loc-x19-y21)
	(visited loc-x19-y28)
	(visited loc-x20-y1)
	(visited loc-x20-y21)
	(visited loc-x20-y29)
	(visited loc-x21-y4)
	(visited loc-x21-y9)
	(visited loc-x22-y7)
	(visited loc-x22-y8)
	(visited loc-x22-y11)
	(visited loc-x22-y18)
	(visited loc-x22-y19)
	(visited loc-x22-y29)
	(visited loc-x23-y12)
	(visited loc-x24-y18)
	(visited loc-x25-y3)
	(visited loc-x25-y12)
	(visited loc-x26-y2)
	(visited loc-x26-y8)
	(visited loc-x26-y21)
	(visited loc-x26-y28)
	(visited loc-x27-y3)
	(visited loc-x27-y8)
	(visited loc-x27-y18)
	(visited loc-x27-y22)
	(visited loc-x27-y29)
	(visited loc-x28-y11)
	(visited loc-x28-y20)
	(visited loc-x29-y22)
	(visited loc-x30-y0)
	(visited loc-x30-y4)
	(visited loc-x30-y18)
	(visited loc-x30-y19)
)
)
)