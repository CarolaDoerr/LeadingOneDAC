(define (problem roverprob) (:domain Rover)
(:objects
	general - Lander
	colour high_res low_res - Mode
	rover0 rover1 rover2 - Rover
	rover0store rover1store rover2store - Store
	waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 waypoint8 waypoint9 waypoint10 waypoint11 waypoint12 waypoint13 waypoint14 waypoint15 waypoint16 waypoint17 waypoint18 waypoint19 waypoint20 waypoint21 waypoint22 waypoint23 waypoint24 - Waypoint
	camera0 camera1 camera2 camera3 camera4 - Camera
	objective0 objective1 objective2 objective3 objective4 objective5 objective6 objective7 objective8 objective9 - Objective
	)
(:init
	(visible waypoint0 waypoint7)
	(visible waypoint7 waypoint0)
	(visible waypoint0 waypoint18)
	(visible waypoint18 waypoint0)
	(visible waypoint0 waypoint22)
	(visible waypoint22 waypoint0)
	(visible waypoint1 waypoint11)
	(visible waypoint11 waypoint1)
	(visible waypoint1 waypoint17)
	(visible waypoint17 waypoint1)
	(visible waypoint1 waypoint22)
	(visible waypoint22 waypoint1)
	(visible waypoint1 waypoint23)
	(visible waypoint23 waypoint1)
	(visible waypoint2 waypoint0)
	(visible waypoint0 waypoint2)
	(visible waypoint2 waypoint1)
	(visible waypoint1 waypoint2)
	(visible waypoint2 waypoint5)
	(visible waypoint5 waypoint2)
	(visible waypoint2 waypoint7)
	(visible waypoint7 waypoint2)
	(visible waypoint2 waypoint13)
	(visible waypoint13 waypoint2)
	(visible waypoint2 waypoint20)
	(visible waypoint20 waypoint2)
	(visible waypoint2 waypoint21)
	(visible waypoint21 waypoint2)
	(visible waypoint3 waypoint1)
	(visible waypoint1 waypoint3)
	(visible waypoint3 waypoint4)
	(visible waypoint4 waypoint3)
	(visible waypoint3 waypoint7)
	(visible waypoint7 waypoint3)
	(visible waypoint3 waypoint14)
	(visible waypoint14 waypoint3)
	(visible waypoint3 waypoint15)
	(visible waypoint15 waypoint3)
	(visible waypoint3 waypoint17)
	(visible waypoint17 waypoint3)
	(visible waypoint3 waypoint20)
	(visible waypoint20 waypoint3)
	(visible waypoint3 waypoint22)
	(visible waypoint22 waypoint3)
	(visible waypoint4 waypoint0)
	(visible waypoint0 waypoint4)
	(visible waypoint4 waypoint1)
	(visible waypoint1 waypoint4)
	(visible waypoint4 waypoint7)
	(visible waypoint7 waypoint4)
	(visible waypoint4 waypoint11)
	(visible waypoint11 waypoint4)
	(visible waypoint4 waypoint16)
	(visible waypoint16 waypoint4)
	(visible waypoint4 waypoint24)
	(visible waypoint24 waypoint4)
	(visible waypoint5 waypoint0)
	(visible waypoint0 waypoint5)
	(visible waypoint5 waypoint3)
	(visible waypoint3 waypoint5)
	(visible waypoint5 waypoint6)
	(visible waypoint6 waypoint5)
	(visible waypoint5 waypoint11)
	(visible waypoint11 waypoint5)
	(visible waypoint5 waypoint17)
	(visible waypoint17 waypoint5)
	(visible waypoint6 waypoint0)
	(visible waypoint0 waypoint6)
	(visible waypoint6 waypoint3)
	(visible waypoint3 waypoint6)
	(visible waypoint6 waypoint7)
	(visible waypoint7 waypoint6)
	(visible waypoint6 waypoint9)
	(visible waypoint9 waypoint6)
	(visible waypoint6 waypoint13)
	(visible waypoint13 waypoint6)
	(visible waypoint6 waypoint19)
	(visible waypoint19 waypoint6)
	(visible waypoint6 waypoint20)
	(visible waypoint20 waypoint6)
	(visible waypoint7 waypoint15)
	(visible waypoint15 waypoint7)
	(visible waypoint7 waypoint20)
	(visible waypoint20 waypoint7)
	(visible waypoint7 waypoint22)
	(visible waypoint22 waypoint7)
	(visible waypoint8 waypoint4)
	(visible waypoint4 waypoint8)
	(visible waypoint8 waypoint7)
	(visible waypoint7 waypoint8)
	(visible waypoint8 waypoint22)
	(visible waypoint22 waypoint8)
	(visible waypoint9 waypoint3)
	(visible waypoint3 waypoint9)
	(visible waypoint9 waypoint18)
	(visible waypoint18 waypoint9)
	(visible waypoint10 waypoint3)
	(visible waypoint3 waypoint10)
	(visible waypoint10 waypoint6)
	(visible waypoint6 waypoint10)
	(visible waypoint10 waypoint20)
	(visible waypoint20 waypoint10)
	(visible waypoint11 waypoint2)
	(visible waypoint2 waypoint11)
	(visible waypoint11 waypoint12)
	(visible waypoint12 waypoint11)
	(visible waypoint12 waypoint1)
	(visible waypoint1 waypoint12)
	(visible waypoint12 waypoint2)
	(visible waypoint2 waypoint12)
	(visible waypoint12 waypoint6)
	(visible waypoint6 waypoint12)
	(visible waypoint12 waypoint7)
	(visible waypoint7 waypoint12)
	(visible waypoint12 waypoint15)
	(visible waypoint15 waypoint12)
	(visible waypoint13 waypoint4)
	(visible waypoint4 waypoint13)
	(visible waypoint13 waypoint10)
	(visible waypoint10 waypoint13)
	(visible waypoint13 waypoint12)
	(visible waypoint12 waypoint13)
	(visible waypoint14 waypoint5)
	(visible waypoint5 waypoint14)
	(visible waypoint14 waypoint11)
	(visible waypoint11 waypoint14)
	(visible waypoint14 waypoint16)
	(visible waypoint16 waypoint14)
	(visible waypoint14 waypoint19)
	(visible waypoint19 waypoint14)
	(visible waypoint15 waypoint16)
	(visible waypoint16 waypoint15)
	(visible waypoint16 waypoint5)
	(visible waypoint5 waypoint16)
	(visible waypoint16 waypoint12)
	(visible waypoint12 waypoint16)
	(visible waypoint16 waypoint20)
	(visible waypoint20 waypoint16)
	(visible waypoint16 waypoint23)
	(visible waypoint23 waypoint16)
	(visible waypoint16 waypoint24)
	(visible waypoint24 waypoint16)
	(visible waypoint17 waypoint13)
	(visible waypoint13 waypoint17)
	(visible waypoint17 waypoint16)
	(visible waypoint16 waypoint17)
	(visible waypoint17 waypoint21)
	(visible waypoint21 waypoint17)
	(visible waypoint18 waypoint4)
	(visible waypoint4 waypoint18)
	(visible waypoint18 waypoint6)
	(visible waypoint6 waypoint18)
	(visible waypoint18 waypoint8)
	(visible waypoint8 waypoint18)
	(visible waypoint18 waypoint15)
	(visible waypoint15 waypoint18)
	(visible waypoint18 waypoint20)
	(visible waypoint20 waypoint18)
	(visible waypoint19 waypoint7)
	(visible waypoint7 waypoint19)
	(visible waypoint19 waypoint10)
	(visible waypoint10 waypoint19)
	(visible waypoint19 waypoint20)
	(visible waypoint20 waypoint19)
	(visible waypoint20 waypoint0)
	(visible waypoint0 waypoint20)
	(visible waypoint20 waypoint9)
	(visible waypoint9 waypoint20)
	(visible waypoint20 waypoint13)
	(visible waypoint13 waypoint20)
	(visible waypoint21 waypoint8)
	(visible waypoint8 waypoint21)
	(visible waypoint21 waypoint19)
	(visible waypoint19 waypoint21)
	(visible waypoint22 waypoint4)
	(visible waypoint4 waypoint22)
	(visible waypoint22 waypoint5)
	(visible waypoint5 waypoint22)
	(visible waypoint22 waypoint11)
	(visible waypoint11 waypoint22)
	(visible waypoint22 waypoint12)
	(visible waypoint12 waypoint22)
	(visible waypoint22 waypoint20)
	(visible waypoint20 waypoint22)
	(visible waypoint23 waypoint5)
	(visible waypoint5 waypoint23)
	(visible waypoint23 waypoint6)
	(visible waypoint6 waypoint23)
	(visible waypoint23 waypoint10)
	(visible waypoint10 waypoint23)
	(visible waypoint23 waypoint11)
	(visible waypoint11 waypoint23)
	(visible waypoint23 waypoint15)
	(visible waypoint15 waypoint23)
	(visible waypoint23 waypoint18)
	(visible waypoint18 waypoint23)
	(visible waypoint24 waypoint18)
	(visible waypoint18 waypoint24)
	(at_soil_sample waypoint1)
	(at_rock_sample waypoint1)
	(at_soil_sample waypoint4)
	(at_soil_sample waypoint5)
	(at_rock_sample waypoint7)
	(at_soil_sample waypoint8)
	(at_rock_sample waypoint8)
	(at_rock_sample waypoint9)
	(at_soil_sample waypoint10)
	(at_soil_sample waypoint12)
	(at_rock_sample waypoint13)
	(at_soil_sample waypoint14)
	(at_rock_sample waypoint14)
	(at_soil_sample waypoint15)
	(at_rock_sample waypoint15)
	(at_rock_sample waypoint16)
	(at_soil_sample waypoint18)
	(at_rock_sample waypoint18)
	(at_soil_sample waypoint19)
	(at_rock_sample waypoint19)
	(at_soil_sample waypoint21)
	(at_rock_sample waypoint21)
	(at_soil_sample waypoint22)
	(at_rock_sample waypoint22)
	(at_rock_sample waypoint23)
	(at_lander general waypoint23)
	(channel_free general)
	(at rover0 waypoint17)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 waypoint17 waypoint1)
	(can_traverse rover0 waypoint1 waypoint17)
	(can_traverse rover0 waypoint17 waypoint3)
	(can_traverse rover0 waypoint3 waypoint17)
	(can_traverse rover0 waypoint17 waypoint5)
	(can_traverse rover0 waypoint5 waypoint17)
	(can_traverse rover0 waypoint17 waypoint13)
	(can_traverse rover0 waypoint13 waypoint17)
	(can_traverse rover0 waypoint17 waypoint21)
	(can_traverse rover0 waypoint21 waypoint17)
	(can_traverse rover0 waypoint1 waypoint4)
	(can_traverse rover0 waypoint4 waypoint1)
	(can_traverse rover0 waypoint1 waypoint11)
	(can_traverse rover0 waypoint11 waypoint1)
	(can_traverse rover0 waypoint1 waypoint12)
	(can_traverse rover0 waypoint12 waypoint1)
	(can_traverse rover0 waypoint1 waypoint22)
	(can_traverse rover0 waypoint22 waypoint1)
	(can_traverse rover0 waypoint3 waypoint6)
	(can_traverse rover0 waypoint6 waypoint3)
	(can_traverse rover0 waypoint3 waypoint7)
	(can_traverse rover0 waypoint7 waypoint3)
	(can_traverse rover0 waypoint3 waypoint9)
	(can_traverse rover0 waypoint9 waypoint3)
	(can_traverse rover0 waypoint3 waypoint10)
	(can_traverse rover0 waypoint10 waypoint3)
	(can_traverse rover0 waypoint3 waypoint14)
	(can_traverse rover0 waypoint14 waypoint3)
	(can_traverse rover0 waypoint3 waypoint15)
	(can_traverse rover0 waypoint15 waypoint3)
	(can_traverse rover0 waypoint5 waypoint0)
	(can_traverse rover0 waypoint0 waypoint5)
	(can_traverse rover0 waypoint5 waypoint2)
	(can_traverse rover0 waypoint2 waypoint5)
	(can_traverse rover0 waypoint5 waypoint23)
	(can_traverse rover0 waypoint23 waypoint5)
	(can_traverse rover0 waypoint13 waypoint20)
	(can_traverse rover0 waypoint20 waypoint13)
	(can_traverse rover0 waypoint21 waypoint8)
	(can_traverse rover0 waypoint8 waypoint21)
	(can_traverse rover0 waypoint21 waypoint19)
	(can_traverse rover0 waypoint19 waypoint21)
	(can_traverse rover0 waypoint4 waypoint16)
	(can_traverse rover0 waypoint16 waypoint4)
	(can_traverse rover0 waypoint4 waypoint18)
	(can_traverse rover0 waypoint18 waypoint4)
	(can_traverse rover0 waypoint4 waypoint24)
	(can_traverse rover0 waypoint24 waypoint4)
	(at rover1 waypoint14)
	(available rover1)
	(store_of rover1store rover1)
	(empty rover1store)
	(equipped_for_rock_analysis rover1)
	(equipped_for_imaging rover1)
	(can_traverse rover1 waypoint14 waypoint3)
	(can_traverse rover1 waypoint3 waypoint14)
	(can_traverse rover1 waypoint14 waypoint11)
	(can_traverse rover1 waypoint11 waypoint14)
	(can_traverse rover1 waypoint14 waypoint19)
	(can_traverse rover1 waypoint19 waypoint14)
	(can_traverse rover1 waypoint3 waypoint1)
	(can_traverse rover1 waypoint1 waypoint3)
	(can_traverse rover1 waypoint3 waypoint6)
	(can_traverse rover1 waypoint6 waypoint3)
	(can_traverse rover1 waypoint3 waypoint15)
	(can_traverse rover1 waypoint15 waypoint3)
	(can_traverse rover1 waypoint3 waypoint20)
	(can_traverse rover1 waypoint20 waypoint3)
	(can_traverse rover1 waypoint3 waypoint22)
	(can_traverse rover1 waypoint22 waypoint3)
	(can_traverse rover1 waypoint11 waypoint12)
	(can_traverse rover1 waypoint12 waypoint11)
	(can_traverse rover1 waypoint11 waypoint23)
	(can_traverse rover1 waypoint23 waypoint11)
	(can_traverse rover1 waypoint19 waypoint7)
	(can_traverse rover1 waypoint7 waypoint19)
	(can_traverse rover1 waypoint19 waypoint10)
	(can_traverse rover1 waypoint10 waypoint19)
	(can_traverse rover1 waypoint6 waypoint0)
	(can_traverse rover1 waypoint0 waypoint6)
	(can_traverse rover1 waypoint6 waypoint5)
	(can_traverse rover1 waypoint5 waypoint6)
	(can_traverse rover1 waypoint6 waypoint9)
	(can_traverse rover1 waypoint9 waypoint6)
	(can_traverse rover1 waypoint6 waypoint13)
	(can_traverse rover1 waypoint13 waypoint6)
	(can_traverse rover1 waypoint6 waypoint18)
	(can_traverse rover1 waypoint18 waypoint6)
	(can_traverse rover1 waypoint20 waypoint16)
	(can_traverse rover1 waypoint16 waypoint20)
	(can_traverse rover1 waypoint22 waypoint4)
	(can_traverse rover1 waypoint4 waypoint22)
	(can_traverse rover1 waypoint22 waypoint8)
	(can_traverse rover1 waypoint8 waypoint22)
	(can_traverse rover1 waypoint12 waypoint2)
	(can_traverse rover1 waypoint2 waypoint12)
	(at rover2 waypoint15)
	(available rover2)
	(store_of rover2store rover2)
	(empty rover2store)
	(equipped_for_rock_analysis rover2)
	(equipped_for_imaging rover2)
	(can_traverse rover2 waypoint15 waypoint3)
	(can_traverse rover2 waypoint3 waypoint15)
	(can_traverse rover2 waypoint15 waypoint12)
	(can_traverse rover2 waypoint12 waypoint15)
	(can_traverse rover2 waypoint15 waypoint16)
	(can_traverse rover2 waypoint16 waypoint15)
	(can_traverse rover2 waypoint15 waypoint18)
	(can_traverse rover2 waypoint18 waypoint15)
	(can_traverse rover2 waypoint3 waypoint4)
	(can_traverse rover2 waypoint4 waypoint3)
	(can_traverse rover2 waypoint3 waypoint5)
	(can_traverse rover2 waypoint5 waypoint3)
	(can_traverse rover2 waypoint3 waypoint6)
	(can_traverse rover2 waypoint6 waypoint3)
	(can_traverse rover2 waypoint3 waypoint7)
	(can_traverse rover2 waypoint7 waypoint3)
	(can_traverse rover2 waypoint3 waypoint9)
	(can_traverse rover2 waypoint9 waypoint3)
	(can_traverse rover2 waypoint3 waypoint10)
	(can_traverse rover2 waypoint10 waypoint3)
	(can_traverse rover2 waypoint3 waypoint14)
	(can_traverse rover2 waypoint14 waypoint3)
	(can_traverse rover2 waypoint3 waypoint17)
	(can_traverse rover2 waypoint17 waypoint3)
	(can_traverse rover2 waypoint3 waypoint20)
	(can_traverse rover2 waypoint20 waypoint3)
	(can_traverse rover2 waypoint12 waypoint1)
	(can_traverse rover2 waypoint1 waypoint12)
	(can_traverse rover2 waypoint12 waypoint2)
	(can_traverse rover2 waypoint2 waypoint12)
	(can_traverse rover2 waypoint12 waypoint13)
	(can_traverse rover2 waypoint13 waypoint12)
	(can_traverse rover2 waypoint12 waypoint22)
	(can_traverse rover2 waypoint22 waypoint12)
	(can_traverse rover2 waypoint16 waypoint24)
	(can_traverse rover2 waypoint24 waypoint16)
	(can_traverse rover2 waypoint18 waypoint23)
	(can_traverse rover2 waypoint23 waypoint18)
	(can_traverse rover2 waypoint4 waypoint0)
	(can_traverse rover2 waypoint0 waypoint4)
	(can_traverse rover2 waypoint4 waypoint8)
	(can_traverse rover2 waypoint8 waypoint4)
	(can_traverse rover2 waypoint4 waypoint11)
	(can_traverse rover2 waypoint11 waypoint4)
	(can_traverse rover2 waypoint6 waypoint19)
	(can_traverse rover2 waypoint19 waypoint6)
	(can_traverse rover2 waypoint17 waypoint21)
	(can_traverse rover2 waypoint21 waypoint17)
	(on_board camera0 rover1)
	(calibration_target camera0 objective7)
	(supports camera0 high_res)
	(on_board camera1 rover1)
	(calibration_target camera1 objective3)
	(supports camera1 high_res)
	(on_board camera2 rover0)
	(calibration_target camera2 objective2)
	(supports camera2 colour)
	(supports camera2 low_res)
	(on_board camera3 rover0)
	(calibration_target camera3 objective7)
	(supports camera3 high_res)
	(supports camera3 low_res)
	(on_board camera4 rover2)
	(calibration_target camera4 objective3)
	(supports camera4 colour)
	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective0 waypoint2)
	(visible_from objective1 waypoint0)
	(visible_from objective1 waypoint1)
	(visible_from objective1 waypoint2)
	(visible_from objective1 waypoint3)
	(visible_from objective1 waypoint4)
	(visible_from objective1 waypoint5)
	(visible_from objective1 waypoint6)
	(visible_from objective1 waypoint7)
	(visible_from objective1 waypoint8)
	(visible_from objective1 waypoint9)
	(visible_from objective1 waypoint10)
	(visible_from objective1 waypoint11)
	(visible_from objective1 waypoint12)
	(visible_from objective1 waypoint13)
	(visible_from objective1 waypoint14)
	(visible_from objective1 waypoint15)
	(visible_from objective1 waypoint16)
	(visible_from objective2 waypoint0)
	(visible_from objective2 waypoint1)
	(visible_from objective2 waypoint2)
	(visible_from objective2 waypoint3)
	(visible_from objective2 waypoint4)
	(visible_from objective2 waypoint5)
	(visible_from objective2 waypoint6)
	(visible_from objective2 waypoint7)
	(visible_from objective2 waypoint8)
	(visible_from objective2 waypoint9)
	(visible_from objective2 waypoint10)
	(visible_from objective2 waypoint11)
	(visible_from objective2 waypoint12)
	(visible_from objective2 waypoint13)
	(visible_from objective2 waypoint14)
	(visible_from objective2 waypoint15)
	(visible_from objective2 waypoint16)
	(visible_from objective2 waypoint17)
	(visible_from objective2 waypoint18)
	(visible_from objective3 waypoint0)
	(visible_from objective3 waypoint1)
	(visible_from objective3 waypoint2)
	(visible_from objective3 waypoint3)
	(visible_from objective3 waypoint4)
	(visible_from objective4 waypoint0)
	(visible_from objective4 waypoint1)
	(visible_from objective4 waypoint2)
	(visible_from objective4 waypoint3)
	(visible_from objective4 waypoint4)
	(visible_from objective4 waypoint5)
	(visible_from objective4 waypoint6)
	(visible_from objective4 waypoint7)
	(visible_from objective4 waypoint8)
	(visible_from objective4 waypoint9)
	(visible_from objective4 waypoint10)
	(visible_from objective4 waypoint11)
	(visible_from objective4 waypoint12)
	(visible_from objective4 waypoint13)
	(visible_from objective5 waypoint0)
	(visible_from objective5 waypoint1)
	(visible_from objective5 waypoint2)
	(visible_from objective5 waypoint3)
	(visible_from objective5 waypoint4)
	(visible_from objective5 waypoint5)
	(visible_from objective5 waypoint6)
	(visible_from objective5 waypoint7)
	(visible_from objective5 waypoint8)
	(visible_from objective5 waypoint9)
	(visible_from objective5 waypoint10)
	(visible_from objective5 waypoint11)
	(visible_from objective5 waypoint12)
	(visible_from objective5 waypoint13)
	(visible_from objective5 waypoint14)
	(visible_from objective5 waypoint15)
	(visible_from objective5 waypoint16)
	(visible_from objective5 waypoint17)
	(visible_from objective5 waypoint18)
	(visible_from objective6 waypoint0)
	(visible_from objective6 waypoint1)
	(visible_from objective6 waypoint2)
	(visible_from objective6 waypoint3)
	(visible_from objective6 waypoint4)
	(visible_from objective6 waypoint5)
	(visible_from objective6 waypoint6)
	(visible_from objective6 waypoint7)
	(visible_from objective6 waypoint8)
	(visible_from objective6 waypoint9)
	(visible_from objective6 waypoint10)
	(visible_from objective6 waypoint11)
	(visible_from objective6 waypoint12)
	(visible_from objective6 waypoint13)
	(visible_from objective6 waypoint14)
	(visible_from objective6 waypoint15)
	(visible_from objective6 waypoint16)
	(visible_from objective6 waypoint17)
	(visible_from objective6 waypoint18)
	(visible_from objective6 waypoint19)
	(visible_from objective7 waypoint0)
	(visible_from objective7 waypoint1)
	(visible_from objective7 waypoint2)
	(visible_from objective7 waypoint3)
	(visible_from objective7 waypoint4)
	(visible_from objective7 waypoint5)
	(visible_from objective7 waypoint6)
	(visible_from objective7 waypoint7)
	(visible_from objective7 waypoint8)
	(visible_from objective7 waypoint9)
	(visible_from objective7 waypoint10)
	(visible_from objective7 waypoint11)
	(visible_from objective7 waypoint12)
	(visible_from objective7 waypoint13)
	(visible_from objective7 waypoint14)
	(visible_from objective7 waypoint15)
	(visible_from objective7 waypoint16)
	(visible_from objective7 waypoint17)
	(visible_from objective7 waypoint18)
	(visible_from objective7 waypoint19)
	(visible_from objective7 waypoint20)
	(visible_from objective7 waypoint21)
	(visible_from objective7 waypoint22)
	(visible_from objective8 waypoint0)
	(visible_from objective8 waypoint1)
	(visible_from objective8 waypoint2)
	(visible_from objective8 waypoint3)
	(visible_from objective8 waypoint4)
	(visible_from objective8 waypoint5)
	(visible_from objective8 waypoint6)
	(visible_from objective8 waypoint7)
	(visible_from objective8 waypoint8)
	(visible_from objective8 waypoint9)
	(visible_from objective8 waypoint10)
	(visible_from objective8 waypoint11)
	(visible_from objective8 waypoint12)
	(visible_from objective8 waypoint13)
	(visible_from objective8 waypoint14)
	(visible_from objective8 waypoint15)
	(visible_from objective8 waypoint16)
	(visible_from objective8 waypoint17)
	(visible_from objective8 waypoint18)
	(visible_from objective9 waypoint0)
	(visible_from objective9 waypoint1)
	(visible_from objective9 waypoint2)
	(visible_from objective9 waypoint3)
)

(:goal (and
(communicated_rock_data waypoint23)
(communicated_rock_data waypoint14)
(communicated_image_data objective4 colour)
(communicated_image_data objective1 colour)
	)
)
)
