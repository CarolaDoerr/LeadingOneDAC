(define (problem roverprob) (:domain Rover)
(:objects
	general - Lander
	colour high_res low_res - Mode
	rover0 rover1 rover2 - Rover
	rover0store rover1store rover2store - Store
	waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 waypoint8 waypoint9 waypoint10 waypoint11 waypoint12 waypoint13 waypoint14 waypoint15 waypoint16 waypoint17 waypoint18 waypoint19 waypoint20 waypoint21 waypoint22 waypoint23 waypoint24 - Waypoint
	camera0 camera1 camera2 camera3 camera4 camera5 - Camera
	objective0 objective1 objective2 objective3 objective4 objective5 objective6 objective7 - Objective
	)
(:init
	(visible waypoint0 waypoint1)
	(visible waypoint1 waypoint0)
	(visible waypoint0 waypoint2)
	(visible waypoint2 waypoint0)
	(visible waypoint0 waypoint5)
	(visible waypoint5 waypoint0)
	(visible waypoint0 waypoint9)
	(visible waypoint9 waypoint0)
	(visible waypoint0 waypoint10)
	(visible waypoint10 waypoint0)
	(visible waypoint0 waypoint18)
	(visible waypoint18 waypoint0)
	(visible waypoint0 waypoint19)
	(visible waypoint19 waypoint0)
	(visible waypoint0 waypoint22)
	(visible waypoint22 waypoint0)
	(visible waypoint0 waypoint23)
	(visible waypoint23 waypoint0)
	(visible waypoint1 waypoint9)
	(visible waypoint9 waypoint1)
	(visible waypoint1 waypoint19)
	(visible waypoint19 waypoint1)
	(visible waypoint1 waypoint20)
	(visible waypoint20 waypoint1)
	(visible waypoint1 waypoint22)
	(visible waypoint22 waypoint1)
	(visible waypoint2 waypoint7)
	(visible waypoint7 waypoint2)
	(visible waypoint2 waypoint14)
	(visible waypoint14 waypoint2)
	(visible waypoint3 waypoint0)
	(visible waypoint0 waypoint3)
	(visible waypoint3 waypoint5)
	(visible waypoint5 waypoint3)
	(visible waypoint3 waypoint6)
	(visible waypoint6 waypoint3)
	(visible waypoint3 waypoint8)
	(visible waypoint8 waypoint3)
	(visible waypoint3 waypoint12)
	(visible waypoint12 waypoint3)
	(visible waypoint3 waypoint16)
	(visible waypoint16 waypoint3)
	(visible waypoint3 waypoint18)
	(visible waypoint18 waypoint3)
	(visible waypoint3 waypoint22)
	(visible waypoint22 waypoint3)
	(visible waypoint3 waypoint24)
	(visible waypoint24 waypoint3)
	(visible waypoint4 waypoint5)
	(visible waypoint5 waypoint4)
	(visible waypoint4 waypoint11)
	(visible waypoint11 waypoint4)
	(visible waypoint4 waypoint18)
	(visible waypoint18 waypoint4)
	(visible waypoint5 waypoint12)
	(visible waypoint12 waypoint5)
	(visible waypoint5 waypoint14)
	(visible waypoint14 waypoint5)
	(visible waypoint5 waypoint24)
	(visible waypoint24 waypoint5)
	(visible waypoint6 waypoint4)
	(visible waypoint4 waypoint6)
	(visible waypoint6 waypoint12)
	(visible waypoint12 waypoint6)
	(visible waypoint6 waypoint15)
	(visible waypoint15 waypoint6)
	(visible waypoint6 waypoint18)
	(visible waypoint18 waypoint6)
	(visible waypoint7 waypoint0)
	(visible waypoint0 waypoint7)
	(visible waypoint7 waypoint3)
	(visible waypoint3 waypoint7)
	(visible waypoint7 waypoint6)
	(visible waypoint6 waypoint7)
	(visible waypoint7 waypoint22)
	(visible waypoint22 waypoint7)
	(visible waypoint8 waypoint15)
	(visible waypoint15 waypoint8)
	(visible waypoint8 waypoint16)
	(visible waypoint16 waypoint8)
	(visible waypoint8 waypoint18)
	(visible waypoint18 waypoint8)
	(visible waypoint9 waypoint19)
	(visible waypoint19 waypoint9)
	(visible waypoint9 waypoint22)
	(visible waypoint22 waypoint9)
	(visible waypoint9 waypoint24)
	(visible waypoint24 waypoint9)
	(visible waypoint10 waypoint5)
	(visible waypoint5 waypoint10)
	(visible waypoint10 waypoint6)
	(visible waypoint6 waypoint10)
	(visible waypoint11 waypoint7)
	(visible waypoint7 waypoint11)
	(visible waypoint11 waypoint17)
	(visible waypoint17 waypoint11)
	(visible waypoint11 waypoint23)
	(visible waypoint23 waypoint11)
	(visible waypoint11 waypoint24)
	(visible waypoint24 waypoint11)
	(visible waypoint12 waypoint2)
	(visible waypoint2 waypoint12)
	(visible waypoint12 waypoint4)
	(visible waypoint4 waypoint12)
	(visible waypoint12 waypoint10)
	(visible waypoint10 waypoint12)
	(visible waypoint12 waypoint11)
	(visible waypoint11 waypoint12)
	(visible waypoint12 waypoint17)
	(visible waypoint17 waypoint12)
	(visible waypoint12 waypoint18)
	(visible waypoint18 waypoint12)
	(visible waypoint13 waypoint4)
	(visible waypoint4 waypoint13)
	(visible waypoint13 waypoint7)
	(visible waypoint7 waypoint13)
	(visible waypoint13 waypoint18)
	(visible waypoint18 waypoint13)
	(visible waypoint14 waypoint24)
	(visible waypoint24 waypoint14)
	(visible waypoint15 waypoint2)
	(visible waypoint2 waypoint15)
	(visible waypoint15 waypoint3)
	(visible waypoint3 waypoint15)
	(visible waypoint15 waypoint5)
	(visible waypoint5 waypoint15)
	(visible waypoint15 waypoint11)
	(visible waypoint11 waypoint15)
	(visible waypoint15 waypoint13)
	(visible waypoint13 waypoint15)
	(visible waypoint15 waypoint14)
	(visible waypoint14 waypoint15)
	(visible waypoint16 waypoint4)
	(visible waypoint4 waypoint16)
	(visible waypoint16 waypoint12)
	(visible waypoint12 waypoint16)
	(visible waypoint17 waypoint2)
	(visible waypoint2 waypoint17)
	(visible waypoint17 waypoint20)
	(visible waypoint20 waypoint17)
	(visible waypoint18 waypoint10)
	(visible waypoint10 waypoint18)
	(visible waypoint18 waypoint14)
	(visible waypoint14 waypoint18)
	(visible waypoint18 waypoint21)
	(visible waypoint21 waypoint18)
	(visible waypoint19 waypoint8)
	(visible waypoint8 waypoint19)
	(visible waypoint19 waypoint15)
	(visible waypoint15 waypoint19)
	(visible waypoint19 waypoint18)
	(visible waypoint18 waypoint19)
	(visible waypoint20 waypoint6)
	(visible waypoint6 waypoint20)
	(visible waypoint20 waypoint11)
	(visible waypoint11 waypoint20)
	(visible waypoint20 waypoint12)
	(visible waypoint12 waypoint20)
	(visible waypoint20 waypoint18)
	(visible waypoint18 waypoint20)
	(visible waypoint21 waypoint3)
	(visible waypoint3 waypoint21)
	(visible waypoint21 waypoint5)
	(visible waypoint5 waypoint21)
	(visible waypoint21 waypoint7)
	(visible waypoint7 waypoint21)
	(visible waypoint21 waypoint15)
	(visible waypoint15 waypoint21)
	(visible waypoint22 waypoint5)
	(visible waypoint5 waypoint22)
	(visible waypoint22 waypoint12)
	(visible waypoint12 waypoint22)
	(visible waypoint22 waypoint16)
	(visible waypoint16 waypoint22)
	(visible waypoint22 waypoint17)
	(visible waypoint17 waypoint22)
	(visible waypoint22 waypoint20)
	(visible waypoint20 waypoint22)
	(visible waypoint22 waypoint21)
	(visible waypoint21 waypoint22)
	(visible waypoint23 waypoint6)
	(visible waypoint6 waypoint23)
	(visible waypoint23 waypoint7)
	(visible waypoint7 waypoint23)
	(visible waypoint23 waypoint8)
	(visible waypoint8 waypoint23)
	(visible waypoint23 waypoint13)
	(visible waypoint13 waypoint23)
	(visible waypoint23 waypoint19)
	(visible waypoint19 waypoint23)
	(visible waypoint23 waypoint22)
	(visible waypoint22 waypoint23)
	(visible waypoint24 waypoint1)
	(visible waypoint1 waypoint24)
	(visible waypoint24 waypoint6)
	(visible waypoint6 waypoint24)
	(visible waypoint24 waypoint16)
	(visible waypoint16 waypoint24)
	(visible waypoint24 waypoint18)
	(visible waypoint18 waypoint24)
	(visible waypoint24 waypoint19)
	(visible waypoint19 waypoint24)
	(visible waypoint24 waypoint20)
	(visible waypoint20 waypoint24)
	(at_soil_sample waypoint0)
	(at_rock_sample waypoint0)
	(at_soil_sample waypoint1)
	(at_rock_sample waypoint1)
	(at_soil_sample waypoint2)
	(at_soil_sample waypoint3)
	(at_soil_sample waypoint5)
	(at_soil_sample waypoint6)
	(at_rock_sample waypoint6)
	(at_rock_sample waypoint7)
	(at_soil_sample waypoint8)
	(at_rock_sample waypoint8)
	(at_soil_sample waypoint9)
	(at_soil_sample waypoint10)
	(at_soil_sample waypoint11)
	(at_soil_sample waypoint12)
	(at_rock_sample waypoint13)
	(at_soil_sample waypoint15)
	(at_rock_sample waypoint17)
	(at_soil_sample waypoint18)
	(at_soil_sample waypoint20)
	(at_rock_sample waypoint20)
	(at_soil_sample waypoint22)
	(at_soil_sample waypoint23)
	(at_lander general waypoint12)
	(channel_free general)
	(at rover0 waypoint6)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 waypoint6 waypoint3)
	(can_traverse rover0 waypoint3 waypoint6)
	(can_traverse rover0 waypoint6 waypoint4)
	(can_traverse rover0 waypoint4 waypoint6)
	(can_traverse rover0 waypoint6 waypoint7)
	(can_traverse rover0 waypoint7 waypoint6)
	(can_traverse rover0 waypoint6 waypoint12)
	(can_traverse rover0 waypoint12 waypoint6)
	(can_traverse rover0 waypoint6 waypoint15)
	(can_traverse rover0 waypoint15 waypoint6)
	(can_traverse rover0 waypoint6 waypoint18)
	(can_traverse rover0 waypoint18 waypoint6)
	(can_traverse rover0 waypoint6 waypoint20)
	(can_traverse rover0 waypoint20 waypoint6)
	(can_traverse rover0 waypoint6 waypoint23)
	(can_traverse rover0 waypoint23 waypoint6)
	(can_traverse rover0 waypoint3 waypoint0)
	(can_traverse rover0 waypoint0 waypoint3)
	(can_traverse rover0 waypoint3 waypoint5)
	(can_traverse rover0 waypoint5 waypoint3)
	(can_traverse rover0 waypoint3 waypoint8)
	(can_traverse rover0 waypoint8 waypoint3)
	(can_traverse rover0 waypoint3 waypoint16)
	(can_traverse rover0 waypoint16 waypoint3)
	(can_traverse rover0 waypoint3 waypoint21)
	(can_traverse rover0 waypoint21 waypoint3)
	(can_traverse rover0 waypoint3 waypoint22)
	(can_traverse rover0 waypoint22 waypoint3)
	(can_traverse rover0 waypoint3 waypoint24)
	(can_traverse rover0 waypoint24 waypoint3)
	(can_traverse rover0 waypoint7 waypoint2)
	(can_traverse rover0 waypoint2 waypoint7)
	(can_traverse rover0 waypoint12 waypoint10)
	(can_traverse rover0 waypoint10 waypoint12)
	(can_traverse rover0 waypoint12 waypoint11)
	(can_traverse rover0 waypoint11 waypoint12)
	(can_traverse rover0 waypoint12 waypoint17)
	(can_traverse rover0 waypoint17 waypoint12)
	(can_traverse rover0 waypoint15 waypoint13)
	(can_traverse rover0 waypoint13 waypoint15)
	(can_traverse rover0 waypoint15 waypoint14)
	(can_traverse rover0 waypoint14 waypoint15)
	(can_traverse rover0 waypoint18 waypoint19)
	(can_traverse rover0 waypoint19 waypoint18)
	(can_traverse rover0 waypoint20 waypoint1)
	(can_traverse rover0 waypoint1 waypoint20)
	(can_traverse rover0 waypoint0 waypoint9)
	(can_traverse rover0 waypoint9 waypoint0)
	(at rover1 waypoint2)
	(available rover1)
	(store_of rover1store rover1)
	(empty rover1store)
	(equipped_for_rock_analysis rover1)
	(equipped_for_imaging rover1)
	(can_traverse rover1 waypoint2 waypoint0)
	(can_traverse rover1 waypoint0 waypoint2)
	(can_traverse rover1 waypoint2 waypoint7)
	(can_traverse rover1 waypoint7 waypoint2)
	(can_traverse rover1 waypoint2 waypoint14)
	(can_traverse rover1 waypoint14 waypoint2)
	(can_traverse rover1 waypoint0 waypoint3)
	(can_traverse rover1 waypoint3 waypoint0)
	(can_traverse rover1 waypoint0 waypoint5)
	(can_traverse rover1 waypoint5 waypoint0)
	(can_traverse rover1 waypoint0 waypoint9)
	(can_traverse rover1 waypoint9 waypoint0)
	(can_traverse rover1 waypoint0 waypoint18)
	(can_traverse rover1 waypoint18 waypoint0)
	(can_traverse rover1 waypoint0 waypoint19)
	(can_traverse rover1 waypoint19 waypoint0)
	(can_traverse rover1 waypoint0 waypoint23)
	(can_traverse rover1 waypoint23 waypoint0)
	(can_traverse rover1 waypoint7 waypoint6)
	(can_traverse rover1 waypoint6 waypoint7)
	(can_traverse rover1 waypoint7 waypoint13)
	(can_traverse rover1 waypoint13 waypoint7)
	(can_traverse rover1 waypoint7 waypoint22)
	(can_traverse rover1 waypoint22 waypoint7)
	(can_traverse rover1 waypoint14 waypoint15)
	(can_traverse rover1 waypoint15 waypoint14)
	(can_traverse rover1 waypoint14 waypoint24)
	(can_traverse rover1 waypoint24 waypoint14)
	(can_traverse rover1 waypoint3 waypoint8)
	(can_traverse rover1 waypoint8 waypoint3)
	(can_traverse rover1 waypoint3 waypoint12)
	(can_traverse rover1 waypoint12 waypoint3)
	(can_traverse rover1 waypoint3 waypoint16)
	(can_traverse rover1 waypoint16 waypoint3)
	(can_traverse rover1 waypoint3 waypoint21)
	(can_traverse rover1 waypoint21 waypoint3)
	(can_traverse rover1 waypoint5 waypoint4)
	(can_traverse rover1 waypoint4 waypoint5)
	(can_traverse rover1 waypoint5 waypoint10)
	(can_traverse rover1 waypoint10 waypoint5)
	(can_traverse rover1 waypoint9 waypoint1)
	(can_traverse rover1 waypoint1 waypoint9)
	(can_traverse rover1 waypoint6 waypoint20)
	(can_traverse rover1 waypoint20 waypoint6)
	(can_traverse rover1 waypoint22 waypoint17)
	(can_traverse rover1 waypoint17 waypoint22)
	(can_traverse rover1 waypoint15 waypoint11)
	(can_traverse rover1 waypoint11 waypoint15)
	(at rover2 waypoint23)
	(available rover2)
	(store_of rover2store rover2)
	(empty rover2store)
	(equipped_for_imaging rover2)
	(can_traverse rover2 waypoint23 waypoint0)
	(can_traverse rover2 waypoint0 waypoint23)
	(can_traverse rover2 waypoint23 waypoint6)
	(can_traverse rover2 waypoint6 waypoint23)
	(can_traverse rover2 waypoint23 waypoint7)
	(can_traverse rover2 waypoint7 waypoint23)
	(can_traverse rover2 waypoint23 waypoint8)
	(can_traverse rover2 waypoint8 waypoint23)
	(can_traverse rover2 waypoint23 waypoint11)
	(can_traverse rover2 waypoint11 waypoint23)
	(can_traverse rover2 waypoint23 waypoint13)
	(can_traverse rover2 waypoint13 waypoint23)
	(can_traverse rover2 waypoint23 waypoint19)
	(can_traverse rover2 waypoint19 waypoint23)
	(can_traverse rover2 waypoint23 waypoint22)
	(can_traverse rover2 waypoint22 waypoint23)
	(can_traverse rover2 waypoint0 waypoint1)
	(can_traverse rover2 waypoint1 waypoint0)
	(can_traverse rover2 waypoint0 waypoint9)
	(can_traverse rover2 waypoint9 waypoint0)
	(can_traverse rover2 waypoint0 waypoint10)
	(can_traverse rover2 waypoint10 waypoint0)
	(can_traverse rover2 waypoint0 waypoint18)
	(can_traverse rover2 waypoint18 waypoint0)
	(can_traverse rover2 waypoint6 waypoint12)
	(can_traverse rover2 waypoint12 waypoint6)
	(can_traverse rover2 waypoint6 waypoint15)
	(can_traverse rover2 waypoint15 waypoint6)
	(can_traverse rover2 waypoint6 waypoint24)
	(can_traverse rover2 waypoint24 waypoint6)
	(can_traverse rover2 waypoint7 waypoint3)
	(can_traverse rover2 waypoint3 waypoint7)
	(can_traverse rover2 waypoint7 waypoint21)
	(can_traverse rover2 waypoint21 waypoint7)
	(can_traverse rover2 waypoint8 waypoint16)
	(can_traverse rover2 waypoint16 waypoint8)
	(can_traverse rover2 waypoint11 waypoint20)
	(can_traverse rover2 waypoint20 waypoint11)
	(can_traverse rover2 waypoint22 waypoint5)
	(can_traverse rover2 waypoint5 waypoint22)
	(can_traverse rover2 waypoint22 waypoint17)
	(can_traverse rover2 waypoint17 waypoint22)
	(can_traverse rover2 waypoint18 waypoint4)
	(can_traverse rover2 waypoint4 waypoint18)
	(can_traverse rover2 waypoint12 waypoint2)
	(can_traverse rover2 waypoint2 waypoint12)
	(can_traverse rover2 waypoint15 waypoint14)
	(can_traverse rover2 waypoint14 waypoint15)
	(on_board camera0 rover0)
	(calibration_target camera0 objective6)
	(supports camera0 high_res)
	(supports camera0 low_res)
	(on_board camera1 rover0)
	(calibration_target camera1 objective1)
	(supports camera1 colour)
	(supports camera1 low_res)
	(on_board camera2 rover0)
	(calibration_target camera2 objective2)
	(supports camera2 high_res)
	(supports camera2 low_res)
	(on_board camera3 rover0)
	(calibration_target camera3 objective7)
	(supports camera3 colour)
	(supports camera3 low_res)
	(on_board camera4 rover1)
	(calibration_target camera4 objective1)
	(supports camera4 colour)
	(supports camera4 high_res)
	(supports camera4 low_res)
	(on_board camera5 rover2)
	(calibration_target camera5 objective6)
	(supports camera5 colour)
	(supports camera5 low_res)
	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective0 waypoint2)
	(visible_from objective0 waypoint3)
	(visible_from objective0 waypoint4)
	(visible_from objective0 waypoint5)
	(visible_from objective0 waypoint6)
	(visible_from objective0 waypoint7)
	(visible_from objective0 waypoint8)
	(visible_from objective0 waypoint9)
	(visible_from objective0 waypoint10)
	(visible_from objective0 waypoint11)
	(visible_from objective0 waypoint12)
	(visible_from objective0 waypoint13)
	(visible_from objective0 waypoint14)
	(visible_from objective0 waypoint15)
	(visible_from objective0 waypoint16)
	(visible_from objective0 waypoint17)
	(visible_from objective0 waypoint18)
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
	(visible_from objective1 waypoint17)
	(visible_from objective1 waypoint18)
	(visible_from objective1 waypoint19)
	(visible_from objective1 waypoint20)
	(visible_from objective1 waypoint21)
	(visible_from objective1 waypoint22)
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
	(visible_from objective3 waypoint0)
	(visible_from objective3 waypoint1)
	(visible_from objective3 waypoint2)
	(visible_from objective3 waypoint3)
	(visible_from objective3 waypoint4)
	(visible_from objective3 waypoint5)
	(visible_from objective3 waypoint6)
	(visible_from objective3 waypoint7)
	(visible_from objective3 waypoint8)
	(visible_from objective3 waypoint9)
	(visible_from objective4 waypoint0)
	(visible_from objective4 waypoint1)
	(visible_from objective4 waypoint2)
	(visible_from objective4 waypoint3)
	(visible_from objective4 waypoint4)
	(visible_from objective5 waypoint0)
	(visible_from objective5 waypoint1)
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
)

(:goal (and
(communicated_rock_data waypoint6)
(communicated_rock_data waypoint8)
(communicated_rock_data waypoint20)
(communicated_rock_data waypoint13)
(communicated_image_data objective1 low_res)
(communicated_image_data objective0 low_res)
(communicated_image_data objective0 high_res)
(communicated_image_data objective7 low_res)
(communicated_image_data objective2 colour)
(communicated_image_data objective6 low_res)
	)
)
)