export OUTPUT_DIR='../../experiments/albert_large_meh'
export RAW_DATA='../data/raw_valid_split2.json'
export TRAIN_DATA='../data/gen_train.json'
export BS=16
export CONTEXT_COL='sentence1'
export TO_PREDICT_COL='sentence2'
export MARKER_COL='marker'
export EPOCHS=4
export WARMUP_STEPS=4000

# gen_valid_M7                       ; CTRL7   82.34                               new: 86.08 (Al 1)
# gen_valid_M7_M8                    ; CTRL8   76.53, 77.6, 78.86                  new: 80.03 (AL 2)
# M9, temp = 0.88
# gen_valid_M7_M8_M9                 ; CTRL9   80.23, 81.66, 82.9                  new: 84.8, 84.8, 86 (AL 3)
# gen_valid_M7_M8_M9_M10             ; CTRL10  63.2                                new: 67, 68, 68.22   (AL 4)
# gen_valid_M7_M8_M9_M10_M11         ; CTRL11                                      new: 79.69, 80.33, 82      (AL5)
# gen_valid_M7_M12                   ; CTRL12                                      new: 82.49, 82.5, 83.52   (RL1)       90.89 on gen_valid
# gen_valid_M7_M12_M4                ; CTRL4                                       new: 80.3, 81, 82.2       (RL2)
# gen_valid_M7_M12_M4_M5             ; CTRL5                                            78.6, 79.7, 81.1     (RL3)
# length_penalty = 0.8
# gen_valid_M7_M12_M4_M6             ; CTRL6                                       new: 77.1, 77.9, 79.7     (RL4)
# gen_valid_M7_M12_M4_M7             ; CTRL7                                       new: 76.7, 78.2, 78.9     (RL5)
# gen_valid_M7_M12_M4_M8             ; CTRL8                                       new: 74.9, 77.06, 77.4    (RL6)
# temp = 1, top_p = 0.98
# gen_valid_M7_M12_M4_M9             ; CTRL9                                       new: 74.6, 76.39, 76.6    (RL7)
# gen_valid_M7_12_M4_M10             ; CTRL10                                      new: 72.2, 74.7, 74.6     (RL8)
# gen_valid_M7_M12_M4_M11            ; CTRL11                                      new: 70.73, 70.9, 72.6    (RL9)
# gen_valid_M7_12_M4_M12             ; CTRL12                                      new: 70.09, 71.06, 70.6   (RL10)
# gen_valid_M7_M12_M4_M12_M1         ; CTRL1                                       new: 67.6, 67.79, 66.7    (RL11)
# gen_valid_M7_M12_M4_M12_M1_M2      ; CTRL2                                       new: 66.5, 68, 67.5       (RL12)
# gen_valid_M7_M12_M4_M12_M1_M3      ; CTRL3                                       new: 64.5, 66.9, 66.54    (RL13)
# gen_valid_M7_M12_M4_M12_M1_M4      ; CTRL4                                       new: 61.9, 65.1, 64.98    (RL14)
# gen_valid_M7_M12_M4_M12_M1_M5      ; CTRL5                                       new: 61.73, 63.4, 63.1    (RL16)
# gen_valid_M7_M12_M4_M12_M1_M6      ; CTRL6                                       new: 59.9, 62.86, 62.06   (RL17)
# gen_valid_M7_M12_M4_M12_M1_M7      ; CTRL7                                       new: 59.3, 61.06, 61.6    (RL18)
# gen_valid_M7_M12_M4_M12_M1_M8      ; CTRL8                                       new: 59.6, 61.19, 59.9    (RL19)
# gen_valid_M7_M12_M4_M12_M1_M9      ; CTRL9                                       new: 57, 61.5, 61         (RL20)
# gen_valid_M7_M12_M4_M12_M1_M10     ; CTRL10                                      new: 58.33, 62.0, 59.9    (RL21)
# gen_valid_M7_M12_M4_M12_M1_M11     ; CTRL11                                      new: 58.66, 60.73, 60.9   (RL22)
# gen_valid_M7_M12_M4_M12_M1_M12     ; CTRL12                                      new: 58.06, 59.2, 60.06   (RL23)
# gen_valid_M7_M12_M4_M12_M1_M13     ; CTRL13                                      new: 56.2, 58.9, 59.9     (RL25)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1 ; fCTRL1                                      new: 55.6, 59, 59.2       (RL26)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1  ; CTRL1                                   new: 55.6, 57.1, 58.5     (RL27)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M2                  ; CTRL2                new: 55.5, 56.8, 56.9     (RL31)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3                  ; CTRL3                new: 55.0, 57.1, 56.9     (RL32)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2              ; fCTRL2               new: 53.7, 55.6, 56.31    (RL36)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4           ; CTRL4                new: 52.4, 56.4, 54.5     (RL37)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M5        ; CTRL5                new: 53.5, 55.1, 55.6     (RL38)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M6        ; CTRL6                new: 51.9, 54.8, 54.7     (RL39)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7        ; CTRL7                new: 51.5, 54.7, 53.3     (RL40)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13   ; CTRL13               new: 51.6, 54.4, 53.9     (RL41)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3; fCTRL3              new: 51.0, 52.4, 53.5     (RL42)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8 ; CTRL8           new: 50.6, 52.3, 53.2     (RL43)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M9; CTRL9         new: 49.7, 52.3, 52.7     (RL44)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10; CTRL10       new: 50.7, 51.8, 53.1     (RL45)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10_fM4; fCTRL4   new: 48.7, 53.7,  53.1    (RL46)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10_fM4_M11; CTRL11                    new: 51.39, 52.8, 53          (RL47)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10_fM4_M11_M12; CTRL12                new: 49.9, 52.3, 51.7      (RL48)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10_fM4_M11_M12_fM5; fCTRL5            new: 49.2, 51.7, 51.3      (RL49)
# gen_valid_M7_M12_M4_M12_M1_M13_fM1_M1_M3_fM2_M4_M7_iM13_fM3_M8_M10_fM4_M11_M12_fM5_M13; CTRL13        new: 47.6, 50.5, 50.3      (RL50)
# ckpt
# gen_valid_ckpt_fM6
# gen_valid_ckpt_fM6_M14                                48.3,  50.26, 49.6    (RL51)
# gen_valid_ckpt_fM6_M14_M4                             48.3,  49.16, 49.74   (RL52)
# gen_valid_ckpt_fM6_M14_M4_M5                          46.13, 49.13, 49.8    (RL53)
# gen_valid_ckpt_fM6_M14_M4_M6                          47.4,  50.03, 49.28   (RL54)
# gen_valid_ckpt_fM6_M14_M4_M7                          47.9,  50.36, 49.88   (RL55)
# gen_valid_ckpt_fM6_M14_M4_M8                          46,    49.7, 49.77    (RL56)
# gen_valid_ckpt_fM6_M14_M4_M8_M14                      46.3,  49.5, 50.13    (RL57)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9                   46.3,  48,   47.9     (RL58)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9_M10               44.9,  49.4, 48.8     (RL59)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9_M11               46.9,  49.8, 48.66    (RL60)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9_M12               46.23, 49.4, 48.94    (RL61)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9_M13               45.7,  48.7, 48.3     (RL62)
# gen_valid_ckpt_fM6_M14_M4_M8_M14_M9_M14               44.6,  47.9, 46.8     (RL63)
# pt
# gen_valid_pt_M4                                       46.2, 48.86, 48.32    (RL65)
# gen_valid_pt_M4_M5                                    45.3, 48.3, 46.87     (RL66)
# gen_valid_pt_M4_M6                                    44.9, 48.76, 48       (RL67)
# gen_valid_pt_M4_M7                                    43.9, 48.69, 47.6     (RL68)
# gen_valid_pt_M4_M8                                    44.2, 47.8, 47.44     (RL69)
# gen_valid_pt_M4_M9                                    44.8, 48.16, 47.44    (RL70)
# gen_valid_pt_M4_M10                                   45.4, 47.9, 46.46     (RL71)
# gen_valid_pt_M4_M11                                   43.5, 48.06, 46.34    (RL72)
# gen_valid_pt_M4_M12                                   45.6, 47.73, 48.09    (RL73)
# gen_valid_pt_M4_M13                                   43.4, 46.73, 47.12    (RL74)
# gen_valid_pt_M4_M14                                   46,   47.99, 46.85    (RL75)
# torch
# gen_valid_torch_M4                                                     44.5,   48.19, 46.64 (RL76)
# gen_valid_torch_M4_M5                                                  43.8,   47.2,  46.5  (RL77)
# gen_valid_torch_M4_M5_fM1                                              44.13,  47.06, 46.1  (RL78)
# gen_valid_torch_M4_M5_fM1_M6                                           43.2,   46.9,  45.5  (RL79)
# gen_valid_torch_M4_M5_fM1_M6_fM2                                       43.09,  45.6,  45.4  (RL80)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7                                    42.5,   46.43, 45.95 (RL81)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M8                                 41.4,   46.2,  45.98 (RL82)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9                                 41.99,  45.7,  45.3  (RL83)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3                             42.89,  45.7,  45.4  (RL84)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10                         43.3,   46.5,  45.38 (RL85)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11                     40.6,   44.2,  43.81 (RL86)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4                 41.2,   46.0,  45.05 (RL87)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12             41.43,  45.6,  45.2  (RL88)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5         41.96,  45.3,  44.8  (RL89)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13     40.83,  43.8,  44.2  (RL90)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6      42.3, 44.7,  44.3  (RL91)
# gen_valid_torch_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14  42, 44.13,  45.1  (RL92)
# flow
# gen_valid_flow_M4                                                   41.6 , 44.5, 44.11     (RL93)
# gen_valid_flow_M4_M5                                                41.93, 43.4, 44.2      (RL94)
# gen_valid_flow_M4_M5_fM1                                            40.7, 44.4, 43.45      (RL95)
# gen_valid_flow_M4_M5_fM1_M6                                         40.83, 43.73, 43.26    (RL96)
# gen_valid_flow_M4_M5_fM1_M6_fM2                                     41.13, 43.2, 43.24     (RL97)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7                                  40.04, 43.73, 43.15    (RL98)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M8                               41.83, 44.4, 43.24     (RL99)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9                               40.05, 43.6, 43.03     (RL100)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3                           41.46, 44.4, 44.6      (RL101)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10                       39.76, 42.23, 43.72    (RL102)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11                   40.76, 42.66, 43.88    (RL103)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4               40.43, 43.7,  43.10    (RL104)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12           40.59  , 43.63, 44.16    (RL105)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5       41.29  , 44,   42.76     (RL106)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13   40.33  , 44.5,  43.88     (RL107)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6   39.73, 43.33, 42.83        (RL108)
# gen_valid_flow_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14  39.59,  42.73, 42.69    (RL109)
# ignite
# gen_valid_ignite_M4  (RL110)                                         38.06, 42.03, 41.29
# gen_valid_ignte_M4_M5 (RL111)                                        40.23, 42.83, 42.39
# gen_valid_ignite_M4_M5_fM1 (RL112)                                   39.7, 43.43, 43.22
# gen_valid_ignite_M4_M5_fM1_M6 (RL113)                                39.46, 42.43, 42.8
# gen_valid_ignite_M4_M5_fM1_M6_fM2 (RL114)                            39.43, 42.33, 42.5
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7 (RL115)                         38.8, 40.86, 41.15
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M8 (RL116)                      39.13, 41.13, 41.31
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9 (RL117)                      38.73, 41.16, 40.78
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3 (RL118)                  38.5, 41.89, 40.99
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10 (RL119)              38.93, 41.06, 42.18
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11 (RL120)          37.99, 41.66, 41.337
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4 (RL121)      39.76, 42.3, 42.37
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12 (RL123)   38.1, 40.5, 42.41
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5 (RL124)       38.83, 40.36, 40.48
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13 (RL125)   37.73, 40.09, 38.78
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6 (RL126)   37.83, 41.36, 40.30
# gen_valid_ignite_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14 (RL127) 39.86, 40.73,
# light
# gen_valid_light_M4  (RL128)                                                       38.2,  40.93, 40.48
# gen_valid_light_M4_M5 (RL129)                                                     38.5,  42.16, 41.0
# gen_valid_light_M4_M5_fM1 (RL130)                                                 38.13, 40.9, 39.65
# gen_valid_light_M4_M5_fM1_M6 (RL131)                                              39.5,  40.59, 40.4
# gen_valid_light_M4_M5_fM1_M6_fM2 (RL132)                                          37.5,  40.56, 40.02
# gen_valid_light_M4_M5_fM1_M6_fM2_M7 (RL133)                                       39.13, 40.79, 40.92
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M8 (RL134)                                    37.06, 40.9, 40.4
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9 (RL135)                                    37.6 , 39.6, 39.9
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3 (RL136)                                39.76, 41.6, 40.92
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10 (RL137)                            36.53, 39.7, 39.24
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11 (RL138)                        38.6 , 41.6, 40.48
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4 (RL139)                    38.19 , 39.76, 40.02
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12 (RL140)                37.46, 40,    40.64
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5 (RL141)            38.13 , 40.43, 40.87
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13 (RL142)        35.69     , 40.23, 37.54
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6 (RL143)    36.39 , 38.73, 39.4
# gen_valid_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14 (RL144) 37.59 , 40.43, 39.77
# fire
# gen_valid_fire_M4  (RL145)                                                         36.59, 39.3,  39.29
# gen_valid_fire_M4_M5 (RL146)                                                       37.5,  40.7,  39.4
# gen_valid_fire_M4_M5_fM1 (RL147)                                                   36.8,  39.8,  38.32
# gen_valid_fire_M4_M5_fM1_M6 (RL148)                                                37.73, 39.16, 38.39
# gen_valid_fire_M4_M5_fM1_M6_fM2 (RL149)                                            35.86, 39,    38.76
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7 (RL150)                                         37.99, 39.1,  39.4
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M8 (RL151)                                      35.9,  39.13, 39
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9 (RL152)                                      36.2,  38.1,  38.69
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3 (RL53)                                   38,    40.03, 39.5
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10 (RL154)                              35.1,  38.19, 38.12
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11 (RL156)                          37.6,  40.76, 39.22
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4 (RL157)                      36.5,  38.1,  38.87
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12 (RL158)                  36.1,  38.6,  39.52
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5 (RL159)              36.59, 38.99, 39.47
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13 (RL160)          34.6,  38.73, 36.44
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6 (RL161)      34.93, 37.3,  37.68
# gen_valid_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14 (RL162)  36.7,  39.6



# replace_one, run_inference_only --no_af
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_145'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_4'
export VALIDATION_DATA='../data/gen_valid_split2_light_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_146'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_5'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_148'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_1'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_149'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_6'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_150'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_2'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_151'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_7'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_152'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_8'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M8.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_153'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_9'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M8.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_154'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_3'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_155'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_10'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
# ##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_156'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_11'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
#################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_157'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_4'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_158'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_12'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_159'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_5'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_160'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_13'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_161'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_flipped_6'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
##################################################
export CLASSIFICATION_MODEL='/shared/mlrdir2/disk1/pxb190028/roberta_large_162'
export AUTOREGRESSIVE_MODEL='prajjwal1/ctrl_discovery_14'
export VALIDATION_DATA='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6.json'
export FILE_OUTPUT_PATH='../data/gen_valid_split2_fire_M4_M5_fM1_M6_fM2_M7_M9_fM3_M10_M11_fM4_M12_fM5_M13_fM6_M14.json'
echo "Creating "$FILE_OUTPUT_PATH; python3 run_af.py --replace_one --run_inference_only --classification_model_name_or_path $CLASSIFICATION_MODEL --autoregressive_model_name_or_path $AUTOREGRESSIVE_MODEL --raw_data_path $RAW_DATA --train_data_path $TRAIN_DATA --validation_data_path $VALIDATION_DATA --output_dir $OUTPUT_DIR --per_device_train_batch_size $BS  --per_device_eval_batch_size $((BS*8)) --num_train_epochs $EPOCHS --file_output_path $FILE_OUTPUT_PATH --context_col $CONTEXT_COL --to_predict_col $TO_PREDICT_COL --marker_col $MARKER_COL --fp16 --save_total_limit 1 --save_strategy epoch --evaluation_strategy epoch --warmup_steps $WARMUP_STEPS
