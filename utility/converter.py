import tables as tb

def adjust_i_v_measurement(group, has_values=False):
    group.HistCurr.attrs["Units"] = "A"
    if "HistCurrErr" in group:
        group.HistCurrErr.attrs["Units"] = "A"
    else:
        print("WHY DOES NO ERROR HIST EXIST FOR THE I-V CURVE?????")
    if has_values:
        group.HistCurrValues.attrs["Units"] = "A"

    group._f_setattr("bias_current_unit", "A")
    group._f_setattr("bias_voltage_unit", "V")
    group._f_setattr("bias_unit", "V")
    group.BiasVoltageHist.attrs["Units"] = "V"


def adjust_c_v_measurement(group, iv_values=False, cv_values=False):
    adjust_i_v_measurement(group, has_values=iv_values)

    assert isinstance(group, tb.Group)
    for subgroup in group._v_groups.values():
        subgroup._f_setattr("bias_voltage_unit", "V")
        subgroup._f_setattr("bias_unit", "V")
        subgroup._f_setattr("freq_unit", "MHz")
        subgroup._f_setattr("current_unit", "A")
        if "HistCurr" not in subgroup:
            print(subgroup)
            print("No entries?")
        else:
            subgroup.HistCurr.attrs["Units"] = "A"
            subgroup.HistCurrErr.attrs["Units"] = "A"
            if cv_values:
                subgroup.HistCurrValues.attrs["Units"] = "A"


def adjust_cap_measurement(group, has_values=False):
    group.HistCurr.attrs["Units"] = "A"
    group.HistCurrErr.attrs["Units"] = "A"
    if has_values:
        group.HistCurrValues.attrs["Units"] = "A"
    group._f_setattr("current_unit", "A")
    group._f_setattr("freq_unit", "MHz")


# FIXME: Why are there no error estimations for I-V curves?

if __name__ == "__main__":
    with tb.open_file("../New_2_Scan.h5", "a") as h5_file:
        adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.unbiased_1.total_cap.measurements, has_values=True)
        adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.biased_80_V.total_cap.measurements, has_values=True)
        adjust_i_v_measurement(h5_file.root.ATLAS_Itk.X2.I_V_Characteristic.biasing.measurements)
        adjust_c_v_measurement(h5_file.root.ATLAS_Itk.X2.C_V_Characteristic.biasing.measurements)


        # group = h5_file.root.ATLAS_Itk.X2.unbiased_1.measurements
        # group.HistCurr.attrs["Units"] = "A"
        # group.HistCurrErr.attrs["Units"] = "A"
        # group.HistCurrValues.attrs["Units"] = "A"
        # group._f_setattr("current_unit", "A")
        # group._f_setattr("freq_unit", "MHz")
        # group._f_setattr("bias_current_unit", "A")
        # group._f_setattr("bias_voltage_unit", "V")
        # group._f_setattr("bias_unit", "V")
        # group.BiasVoltageHist.attrs["Units"] = "V"
        # assert isinstance(group, tb.Group)
        # for subgroup in group._v_groups.values():
        #     subgroup._f_setattr("bias_voltage_unit", "V")
        #     subgroup._f_setattr("bias_unit", "V")
        #     subgroup._f_setattr("freq_unit", "MHz")
        #     subgroup._f_setattr("current_unit", "A")
        #     if "HistCurr" not in subgroup:
        #         print(subgroup)
        #         print("No entries?")
        #     else:
        #         subgroup.HistCurr.attrs["Units"] = "A"
        #         subgroup.HistCurrErr.attrs["Units"] = "A"
        #         # subgroup.HistCurrValues.attrs["Units"] = "A"