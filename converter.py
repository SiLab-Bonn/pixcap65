import tables as tb

if __name__ == "__main__":
    with tb.open_file("Data/New_1_Initial_6_Scan.h5", "a") as h5_file:
        group = h5_file.root["ATLAS ITk"].C_V_Characteristic.biasing.measurements
        group.HistCurr.attrs["Units"] = "A"
        group.HistCurrErr.attrs["Units"] = "A"
        # group.HistCurrValues.attrs["Units"] = "A"
        # group._f_setattr("current_unit", "A")
        # group._f_setattr("freq_unit", "MHz")
        group._f_setattr("bias_current_unit", "A")
        group._f_setattr("bias_voltage_unit", "V")
        group._f_setattr("bias_unit", "V")
        group.BiasVoltageHist.attrs["Units"] = "V"
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
                # subgroup.HistCurrValues.attrs["Units"] = "A"