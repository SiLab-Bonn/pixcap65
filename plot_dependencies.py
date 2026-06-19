import numpy as np
import tables as tb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from iminuit import Minuit
from iminuit.cost import LeastSquares

from pixcap65.analysis_util.summary import field_names
from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR
from pixcap65.utility.homogenize_plots import set_params
from scipy.stats.distributions import chi2 as sample_chi2

def chi2(x: float, dof: int):
    return sample_chi2.cdf(x, dof)

interesting_keys = [
    "implantation_area",
    "implantation_depth",
    "pixel_separation_x",
    "pixel_separation_y",
]

# format the table output
GENERAL_SIUNITX_FORMAT = "\\qty{{{:#0.6g}({:d})({:d})({:d})}}{{{}}}"
GENERAL_SIUNITX_FORMAT_2 = "\\qty{{{:.6f}({:.6f})({:.6f})({:.6f})}}{{{}}}"

GENERAL_PART = "\\qty{{{{{{:#0.{}g}}({{:d}})({{:d}})({{:d}})}}}}{{{{{{}}}}}}"
REDUCED_GENERAL_PART = "{{:#0.{}g}}({{:d}})({{:d}})({{:d}})"
SECOND_GENERAL_PART = "{{:#0.{}f}}({{:d}})({{:d}})({{:d}})"

GENERAL_PREC = 6
GENERAL_TEST_FORMAT = "{:-6g}"
GENERAL_UNCERT_MULTIPLIER = 1e6



def generate_siunitx(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        test_format = "{:.3g}".format(data_set.dep_voltage_err)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.dep_voltage,
                                                     int(data_set.dep_voltage_err * 10 ** n_digits),
                                                     int(data_set.dep_voltage_systematic_general * 10 ** n_digits),
                                                     int(data_set.dep_voltage_systematic_dispersion * 10 ** n_digits),)


        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.dep_voltage)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.dep_voltage,
        #                                     int(data_set.dep_voltage_err * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.dep_voltage_systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.dep_voltage_systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\volt")
    else:
        if np.isnan(data_set.capacitance):
            return "-2"

        test_format = "{:.3g}".format(data_set.capacitance_err)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.capacitance,
                                                           int(data_set.capacitance_err * 10 ** n_digits),
                                                           int(data_set.capacitance_systematic_general * 10 ** n_digits),
                                                           int(data_set.capacitance_systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.capacitance)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.capacitance,
        #                                        int(data_set.capacitance_err * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.capacitance_systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.capacitance_systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\femto\\farad")


def generate_siunitx_2(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.dep_voltage,
                                             data_set.dep_voltage_err,
                                             data_set.dep_voltage_systematic_general,
                                             data_set.dep_voltage_systematic_dispersion, "\\volt")
    else:
        if np.isnan(data_set.capacitance):
            return "-2"
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.capacitance,
                                             data_set.capacitance_err,
                                             data_set.capacitance_systematic_general,
                                             data_set.capacitance_systematic_dispersion, "\\femto\\farad")

def read_rec_array(table: tb.Table, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read(*args, **kwargs), dtype=table.dtype)


def read_rec_array_sorted(table: tb.Table, primary_key, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_sorted(primary_key, *args, **kwargs), dtype=table.dtype)


def read_rec_array_where(table: tb.Table, condition, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_where(condition, *args, **kwargs), dtype=table.dtype)

def cap_model(xy, a0, a1, a2, a3, a4):
    A, W = xy
    return a0 + a1 * A + a2 * W + a3 / W + a4 * A * W

def simplified_cap_model(xy, a0, a1, a2):
    A, W = xy
    return a0 + a1 * A + a2 * W


def extended_cap_model(xy, a0, a1, a2, a3, a4, a5):
    A, W, separation = xy
    return a0 + a1 * A + a2 * W + a3 / W + a4 * A * W + a5 * separation

if __name__ == "__main__":
    # we need to acquire all the data from the
    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=12, dpi=1200)
    with tb.open_file('conclude_summary.h5', mode='a') as h5_conclusion:
        summary_data = h5_conclusion.root.GeneralSummaryTable
        sensor_properties = h5_conclusion.root.SensorTypes
        assert isinstance(sensor_properties, tb.Table)
        final_properties = read_rec_array_sorted(sensor_properties, 'sensor')
        final_summary = read_rec_array_sorted(summary_data, 'sensor')
        assert isinstance(final_properties, np.recarray)

        with PdfPages("Dependencies.pdf") as pdf:
            physical_property = None
            for physical_property in interesting_keys:
                fig, ax = plt.subplots()
                ax.set_title("Dependence of the capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                mask = np.isfinite(final_summary.biased_capacitance.capacitance)
                ax.errorbar(final_properties[physical_property][mask], final_summary.biased_capacitance.capacitance[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the inter-pixel capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                mask = np.logical_and(np.isfinite(final_summary.biased_inter_capacitance.capacitance),
                                      final_summary.biased_inter_capacitance.capacitance >= 0)
                ax.errorbar(final_properties[physical_property][mask], final_summary.biased_inter_capacitance.capacitance[mask],
                            fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
                ax.set_ylabel("$U$ in \\unit{{\\volt}}")
                mask = np.isfinite(final_summary.depletion_voltage.dep_voltage)
                ax.errorbar(final_properties[physical_property][mask], final_summary.depletion_voltage.dep_voltage[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            if physical_property is None:
                print("No physical property was investigated at all.")

        with PdfPages("Dependencies-log.pdf") as pdf:
            physical_property = None
            for physical_property in interesting_keys:
                fig, ax = plt.subplots()
                ax.set_yscale('log')
                ax.set_title("Dependence of the capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                mask = np.isfinite(final_summary.biased_capacitance.capacitance)
                ax.errorbar(final_properties[physical_property][mask],
                            final_summary.biased_capacitance.capacitance[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the inter-pixel capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                ax.set_yscale('log')
                mask = np.logical_and(np.isfinite(final_summary.biased_inter_capacitance.capacitance),
                                      final_summary.biased_inter_capacitance.capacitance >= 0)
                ax.errorbar(final_properties[physical_property][mask],
                            final_summary.biased_inter_capacitance.capacitance[mask],
                            fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
                ax.set_ylabel("$U$ in \\unit{{\\volt}}")
                ax.set_yscale('log')
                mask = np.isfinite(final_summary.depletion_voltage.dep_voltage)
                ax.errorbar(final_properties[physical_property][mask],
                            - final_summary.depletion_voltage.dep_voltage[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            if physical_property is None:
                print("No physical property was investigated at all.")

        with PdfPages("Dependencies-log-log.pdf") as pdf:
            physical_property = None
            for physical_property in interesting_keys:
                fig, ax = plt.subplots()
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_title("Dependence of the capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                mask = np.isfinite(final_summary.biased_capacitance.capacitance)
                ax.errorbar(final_properties[physical_property][mask],
                            final_summary.biased_capacitance.capacitance[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the inter-pixel capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                ax.set_yscale('log')
                ax.set_xscale('log')
                mask = np.logical_and(np.isfinite(final_summary.biased_inter_capacitance.capacitance),
                                      final_summary.biased_inter_capacitance.capacitance >= 0)
                ax.errorbar(final_properties[physical_property][mask],
                            final_summary.biased_inter_capacitance.capacitance[mask],
                            fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
                ax.set_ylabel("$U$ in \\unit{{\\volt}}")
                ax.set_yscale('log')
                ax.set_xscale('log')
                mask = np.isfinite(final_summary.depletion_voltage.dep_voltage)
                ax.errorbar(final_properties[physical_property][mask],
                            - final_summary.depletion_voltage.dep_voltage[mask], fmt='x')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            if physical_property is None:
                print("No physical property was investigated at all.")

        # it is also necessary to get a ND-Fit of our model for the capacitance distribution!
        print("Handle the standard model for capacitance's.")
        model = LeastSquares((final_properties.implantation_area[:-3], final_properties.implantation_depth[:-3]),
                             final_summary.biased_capacitance.capacitance[:-3],
                             final_summary.biased_capacitance.capacitance_err[:-3],
                             cap_model)
        m = Minuit(model, 2, 0.005, 0.16, 0, 0)
        m.migrad()
        m.hesse()
        print(m.fmin)
        print(m.values)
        print(m.errors)
        print(m.fmin.fval)
        print(m.ndof)
        print("p-value", 1-chi2(m.fmin.fval, m.ndof))

        print("Handle the simplified model for capacitance's.")
        model = LeastSquares((final_properties.implantation_area, final_properties.implantation_depth),
                             final_summary.biased_capacitance.capacitance,
                             final_summary.biased_capacitance.capacitance_err,
                             simplified_cap_model)
        m = Minuit(model, 2, 0.005, 0.16)
        m.migrad()
        m.hesse()
        m.minos()
        print(m.fmin)
        print(str(m.values))
        print(m.errors)
        print(m.fmin.fval)
        print(m.ndof)
        print(m.params)
        print("Investigate the resiuduals")
        prediction = np.array([simplified_cap_model(pair, **m.values.to_dict()) for pair in zip(final_properties.implantation_area, final_properties.implantation_depth)])
        print("p-value", 1 - chi2(m.fmin.fval, m.ndof))
        print(np.abs(final_summary.biased_capacitance.capacitance - prediction)[:-3])
        print(final_summary.sensor[:-3])
        print(prediction[-4])
        print(prediction[-7])


        # generate a latex table of the test capacitances:
        test_table = h5_conclusion.root.TestCap
        test_properties = read_rec_array_sorted(test_table, 'Sensor')

        with open("tab_test_capacitance.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            header_output = "{\\(\\text{sensor}\\)}"
            for key in field_names:
                print("S[table_format=3.3(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                      **print_args)
                header_output += " & {{\\(C_\\text{{{}}}\\)}}".format(key.removeprefix('test_').split('_')[0])

            header_output += "\\\\"
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print(header_output, **print_args)
            print("\\midrule", **print_args)
            sensor_dict = {}
            for record in test_properties:
                # ignore any filtering for now and it get arbitraryly large
                actual_a_output = record.Sensor.decode().replace('_', "\\_")
                for key in field_names:
                    if key not in sensor_dict:
                        sensor_dict[key] = key.removeprefix('test_').split('_')[0]
                    actual_a_output += " & {:.6g}\\pm{:.6g}".format(record["test_" + key] * CAPACITANCE_CONVERSION_FACTOR, record["test_" + key + "_error"] * CAPACITANCE_CONVERSION_FACTOR)
                    sensor_dict[key] += " & {:.6g}\\pm{:.6g}".format(record["test_" + key] * CAPACITANCE_CONVERSION_FACTOR, record["test_" + key + "_error"] * CAPACITANCE_CONVERSION_FACTOR)

                print(actual_a_output, **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        with open("tab_test_capacitance_trans.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            header_output = "{\\(\\text{sensor}\\)}"
            for key in test_properties.Sensor:
                print("S[table_format=3.3(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                      **print_args)
                header_output += " & {{\\(C_\\text{{{}}}\\)}}".format(key.decode().replace('_', "\\_"))

            header_output += "\\\\"
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print(header_output, **print_args)
            print("\\midrule", **print_args)
            for item in sensor_dict.values():
                print(item, **print_args)
            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the biased capacitance latex table
        print("Create the tables to use.")
        with open("tab_capacitance_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=2.4(4)(4)(4), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                  **print_args)
            print("S[table-format=2.4(4)(4)(4), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "{\\(C_\\text{biased}\\)}", "&", "{\\(C_\\text{inter}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for record in final_summary:
                print(record.sensor.decode().replace('_', "\\_"), "&",
                      generate_siunitx(record.biased_capacitance, False), "&",
                      generate_siunitx(record.biased_inter_capacitance, False), "&",
                      "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the planar depletion voltages latex table
        with open("tab_depletion_planar_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=2.4(4)(4)(4), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "{\\(U_\\text{depletion}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() in ("X5", "X6","X7", "X8", "X3", "X4"):
                    continue
                print(sensor.decode().replace('_', '\\_'), "&", generate_siunitx(dep_set, True), "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the 3d depletion voltages latex table
        with open("tab_depletion_3d_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=2.4(4)(4)(4), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{{Sensor}}\\)}", "&", "{\\(U_\\text{depletion}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() not in ("X5", "X6", "X7", "X8", "X3", "X4"):
                    continue
                print(sensor.decode().replace('_', '\\_'), "&", generate_siunitx(dep_set, True), "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

