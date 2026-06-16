import numpy as np
import tables as tb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

interesting_keys = [
    "implantation_area",
    "implantation_depth",
    "pixel_separation_x",
    "pixel_separation_y",
]

def read_rec_array(table: tb.Table, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read(*args, **kwargs), dtype=table.dtype)


def read_rec_array_sorted(table: tb.Table, primary_key, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_sorted(primary_key, *args, **kwargs), dtype=table.dtype)


def read_rec_array_where(table: tb.Table, condition, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_where(condition, *args, **kwargs), dtype=table.dtype)


if __name__ == "__main__":
    # we need to acquire all the data from the
    with tb.open_file('conclude_summary.h5', mode='a') as h5_conclusion:
        summary_data = h5_conclusion.root.GeneralSummaryTable
        sensor_properties = h5_conclusion.root.SensorTypes
        assert isinstance(sensor_properties, tb.Table)
        final_properties = read_rec_array_sorted(sensor_properties, 'sensor')
        final_summary = read_rec_array_sorted(summary_data, 'sensor')
        assert isinstance(final_properties, np.recarray)
        print(final_properties.shape)
        print(final_properties.dtype)
        print(final_properties)
        print(final_summary.shape)
        print(final_summary.dtype)
        print(final_summary)
        print(final_summary.biased_capacitance.capacitance)

        with PdfPages("Dependencies.pdf") as pdf:
            physical_property = None
            for physical_property in interesting_keys:
                fig, ax = plt.subplots()
                ax.set_title("Dependence of the capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                ax.plot(final_properties[physical_property], final_summary.biased_capacitance.capacitance)
                pdf.savefig(pdf, bbox_inches='tight')

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the inter-pixel capacitance on {}".format(physical_property))
                ax.set_ylabel("$C$ in \\unit{{\\femto\\farad}}")
                ax.plot(final_properties[physical_property], final_summary.biased_inter_capacitance.capacitance)
                pdf.savefig(pdf, bbox_inches='tight')

                fig, ax = plt.subplots()
                ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
                ax.set_ylabel("$U$ in \\unit{{\\volt}}")
                ax.plot(final_properties[physical_property], final_summary.depletion_voltage.dep_voltage)
                pdf.savefig(pdf, bbox_inches='tight')

            if physical_property is None:
                print("No physical property was investigated at all.")