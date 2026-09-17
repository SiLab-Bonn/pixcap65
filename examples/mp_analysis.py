# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------

import datetime
import logging
import multiprocessing as mp
import time
from contextlib import contextmanager

from examples.full_analysis import r1_analysator, r13_analysator_second
from pixcap65.utility import synchronized_process_open_file

AUTHKEY_OUTPUT = "Fetch new authkey:"
SECOND_LABEL = " Second Try."
OLD_API = True

bare_correction_args = {
    "apply_correction": True,
    "bare_file": "packaged/data/Bare_Sample_05_Extended_Scan.h5",
    "bare_hdf_path": "Reference/Bare/unbiased_full/total_cap",
}
logger = logging.getLogger(__name__)

def get_name_appendix(name, api=OLD_API):
    if "full" in name and not api:
        return name.replace('full', 'full_model')
    else:
        return name + '_model'


def synchronize_full_model(file, reference, name, bias, p_lock, **kwargs):
    from tables import Group
    # perhaps we should refactor this function to be more general applicable?
    # in particular there are some inconsistencies in the naming scheme.
    unbiased_name = kwargs.get("unbiased_group", "unbiased_full")
    inter_unbiased_name = kwargs.get("inter_unbiased_group", "inter_unbiased_full")
    biased_name = kwargs.get("biased_group", "biased_{}_V_full")
    inter_biased_name = kwargs.get("inter_biased_group", "inter_biased_M_{}_V_full")
    add_extensions = kwargs.get("inter_pix_extension_active", False)

    biased_name = biased_name.format(bias)
    inter_biased_name = inter_biased_name.format(bias)

    inter_pixel_names = [
        inter_unbiased_name,
        inter_biased_name,
    ]

    if add_extensions:
        for add_on in ['sides', 'diagonals', 'tops']:
            inter_pixel_names.append("{}__{}".format(inter_unbiased_name, add_on))
            inter_pixel_names.append("{}__{}".format(inter_biased_name, add_on))

    if not kwargs.get('api', False):
        from warnings import warn
        warn("It is highly encouraged to change the implementation such that the new replacement api is used. "
             "This might require the adjustment of paths.", stacklevel=2)

    with synchronized_process_open_file(file, mode='a', lock=p_lock) as h5_file:
        reference_node = h5_file._get_or_create_path("/{}/{}".format(reference, name), create=False)
        assert isinstance(reference_node, Group)
        if unbiased_name in reference_node:
            h5_file.copy_node(where=reference_node,
                              newname=get_name_appendix(unbiased_name, api=kwargs.get("api", OLD_API)),
                              name=unbiased_name,
                              recursive=True, overwrite=True)
        if biased_name in reference_node:
            h5_file.copy_node(where=reference_node,
                              newname=get_name_appendix(biased_name, api=kwargs.get("api", OLD_API)),
                              name=biased_name,
                              recursive=True, overwrite=True)

        # handle the inter-pixel analysis
        for inter_pixel_name in inter_pixel_names:
            if inter_pixel_name in reference_node:
                h5_file.copy_node(where=reference_node,
                                  newname=get_name_appendix(inter_pixel_name, api=kwargs.get("api", OLD_API)),
                                  name=inter_pixel_name,
                                  recursive=True, overwrite=True)


def error_handler(exc):
    logger.error("While performing the analysis in multiple processes an error occured.", exc_info=exc)


@contextmanager
def processed_manager(**kwargs):
    import pixcap65.concurrency
    import gc
    with pixcap65.concurrency.get_context_manager(**kwargs) as manager_ctx:
        try:
            lock = manager_ctx.RLock()
            yield manager_ctx, lock
        finally:
            del lock
            gc.collect()

    gc.collect()


if __name__ == "__main__":
    # perhaps it is necessary to provide the different locks as arguments to the
    import pixcap65.concurrency

    start_time = time.time()

    process_handles = [
        # x1_analysator,
        # x2_analysator,
        # x5_analysator,
        # x6_analysator,
        # x7_analysator,
        # e1_analysator_second,
        r13_analysator_second,
        r1_analysator,
        # x4_analysator,
    ]

    with processed_manager() as (manager, tables_lock):
        print("started the primary manager")
        # necessary to connect to the manager from the additional processes correctly
        authkey = mp.current_process().authkey
        manager_args = {
            # "authkey": authkey,
            "address": manager.address,
        }
        print("The following authkey is in use:", authkey)
        print("Fetched the authkey from the manager:", manager._authkey)
        # bare_analysis_handler(tables_lock)
        # handle all the processes
        # this could not be transformed to process handled as we could not submit authkeys!
        processes = {process.__name__: mp.Process(
            target=process,
            name=process.__name__,
            args=(tables_lock, bare_correction_args,),
            kwargs=manager_args
        ) for process in process_handles}

        # bare_analysis_handler(tables_lock)

        for p in processes.values():
            p.start()

        while len(processes) > 0:
            remove_names = []
            for name, p in processes.items():
                p.join()
                if p.exitcode is None:
                    print("There was a process which has not terminated after join. The process is", name)
                else:
                    p.close()
                    remove_names.append(name)

            while len(remove_names) > 0:
                name = remove_names.pop()
                del processes[name]

        print(processes)
        print("Elapsed time: ", time.time() - start_time)

    import gc
    gc.collect()
    print("Finished collection!")
    with open('depletion_manager_information_??.txt', 'a') as f:
        date_obj = datetime.datetime.now()
        full_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        date = date_obj.strftime("%Y-%m-%d")
        time_str = date_obj.strftime("%H:%M:%S")
        print(date, mp.current_process().pid, time_str,
              mp.current_process().name, mp.current_process().authkey, "FINISH - MARK", file=f)
